#!/usr/bin/env python3
"""
Best-of-N Selection and Evaluation for Planner Model — TWO-JUDGE VARIANT

Adds --eval-judge-url so that BoN selection and the reported quality metrics
can be done by DIFFERENT judges. This is the eval setup for the perfect-only
judge ablation: select via the (broken) perfect-only judge, but report
metrics scored by the (full) FORTE judge so the ablation row is comparable
to the main FORTE row.

Quick mental model:
  --judge_url       = selection judge (BoN argmax)         [old default behavior]
  --eval-judge-url  = evaluation judge (reported metrics)  [new; defaults to --judge_url]

When the two URLs are DIFFERENT, the script automatically rich-scores `bon1`
(the lowest-temperature candidate) with the eval judge as well, so the
`bon_gain` metric is computed on rich-eval-judge scores at both ends. In
single-judge mode (default), behavior is byte-identical to the original
script for backward compatibility.

NEW output fields per example (when two-judge mode is active):
  best_judge_score                 → eval judge's rich score on the winner
  selection_judge_score_on_winner  → selection judge's fast score on the winner
  bon1_judge_score                 → eval judge's rich score on bon1 (was: fast)
  bon1_judge_source                → "eval_rich" or "selection_fast"
  judges_agree_on_winner           → both judges binary-agree on success(≥80)
  two_judge_mode                   → True if eval-judge-url != judge_url

Default temperature ladder: [0.2, 0.4, 0.6, 0.8, 1.0] for N=5 (FORTE main).

Usage (single-judge, unchanged from original):
    python best_of_n_selection.py \
        --planner-model /path/to/planner \
        --test-parquet /path/to/test.parquet \
        --n 5 --full \
        --judge_url http://localhost:8001/v1/chat/completions \
        --output results/run/results_full.json

Usage (TWO-JUDGE — perfect-only judge ablation):
    CUDA_VISIBLE_DEVICES=2 python best_of_n_two_judges.py \
        --planner-model ${FORTE_ROOT}/planner_rl/checkpoints_grpo_3b_perfectonly/global_step_120/actor \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --n 5 --full \
        --judge_url      http://localhost:8001/v1/chat/completions  \
        --eval-judge-url http://localhost:8002/v1/chat/completions  \
        --output results/ablation-perfectonly/results_full.json

STRUCTURAL EVAL CONVENTION (unchanged):
  For every error_type other than "none", the GT in the dataset is a
  deliberately-flawed plan and the *correct* model behavior is to diverge
  from it. ARTIFACT_ERROR_TYPES contains all 9 non-`none` error types: when
  scoring structurally, we remap the GT to the perfect plan for that query
  (looked up by query_id) so the model is rewarded for fixing the seeded
  error rather than reproducing it.
"""

import json
import re
import time
import argparse
import requests
import requests.adapters
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Prompt format — must exactly match SFT/RL training format ────────────────

SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)

ARTIFACT_ERROR_TYPES = {
    "circular_dependency",
    "forward_reference",
    "incomplete_plan",
    "inefficient_order",
    "missing_dependency",
    "parameter_typo",
    "type_mismatch",
    "unnecessary_steps",
    "wrong_tool",
}

DEFAULT_TEMPERATURE_LADDER: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]


def format_tools_for_prompt(tools: Dict[str, Any]) -> str:
    tools_str = "Available Tools:\n"
    unique_tools = {}
    for sub_question, tool_info in tools.items():
        tool_name = tool_info.get("name", sub_question)
        if tool_name not in unique_tools:
            unique_tools[tool_name] = tool_info
    for tool_name, tool_info in unique_tools.items():
        params = tool_info.get("parameters", {})
        properties = params.get("properties", {})
        required_params = params.get("required", [])
        param_parts = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            req_marker = " (required)" if param_name in required_params else ""
            param_parts.append(f"{param_name}: {param_type}{req_marker}")
        tools_str += f"- {tool_name}({', '.join(param_parts)})\n"
    return tools_str


def build_user_message(question: str, tools: Dict[str, Any]) -> str:
    tools_str = format_tools_for_prompt(tools)
    return (
        f"Generate a tool execution plan to answer this query.\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n"
        f"Generate a step-by-step plan using the available tools. Each step should:\n"
        f"1. Call exactly one tool\n"
        f"2. Use output variables {{{{0}}}}, {{{{1}}}}, {{{{2}}}}, etc. for results\n"
        f"3. Reference previous step outputs using {{{{N}}}} - never substitute a hardcoded value for an output that comes from a prior step\n"
        f"4. Use the exact parameter names shown in the tool signatures above\n"
        f"5. Provide all required parameters\n\n"
        f"Generate only the steps the query requires - no redundant steps, no missing steps.\n\n"
        f"Format each step as: Step N: {{{{N}}}} = tool_name(param1=value1, param2=value2, ...)"
    )


# ── Judge server client (per-URL session management) ─────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert judge for evaluating tool execution plans. Your task is to:
1. Analyze the plan's correctness and efficiency
2. Assign a quality score (0-100)
3. Predict success likelihood (yes/likely_yes/uncertain/likely_no/no)
4. Identify specific issues with severity levels
5. Provide detailed reasoning

Scoring guidelines:
- 100: Perfect execution, no errors
- 80-99: Minor issues, likely to succeed
- 60-79: Moderate issues, uncertain outcome
- 40-59: Major issues, likely to fail
- 0-39: Critical errors, will fail"""

# Persistent HTTP sessions, keyed by judge URL. Allows reusing TCP connections
# to each judge server independently when two judges are running concurrently.
_http_sessions: Dict[str, requests.Session] = {}


def _get_session(judge_url: str) -> requests.Session:
    sess = _http_sessions.get(judge_url)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=16,
            pool_maxsize=16,
            max_retries=0,
        )
        sess.mount("http://", adapter)
        sess.headers.update({"Connection": "keep-alive"})
        _http_sessions[judge_url] = sess
    return sess


def _format_tools_for_judge(tools: Dict[str, Any]) -> str:
    if not tools:
        return ""
    lines = ["Available Tools:"]
    unique_tools = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in unique_tools:
            unique_tools[name] = tool_info
    for tool_name, tool_info in unique_tools.items():
        props = tool_info.get("parameters", {}).get("properties", {})
        params_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {tool_name}({params_str})")
    return "\n".join(lines)


def score_plan_via_judge(judge_url: str, query: str, plan_str: str, tools: Dict,
                          max_tokens: int = 300, retries: int = 3) -> Dict[str, Any]:
    """Score a plan via the judge at `judge_url`. URL is the first arg so it
    composes cleanly with pool.submit() and partial application."""
    tools_str = _format_tools_for_judge(tools)
    user_content = f"""Query: {query}

{tools_str}

Plan to Evaluate:
{plan_str}

Please evaluate this plan and provide:
1. Quality score (0-100)
2. Success prediction (yes/likely_yes/uncertain/likely_no/no)
3. Detailed reasoning
4. List of issues (if any)
5. Confidence (0.0-1.0)

Format your response as JSON:
{{
  "quality_score": <int>,
  "success_prediction": "<string>",
  "reasoning": "<string>",
  "issues": [...],
  "confidence": <float>
}}"""

    payload = {
        "model": "judge",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    session = _get_session(judge_url)
    content = ""
    for attempt in range(retries):
        try:
            resp = session.post(judge_url, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            if "```json" in content:
                content = content[content.find("```json") + 7 : content.rfind("```")].strip()
            elif "```" in content:
                content = content[content.find("```") + 3 : content.rfind("```")].strip()
            if not content.endswith("}"):
                last = content.rfind("}")
                if last != -1:
                    content = content[:last + 1]

            annotation = json.loads(content)
            annotation["quality_score"] = max(0, min(100, int(annotation.get("quality_score", 50))))
            annotation["confidence"]    = max(0.0, min(1.0, float(annotation.get("confidence", 0.5))))
            annotation["_full_parse"]   = True
            annotation["_judge_url"]    = judge_url
            return annotation

        except (requests.ConnectionError, requests.Timeout):
            # Drop the cached session for this URL so the retry re-establishes
            _http_sessions.pop(judge_url, None)
            if attempt < retries - 1:
                time.sleep(2.0)
                session = _get_session(judge_url)
        except (json.JSONDecodeError, KeyError, ValueError):
            try:
                match = re.search(r'"quality_score"\s*:\s*(\d+)', content)
                if match:
                    return {
                        "quality_score":      max(0, min(100, int(match.group(1)))),
                        "success_prediction": "uncertain",
                        "reasoning":          "partial parse - increase --judge-eval-max-tokens",
                        "issues":             [],
                        "confidence":         0.5,
                        "_full_parse":        False,
                        "_judge_url":         judge_url,
                    }
            except Exception:
                pass
            break

    return {
        "quality_score": 0, "success_prediction": "no",
        "reasoning": "judge call failed", "issues": [], "confidence": 0.0,
        "_full_parse": False, "_judge_url": judge_url,
    }


# ── Plan parsing (unchanged from original) ────────────────────────────────────

def _is_nl_tool_name(name: str) -> bool:
    return len(name.split()) > 4 or name.endswith("?")


def _functional_tool_match(gen_name: str, gt_name: str) -> float:
    STOP = {"what", "is", "the", "of", "in", "a", "an", "and", "or", "to",
            "how", "many", "who", "which", "are", "was", "were", "be", "been",
            "at", "on", "for", "with", "that", "this", "it", "its", "from"}

    def keywords(s: str) -> set:
        words = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
        return {w for w in words if w not in STOP and len(w) > 2}

    gen_kw = keywords(gen_name)
    gt_kw  = keywords(gt_name)

    if not gen_kw or not gt_kw:
        return 0.0

    intersection = gen_kw & gt_kw
    union        = gen_kw | gt_kw
    jaccard      = len(intersection) / len(union)
    return round(jaccard, 3)


def parse_plan_steps(plan_text: str) -> List[Dict]:
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("Step "):
            continue
        try:
            step_match = re.match(r"Step (\d+):", line)
            if not step_match:
                continue
            step_id = int(step_match.group(1))

            var_match = re.search(r"(\{\{\d+\}\})\s*=", line)
            if not var_match:
                continue
            output_var = var_match.group(1)

            tool_match = re.search(r"=\s*([^\(]+)\((.*)\)\s*$", line)
            if not tool_match:
                tool_match_empty = re.search(r"=\s*([^\(]+)\(\)\s*$", line)
                if tool_match_empty:
                    tool_name = tool_match_empty.group(1).strip()
                    params = {}
                else:
                    continue
            else:
                tool_name = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                params = {}
                if params_str:
                    param_parts = []
                    current = ""
                    depth = 0
                    in_str = False
                    str_char = None
                    for ch in params_str:
                        if ch in ('"', "'") and (not in_str or ch == str_char):
                            in_str = not in_str
                            str_char = ch if in_str else None
                        if not in_str:
                            if ch in "([{":
                                depth += 1
                            elif ch in ")]}":
                                depth -= 1
                            elif ch == "," and depth == 0:
                                param_parts.append(current.strip())
                                current = ""
                                continue
                        current += ch
                    if current.strip():
                        param_parts.append(current.strip())
                    for part in param_parts:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            params[k.strip()] = v.strip()

            steps.append({
                "step_id":         step_id,
                "output_variable": output_var,
                "tool_name":       tool_name,
                "parameters":      params,
            })
        except Exception:
            continue
    return steps


def normalize_value(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())


# ── Structural evaluation (vs ground truth) — unchanged ───────────────────────

def _remap_gt_tool_name(nl_name: str, tools: Dict[str, Any]) -> str:
    if nl_name in tools:
        api_name = tools[nl_name].get("name")
        if api_name:
            return api_name
    for key, tool_info in tools.items():
        if nl_name in key or key in nl_name:
            api_name = tool_info.get("name")
            if api_name:
                return api_name
    return nl_name


def evaluate_plan_vs_gt(gen_steps: List[Dict], gt_steps: List[Dict],
                        tools: Dict[str, Any] = None) -> Dict:
    empty = {
        "valid": False, "error": "",
        "step_count_match": False,
        "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
        "param_accuracy": 0.0, "dependency_accuracy": 0.0,
        "exact_match": False, "functional_match": False, "param_only_match": False,
        "gt_uses_nl_tool_names": False, "step_details": [],
    }

    if not gen_steps:
        return {**empty, "error": "no steps generated"}
    if not gt_steps:
        return {**empty, "error": "no ground truth steps"}

    gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)

    gen_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gen_steps)
    if gt_uses_nl and tools and not gen_uses_nl:
        gt_steps = [
            {**s, "tool_name": _remap_gt_tool_name(s["tool_name"], tools)}
            for s in gt_steps
        ]
        gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)

    step_count_match = len(gen_steps) == len(gt_steps)
    correct_exact_tools = 0
    total_functional    = 0.0
    total_params_correct = 0
    total_params = 0
    correct_deps = 0
    total_deps   = 0
    step_details = []

    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt  = gt_steps[i]  if i < len(gt_steps)  else None
        detail: Dict[str, Any] = {"step_id": i}

        if gen and gt:
            exact_tool_ok = gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower()
            detail["exact_tool_correct"] = exact_tool_ok
            if exact_tool_ok:
                correct_exact_tools += 1

            func_score = _functional_tool_match(gen["tool_name"], gt["tool_name"])
            detail["functional_tool_score"] = func_score
            total_functional += func_score

            gt_keys  = set(gt["parameters"].keys())
            gen_keys = set(gen["parameters"].keys())
            common   = gt_keys & gen_keys
            missing  = list(gt_keys - gen_keys)
            extra    = list(gen_keys - gt_keys)
            incorrect = []
            for k in common:
                gt_v  = normalize_value(gt["parameters"][k])
                gen_v = normalize_value(gen["parameters"][k])
                if gt_v == gen_v or gt_v in gen_v or gen_v in gt_v:
                    total_params_correct += 1
                else:
                    incorrect.append({"param": k,
                                      "generated": gen["parameters"][k],
                                      "ground_truth": gt["parameters"][k]})
            total_params += len(gt_keys)
            detail["param_comparison"] = {
                "total_gt_params": len(gt_keys),
                "correct": len(common) - len(incorrect),
                "missing": missing, "extra": extra, "incorrect": incorrect,
            }

            gt_refs  = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            total_deps  += len(gt_refs)
            correct_deps += len(gt_refs & gen_refs)
            detail["dependency_refs_match"] = (gt_refs == gen_refs)

        else:
            detail["exact_tool_correct"] = False
            detail["functional_tool_score"] = 0.0
            detail["param_comparison"] = None
            detail["dependency_refs_match"] = False

        step_details.append(detail)

    n_gt = len(gt_steps)
    exact_tool_acc    = correct_exact_tools / n_gt
    functional_acc    = total_functional / n_gt
    param_accuracy    = total_params_correct / total_params if total_params > 0 else 0.0
    dep_accuracy      = correct_deps / total_deps if total_deps > 0 else 1.0

    exact_match      = step_count_match and exact_tool_acc == 1.0 and param_accuracy == 1.0
    functional_match = step_count_match and functional_acc >= 0.5 and param_accuracy >= 0.5
    param_only_match = param_accuracy >= 0.5

    return {
        "valid":                    True,
        "gt_uses_nl_tool_names":    gt_uses_nl,
        "step_count_match":         step_count_match,
        "generated_steps":          len(gen_steps),
        "ground_truth_steps":       n_gt,
        "exact_tool_accuracy":      exact_tool_acc,
        "functional_tool_accuracy": functional_acc,
        "param_accuracy":           param_accuracy,
        "dependency_accuracy":      dep_accuracy,
        "exact_match":              exact_match,
        "functional_match":         functional_match,
        "param_only_match":         param_only_match,
        "step_details":             step_details,
    }


# ── Planner model (unchanged) ─────────────────────────────────────────────────

class PlannerModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading planner model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"OK Planner loaded on {self.device}")

    def _build_prompt(self, question: str, tools: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(question, tools)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        ).to(self.device)

    def _generate_one(self, inputs: Dict[str, torch.Tensor],
                      temperature: float, max_new_tokens: int = 512,
                      top_p: float = 0.9) -> str:
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                top_p=top_p if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        )

    def generate_plan(self, question: str, tools: Dict[str, Any],
                      temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        inputs = self._build_prompt(question, tools)
        return self._generate_one(inputs, temperature, max_new_tokens)

    def generate_n_plans(
        self,
        question: str,
        tools: Dict[str, Any],
        temperatures: List[float],
        max_new_tokens: int = 512,
    ) -> Tuple[List[str], List[float]]:
        n = len(temperatures)
        if n == 0:
            raise ValueError("temperatures must contain at least one value")

        inputs = self._build_prompt(question, tools)
        all_equal = all(abs(t - temperatures[0]) < 1e-9 for t in temperatures)

        if all_equal:
            t = temperatures[0]
            input_ids      = inputs["input_ids"].expand(n, -1)
            attention_mask = inputs["attention_mask"].expand(n, -1)
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=t,
                        do_sample=t > 0.0,
                        top_p=0.9 if t > 0.0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                plans = [
                    self.tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
                    for i in range(n)
                ]
                return plans, list(temperatures)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        plans: List[str] = []
        for t in temperatures:
            try:
                plan = self._generate_one(inputs, t, max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                plan = ""
            plans.append(plan)
        return plans, list(temperatures)


# ── Data loading (unchanged) ──────────────────────────────────────────────────

def load_test_parquet(parquet_path: str, perfect_only: bool = False) -> Tuple[List[Dict], Dict]:
    import pyarrow.parquet as pq

    table         = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    examples      = []

    for i in range(len(extra_infos)):
        extra_info = extra_infos[i]
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        if not isinstance(extra_info, dict):
            extra_info = {}

        reward_model = reward_models[i]
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        if not isinstance(reward_model, dict):
            reward_model = {}

        data_json = extra_info.get("data_json", "{}")
        if isinstance(data_json, str):
            data_json = json.loads(data_json)
        if not isinstance(data_json, dict):
            data_json = {}

        error_type    = str(extra_info.get("error_type", "none"))
        quality_score = int(extra_info.get("quality_score", 0))

        if perfect_only and not (error_type == "none" and quality_score >= 100):
            continue

        examples.append({
            "question":      data_json.get("question", ""),
            "tools":         data_json.get("tools", {}),
            "ground_truth":  reward_model.get("ground_truth", ""),
            "error_type":    error_type,
            "quality_score": quality_score,
            "query_id":      extra_info.get("query_id", -1),
        })

    perfect_gt_by_qid = {
        ex["query_id"]: ex["ground_truth"]
        for ex in examples
        if ex["error_type"] == "none" and ex["quality_score"] >= 100
    }

    return examples, perfect_gt_by_qid


# ── Best-of-N evaluation (pipelined, two-judge aware) ────────────────────────

def _score_candidates_parallel(pool: ThreadPoolExecutor, judge_url: str,
                                question: str, candidates: List[str],
                                tools: Dict, max_tokens: int) -> List[Dict]:
    futures = {
        pool.submit(score_plan_via_judge, judge_url, question, plan, tools, max_tokens): idx
        for idx, plan in enumerate(candidates)
    }
    results = [None] * len(candidates)
    for future in as_completed(futures):
        results[futures[future]] = future.result()
    return results


def _pick_best(candidates: List[str], fast_scores: List[Dict]) -> Tuple[int, str]:
    max_quality = max(s["quality_score"] for s in fast_scores)
    tied = [
        (i, s, candidates[i]) for i, s in enumerate(fast_scores)
        if s["quality_score"] == max_quality
    ]
    if len(tied) == 1:
        best_idx, _, best_plan = tied[0]
    else:
        best_idx, _, best_plan = max(
            tied, key=lambda x: (x[1].get("confidence", 0.5), -len(x[2]))
        )
    return int(best_idx), best_plan


def _build_result(ex: Dict, candidates: List[str], cand_temperatures: List[float],
                  fast_scores: List[Dict],
                  best_idx: int, best_plan: str,
                  eval_winner_score: Dict,
                  eval_bon1_score: Optional[Dict],
                  gt_steps: List[Dict], structural_gt_steps: List[Dict],
                  n: int, two_judge_mode: bool,
                  return_all_candidates: bool) -> Dict:
    question = ex["question"]
    tools    = ex["tools"]

    best_steps  = parse_plan_steps(best_plan)
    struct_eval = evaluate_plan_vs_gt(best_steps, structural_gt_steps, tools=tools)

    # judge_success and best_judge_score always come from the EVAL judge's rich call.
    judge_success    = eval_winner_score["quality_score"] >= 80
    best_judge_score = eval_winner_score["quality_score"]

    # bon1: in two-judge mode, use eval-judge rich score; otherwise fall back to
    # the fast-scored value (backward-compatible with the original script).
    if two_judge_mode and eval_bon1_score is not None:
        bon1_judge_score  = eval_bon1_score["quality_score"]
        bon1_judge_source = "eval_rich"
        bon1_full_parse   = eval_bon1_score.get("_full_parse", False)
    else:
        bon1_judge_score  = fast_scores[0]["quality_score"]
        bon1_judge_source = "selection_fast"
        bon1_full_parse   = fast_scores[0].get("_full_parse", False)

    # Selection judge's fast view of the winner — useful diagnostic.
    selection_judge_score_on_winner = fast_scores[best_idx]["quality_score"]
    selection_judges_winner_success = selection_judge_score_on_winner >= 80
    judges_agree_on_winner          = selection_judges_winner_success == judge_success

    ref_error_type = ex["error_type"]
    ref_is_perfect = (ref_error_type == "none" and ex["quality_score"] >= 100)

    if ref_is_perfect:
        error_type_handled = judge_success
    else:
        error_type_handled = best_judge_score >= ex["quality_score"]

    judge_agrees_with_ref = (ref_is_perfect == judge_success)

    result = {
        "query_id":              ex["query_id"],
        "question":              question,
        "error_type":            ref_error_type,
        "ref_quality_score":     ex["quality_score"],
        "ref_is_perfect":        ref_is_perfect,
        "ground_truth":          ex["ground_truth"],

        "best_plan":             best_plan,
        "best_candidate_idx":    best_idx,
        "best_candidate_temperature": (cand_temperatures[best_idx]
                                        if 0 <= best_idx < len(cand_temperatures)
                                        else None),

        # Reported metrics — scored by EVAL judge
        "judge_success":         judge_success,
        "best_judge_score":      best_judge_score,
        "best_success":          eval_winner_score["success_prediction"],
        "best_confidence":       eval_winner_score["confidence"],
        "judge_full_parse":      eval_winner_score.get("_full_parse", False),

        # bon1 — eval-judge rich in two-judge mode, selection-judge fast otherwise
        "bon1_judge_score":      bon1_judge_score,
        "bon1_judge_source":     bon1_judge_source,
        "bon1_full_parse":       bon1_full_parse,
        "bon1_temperature":      cand_temperatures[0] if cand_temperatures else None,

        # Selection judge's view (diagnostic in two-judge mode)
        "selection_judge_score_on_winner": selection_judge_score_on_winner,
        "selection_judge_winner_success":  selection_judges_winner_success,
        "judges_agree_on_winner":          judges_agree_on_winner,
        "two_judge_mode":                  two_judge_mode,

        # All candidates' selection-judge fast scores (used for BoN argmax)
        "all_candidate_scores":  [s["quality_score"] for s in fast_scores],
        "all_candidate_temperatures": list(cand_temperatures),
        "mean_candidate_score":  float(np.mean([s["quality_score"] for s in fast_scores])),
        "candidate_score_std":   float(np.std([s["quality_score"] for s in fast_scores])),

        # Structural
        "gt_uses_nl_tool_names":      struct_eval["gt_uses_nl_tool_names"],
        "exact_match":                struct_eval["exact_match"],
        "functional_match":           struct_eval["functional_match"],
        "param_only_match":           struct_eval["param_only_match"],
        "step_count_match":           struct_eval["step_count_match"],
        "generated_n_steps":          struct_eval.get("generated_steps", 0),
        "gt_n_steps":                 struct_eval.get("ground_truth_steps", len(gt_steps)),
        "exact_tool_accuracy":        struct_eval["exact_tool_accuracy"],
        "functional_tool_accuracy":   struct_eval["functional_tool_accuracy"],
        "param_accuracy":             struct_eval["param_accuracy"],
        "dependency_accuracy":        struct_eval["dependency_accuracy"],

        "error_type_handled":         error_type_handled,
        "judge_agrees_with_ref":      judge_agrees_with_ref,
        "n_candidates":               n,
    }

    if return_all_candidates:
        result["all_candidates"] = [
            {"plan": p,
             "temperature": t,
             "judge_score": s["quality_score"],
             "success": s["success_prediction"],
             "functional_match": evaluate_plan_vs_gt(
                 parse_plan_steps(p), gt_steps, tools=tools)["functional_match"]}
            for p, t, s in sorted(
                zip(candidates, cand_temperatures, fast_scores),
                key=lambda x: x[2]["quality_score"], reverse=True
            )
        ]

    return result


def evaluate_best_of_n(
    planner: PlannerModel,
    examples: List[Dict],
    perfect_gt_by_qid: Dict[int, str] = None,
    n: int = 5,
    temperatures: Optional[List[float]] = None,
    selection_judge_url: str = "http://localhost:8001/v1/chat/completions",
    eval_judge_url: Optional[str] = None,
    max_new_tokens: int = 512,
    judge_max_tokens: int = 32,
    judge_eval_max_tokens: int = 300,
    return_all_candidates: bool = False,
) -> List[Dict]:
    """
    Pipelined Best-of-N evaluation.

    selection_judge_url: judge used for BoN argmax (fast call across N candidates).
    eval_judge_url:      judge used to score the winner (and bon1, in two-judge mode)
                         for reported metrics. Defaults to selection_judge_url.

    When the two URLs differ ("two-judge mode"), bon1 is rich-scored by the eval
    judge so that the bon_gain metric compares rich-vs-rich on the same judge.
    """
    if temperatures is None:
        temperatures = list(DEFAULT_TEMPERATURE_LADDER)
    if len(temperatures) != n:
        raise ValueError(
            f"len(temperatures)={len(temperatures)} but n={n}; they must match"
        )

    if eval_judge_url is None:
        eval_judge_url = selection_judge_url
    two_judge_mode = (eval_judge_url != selection_judge_url)

    print(f"Selection judge: {selection_judge_url}")
    print(f"Eval judge:      {eval_judge_url}")
    if two_judge_mode:
        print(f"  → Two-judge mode: BoN argmax by selection judge; reported "
              f"metrics by eval judge.")
        print(f"  → bon1 will be rich-scored by eval judge for clean bon_gain.")
    else:
        print(f"  → Single-judge mode (selection == eval).")

    results = []
    partial_parse_count = 0
    partial_parse_count_bon1 = 0
    pool = ThreadPoolExecutor(max_workers=n + 8)

    prev_candidates:        Optional[List[str]]   = None
    prev_cand_temps:        Optional[List[float]] = None
    prev_fast_scores:       Optional[List[Dict]]  = None
    prev_best_idx:          Optional[int]         = None
    prev_best_plan:         Optional[str]         = None
    prev_rich_winner_fut:   Optional[Future]      = None
    prev_rich_bon1_fut:     Optional[Future]      = None
    prev_ex:                Optional[Dict]        = None
    prev_gt_steps:          Optional[List[Dict]]  = None
    prev_struct_gt_steps:   Optional[List[Dict]]  = None

    def _get_structural_gt(ex, gt_steps):
        if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                and perfect_gt_by_qid
                and ex["query_id"] in perfect_gt_by_qid):
            return parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        return gt_steps

    def _finalize_previous():
        nonlocal partial_parse_count, partial_parse_count_bon1
        if prev_rich_winner_fut is None:
            return
        eval_winner_score = prev_rich_winner_fut.result()
        if not eval_winner_score.get("_full_parse", False):
            partial_parse_count += 1

        eval_bon1_score = None
        if prev_rich_bon1_fut is not None:
            eval_bon1_score = prev_rich_bon1_fut.result()
            if not eval_bon1_score.get("_full_parse", False):
                partial_parse_count_bon1 += 1

        result = _build_result(
            prev_ex, prev_candidates, prev_cand_temps, prev_fast_scores,
            prev_best_idx, prev_best_plan,
            eval_winner_score, eval_bon1_score,
            prev_gt_steps, prev_struct_gt_steps,
            n, two_judge_mode, return_all_candidates,
        )
        results.append(result)

    pbar = tqdm(examples, desc=f"Best-of-{n} evaluation (temps={temperatures})")

    for ex_idx, ex in enumerate(pbar):
        question     = ex["question"]
        tools        = ex["tools"]
        ground_truth = ex["ground_truth"]
        gt_steps     = parse_plan_steps(ground_truth)
        struct_gt    = _get_structural_gt(ex, gt_steps)

        # Phase 1: Generate candidates (GPU)
        candidates, cand_temps = planner.generate_n_plans(
            question, tools,
            temperatures=temperatures,
            max_new_tokens=max_new_tokens,
        )

        # Phase 2: Finalize previous example (collects async judge futures)
        _finalize_previous()

        # Phase 3: Fast-score all candidates with SELECTION judge (parallel)
        fast_scores = _score_candidates_parallel(
            pool, selection_judge_url, question, candidates, tools, judge_max_tokens
        )

        # Phase 4: Pick best by selection-judge fast scores
        best_idx, best_plan = _pick_best(candidates, fast_scores)

        # Phase 5: Rich-score winner with EVAL judge (async)
        rich_winner_fut = pool.submit(
            score_plan_via_judge, eval_judge_url,
            question, best_plan, tools, judge_eval_max_tokens
        )

        # Phase 6: In two-judge mode, also rich-score bon1 with EVAL judge (async)
        rich_bon1_fut: Optional[Future] = None
        if two_judge_mode:
            rich_bon1_fut = pool.submit(
                score_plan_via_judge, eval_judge_url,
                question, candidates[0], tools, judge_eval_max_tokens
            )

        prev_candidates       = candidates
        prev_cand_temps       = cand_temps
        prev_fast_scores      = fast_scores
        prev_best_idx         = best_idx
        prev_best_plan        = best_plan
        prev_rich_winner_fut  = rich_winner_fut
        prev_rich_bon1_fut    = rich_bon1_fut
        prev_ex               = ex
        prev_gt_steps         = gt_steps
        prev_struct_gt_steps  = struct_gt

        if results:
            recent = results[-10:]
            avg_score    = np.mean([r["best_judge_score"] for r in recent])
            success_rate = np.mean([r["judge_success"]    for r in recent])
            postfix = {
                "avg_score": f"{avg_score:.0f}",
                "success%":  f"{100*success_rate:.0f}",
            }
            if two_judge_mode:
                agree_rate = np.mean([r["judges_agree_on_winner"] for r in recent])
                postfix["agree%"] = f"{100*agree_rate:.0f}"
            pbar.set_postfix(postfix)

    _finalize_previous()
    pool.shutdown(wait=False)

    if partial_parse_count > 0:
        pct = 100 * partial_parse_count / len(results) if results else 0
        print(f"\n  WARN  Partial parse on winner rich calls: "
              f"{partial_parse_count}/{len(results)} ({pct:.1f}%)")
        if pct > 20:
            print("        Consider increasing --judge-eval-max-tokens (current may be too low)")
    if two_judge_mode and partial_parse_count_bon1 > 0:
        pct = 100 * partial_parse_count_bon1 / len(results) if results else 0
        print(f"  WARN  Partial parse on bon1 rich calls (eval judge): "
              f"{partial_parse_count_bon1}/{len(results)} ({pct:.1f}%)")

    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(results: List[Dict], label: str) -> Dict:
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}

    best_scores = [r["best_judge_score"]         for r in results]
    bon1_scores = [r["bon1_judge_score"]          for r in results]
    mean_scores = [r["mean_candidate_score"]      for r in results]
    func_tools  = [r["functional_tool_accuracy"]  for r in results]
    param_accs  = [r["param_accuracy"]            for r in results]
    dep_accs    = [r["dependency_accuracy"]       for r in results]

    judge_success_rate    = np.mean([r["judge_success"]      for r in results])
    error_handled_rate    = np.mean([r["error_type_handled"] for r in results])
    exact_match_rate      = np.mean([r["exact_match"]        for r in results])
    functional_match_rate = np.mean([r["functional_match"]   for r in results])
    param_only_match_rate = np.mean([r["param_only_match"]   for r in results])
    step_match_rate       = np.mean([r["step_count_match"]   for r in results])
    full_parse_rate       = np.mean([r.get("judge_full_parse", False) for r in results])

    bon_gain = float(np.mean(best_scores) - np.mean(bon1_scores))
    bon1_success_rate = float(np.mean([s >= 80 for s in bon1_scores]))

    # Two-judge agreement (only meaningful when two_judge_mode is True for results)
    two_judge_mode_active = bool(results and results[0].get("two_judge_mode", False))
    judges_agreement_rate = None
    if two_judge_mode_active:
        judges_agreement_rate = float(np.mean([r["judges_agree_on_winner"] for r in results]))

    # Source of bon1 scoring (eval_rich in two-judge mode, selection_fast otherwise)
    bon1_judge_source = results[0].get("bon1_judge_source", "selection_fast") if results else None

    # Per-temperature win statistics (unchanged)
    temp_stats: Dict[str, Any] = {}
    if results and results[0].get("all_candidate_temperatures"):
        all_temps = results[0]["all_candidate_temperatures"]
        win_counts = {f"{t:.2f}": 0 for t in all_temps}
        score_by_temp: Dict[str, List[float]] = {f"{t:.2f}": [] for t in all_temps}
        for r in results:
            best_t = r.get("best_candidate_temperature")
            if best_t is not None:
                key = f"{best_t:.2f}"
                if key in win_counts:
                    win_counts[key] += 1
            r_temps = r.get("all_candidate_temperatures") or []
            r_scores = r.get("all_candidate_scores") or []
            for t, s in zip(r_temps, r_scores):
                key = f"{t:.2f}"
                if key in score_by_temp:
                    score_by_temp[key].append(s)
        temp_stats = {
            "ladder": all_temps,
            "win_counts": win_counts,
            "win_pct": {k: round(100 * v / n, 1) for k, v in win_counts.items()},
            "mean_score_by_temperature": {
                k: round(float(np.mean(v)), 2) if v else 0.0
                for k, v in score_by_temp.items()
            },
        }

    error_types = sorted(set(r["error_type"] for r in results))
    per_error   = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        et_dict = {
            "n":                       len(sub),
            "judge_success_rate":      float(np.mean([r["judge_success"]             for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"]        for r in sub])),
            "mean_judge_score":        float(np.mean([r["best_judge_score"]          for r in sub])),
            "functional_tool_acc":     float(np.mean([r["functional_tool_accuracy"]  for r in sub])),
            "mean_param_accuracy":     float(np.mean([r["param_accuracy"]            for r in sub])),
            "exact_match_rate":        float(np.mean([r["exact_match"]               for r in sub])),
            "functional_match_rate":   float(np.mean([r["functional_match"]          for r in sub])),
            "param_only_match_rate":   float(np.mean([r["param_only_match"]          for r in sub])),
            "step_count_match_rate":   float(np.mean([r["step_count_match"]          for r in sub])),
        }
        if two_judge_mode_active:
            et_dict["judges_agree_on_winner_rate"] = float(
                np.mean([r["judges_agree_on_winner"] for r in sub])
            )
            et_dict["selection_judge_winner_success_rate"] = float(
                np.mean([r["selection_judge_winner_success"] for r in sub])
            )
        per_error[et] = et_dict

    success_dist = {}
    for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
        c = sum(r["best_success"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100 * c / n, 1)}

    gt_uses_nl = bool(results[0].get("gt_uses_nl_tool_names", False))

    structural_section = {
        "exact_match_rate":         round(float(exact_match_rate),      3),
        "functional_match_rate":    round(float(functional_match_rate), 3),
        "param_only_match_rate":    round(float(param_only_match_rate), 3),
        "step_count_match_rate":    round(float(step_match_rate),       3),
        "mean_functional_tool_acc": round(float(np.mean(func_tools)),   3),
        "mean_param_accuracy":      round(float(np.mean(param_accs)),   3),
        "mean_dependency_accuracy": round(float(np.mean(dep_accs)),     3),
        "tool_accuracy_dist": {
            "0.0":    sum(a == 0.0 for a in func_tools),
            "0-0.2":  sum(0 < a < 0.2 for a in func_tools),
            "0.2-0.5":sum(0.2 <= a < 0.5 for a in func_tools),
            "0.5-0.8":sum(0.5 <= a < 0.8 for a in func_tools),
            "0.8-1.0":sum(0.8 <= a < 1.0 for a in func_tools),
            "1.0":    sum(a == 1.0 for a in func_tools),
        },
    }
    if gt_uses_nl:
        structural_section["NOTE_tool_accuracy"] = (
            "GT uses natural-language tool names; exact_tool_accuracy will be ~0. "
            "Use functional_tool_accuracy instead."
        )

    accuracy_section = {
        "judge_success_rate":    round(float(judge_success_rate),    3),
        "error_handled_rate":    round(float(error_handled_rate),    3),
        "bon1_success_rate":     round(float(bon1_success_rate),     3),
        "bon_gain_pts":          round(bon_gain,                     2),
    }
    if two_judge_mode_active:
        accuracy_section["judges_agreement_rate"] = round(judges_agreement_rate, 3)
        # Selection-judge's own success rate on its picks
        sel_success_rate = float(np.mean([r["selection_judge_winner_success"] for r in results]))
        accuracy_section["selection_judge_winner_success_rate"] = round(sel_success_rate, 3)

    stats = {
        "label":               label,
        "n_examples":          n,
        "n_candidates":        results[0]["n_candidates"],
        "two_judge_mode":      two_judge_mode_active,
        "bon1_judge_source":   bon1_judge_source,
        "gt_uses_nl_tools":    gt_uses_nl,
        "judge_full_parse_rate": round(float(full_parse_rate), 3),
        "accuracy": accuracy_section,
        "judge_scores": {
            "bon_mean":    round(float(np.mean(best_scores)), 2),
            "bon_median":  round(float(np.median(best_scores)), 2),
            "bon_std":     round(float(np.std(best_scores)), 2),
            "bon1_mean":   round(float(np.mean(bon1_scores)), 2),
            "mean_of_n":   round(float(np.mean(mean_scores)), 2),
            "pct_gte_80":  round(100 * sum(s >= 80  for s in best_scores) / n, 1),
            "pct_eq_100": round(100 * sum(s == 100 for s in best_scores) / n, 1),
        },
        "temperature_diversity": temp_stats,
        "structural": structural_section,
        "success_prediction_dist": success_dist,
        "per_error_type": per_error,
    }

    # Print
    W = 70
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  N examples : {n}  |  N candidates : {results[0]['n_candidates']}")
    print(f"  Two-judge mode : {two_judge_mode_active}  "
          f"|  bon1 source : {bon1_judge_source}")
    if temp_stats:
        print(f"  Temperature ladder : {temp_stats['ladder']}")
    if stats["gt_uses_nl_tools"]:
        print(f"  WARN  GT uses NL tool names - exact_tool_accuracy unreliable")
    if full_parse_rate < 0.9:
        print(f"  WARN  Judge full-parse rate: {100*full_parse_rate:.0f}%")

    print(f"\n  -- Primary Accuracy ---------------------------------------")
    print(f"  BoN  judge success (score>=80) : {100*judge_success_rate:.1f}%")
    print(f"  Bo1  judge success (baseline)  : {100*bon1_success_rate:.1f}%")
    print(f"  BoN gain                       : +{bon_gain:.1f} pts")
    print(f"  Error type handled             : {100*error_handled_rate:.1f}%")
    if two_judge_mode_active:
        print(f"  Two-judge agreement on winner  : {100*judges_agreement_rate:.1f}%")
        print(f"  Selection-judge winner success : "
              f"{100*accuracy_section['selection_judge_winner_success_rate']:.1f}%")

    print(f"\n  -- Judge Scores -------------------------------------------")
    print(f"  BoN  mean / median / std : {stats['judge_scores']['bon_mean']:.1f} / "
          f"{stats['judge_scores']['bon_median']:.1f} / {stats['judge_scores']['bon_std']:.1f}")
    print(f"  Bo1  mean                : {stats['judge_scores']['bon1_mean']:.1f}")
    print(f"  Mean-of-N                : {stats['judge_scores']['mean_of_n']:.1f}")
    print(f"  >=80 (good)              : {stats['judge_scores']['pct_gte_80']:.1f}%")
    print(f"  =100 (perfect)           : {stats['judge_scores']['pct_eq_100']:.1f}%")

    if temp_stats:
        print(f"\n  -- Temperature Diversity (which temperature wins?) --------")
        for t in temp_stats["ladder"]:
            key = f"{t:.2f}"
            wp = temp_stats["win_pct"].get(key, 0.0)
            ms = temp_stats["mean_score_by_temperature"].get(key, 0.0)
            bar = "#" * int(wp / 2)
            print(f"  T={key}  win {wp:5.1f}%  mean_score {ms:5.1f}  {bar}")

    print(f"\n  -- Structural Metrics (vs GT) ------------------------------")
    print(f"  Exact match                : {100*exact_match_rate:.1f}%")
    print(f"  Functional match           : {100*functional_match_rate:.1f}%")
    print(f"  Param-only match (>=50%)   : {100*param_only_match_rate:.1f}%")
    print(f"  Step count match           : {100*step_match_rate:.1f}%")
    print(f"  Mean functional tool acc   : {np.mean(func_tools):.3f}")
    print(f"  Mean param accuracy        : {np.mean(param_accs):.3f}")
    print(f"  Mean dependency accuracy   : {np.mean(dep_accs):.3f}")

    print(f"\n  -- Success Prediction --------------------------------------")
    for pred, d in success_dist.items():
        print(f"    {pred:12s}: {d['count']:3d}  ({d['pct']:.1f}%)")

    if len(error_types) > 1:
        print(f"\n  -- Per Error-Type ------------------------------------------")
        hdr_cols = ["Error Type", "N", "Success%", "Handled%", "Judge",
                    "FuncTool%", "Param%", "FuncMatch%"]
        if two_judge_mode_active:
            hdr_cols.append("Agree%")
        widths = [28, 4, 8, 8, 6, 9, 6, 10] + ([6] if two_judge_mode_active else [])
        hdr = "  " + "  ".join(f"{c:>{w}s}" if i > 0 else f"{c:<{w}s}"
                                for i, (c, w) in enumerate(zip(hdr_cols, widths)))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for et, d in per_error.items():
            row = (
                f"  {et:28s}  {d['n']:>4}  "
                f"{100*d['judge_success_rate']:>8.1f}  "
                f"{100*d['error_type_handled_rate']:>8.1f}  "
                f"{d['mean_judge_score']:>6.1f}  "
                f"{100*d['functional_tool_acc']:>9.1f}  "
                f"{100*d['mean_param_accuracy']:>6.1f}  "
                f"{100*d['functional_match_rate']:>10.1f}"
            )
            if two_judge_mode_active:
                row += f"  {100*d['judges_agree_on_winner_rate']:>6.1f}"
            print(row)
    print()
    return stats


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _parse_temperatures(arg: Optional[str], n: int,
                        single_temperature: Optional[float]) -> List[float]:
    if arg is not None and single_temperature is not None:
        raise ValueError(
            "Pass exactly one of --temperatures or --temperature, not both."
        )
    if arg is not None:
        try:
            temps = [float(x.strip()) for x in arg.split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(f"Could not parse --temperatures '{arg}': {e}")
        if len(temps) != n:
            raise ValueError(
                f"--temperatures has {len(temps)} values but --n={n}; "
                f"they must match."
            )
        return temps
    if single_temperature is not None:
        return [float(single_temperature)] * n
    if n == 5:
        return list(DEFAULT_TEMPERATURE_LADDER)
    if n == 1:
        return [0.6]
    return list(np.linspace(0.2, 1.0, n).round(3).tolist())


def _check_judge_health(judge_url: str, label: str) -> bool:
    health_url = judge_url.replace("/v1/chat/completions", "/health")
    try:
        r = requests.get(health_url, timeout=5)
        print(f"OK  {label:20s} healthy: {r.json()}  ({judge_url})")
        return True
    except Exception as e:
        print(f"FAIL {label:20s} not reachable at {judge_url}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Best-of-N evaluation for planner — two-judge variant."
    )
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--test-parquet",  required=True)
    parser.add_argument("--n", type=int, default=5,
        help="Number of candidates per query (default: 5)")
    parser.add_argument("--temperatures", type=str, default=None,
        help="Comma-separated list of temperatures, length must equal --n.")
    parser.add_argument("--temperature", type=float, default=None,
        help="Single temperature applied to all N candidates.")
    parser.add_argument("--max-new-tokens",        type=int,   default=512)
    parser.add_argument("--judge-max-tokens",      type=int,   default=32,
        help="Max tokens for FAST scoring of all N candidates by SELECTION judge. Use 32.")
    parser.add_argument("--judge-eval-max-tokens", type=int,   default=300,
        help="Max tokens for RICH scoring by EVAL judge. Use >=200.")
    parser.add_argument("--output", default="best_of_n_results.json")
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--return-all", action="store_true")
    parser.add_argument("--device", default="cuda:0")

    # Two-judge support: selection vs eval
    parser.add_argument("--judge_url",
        default="http://localhost:8001/v1/chat/completions",
        help="SELECTION judge URL — used for BoN argmax (fast scoring of all N candidates).")
    parser.add_argument("--eval-judge-url",
        default=None,
        help="EVAL judge URL — used for reported metrics (rich scoring of winner, "
             "and of bon1 when different from --judge_url). Defaults to --judge_url, "
             "in which case behavior is identical to the original single-judge script.")

    parser.add_argument("--perfect-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if not args.perfect_only and not args.full:
        parser.error("Specify at least one of --perfect-only or --full")

    try:
        temperature_ladder = _parse_temperatures(args.temperatures, args.n, args.temperature)
    except ValueError as e:
        parser.error(str(e))
    args.resolved_temperatures = temperature_ladder

    # Resolve eval-judge-url
    eval_judge_url = args.eval_judge_url if args.eval_judge_url else args.judge_url
    two_judge_mode = (eval_judge_url != args.judge_url)
    args.resolved_eval_judge_url = eval_judge_url
    args.resolved_two_judge_mode = two_judge_mode

    print(f"Inference temperature ladder (N={args.n}): {temperature_ladder}")
    if len(set(temperature_ladder)) == 1:
        print(f"  -> Single-temperature mode (T={temperature_ladder[0]}).")
    else:
        print(f"  -> Temperature-diversity mode.")
    print(f"ARTIFACT_ERROR_TYPES (structural GT remap): {sorted(ARTIFACT_ERROR_TYPES)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats_output = args.stats_output or args.output.replace(".json", ".stats.json")

    # Health-check both judges
    ok_sel = _check_judge_health(args.judge_url, "Selection judge")
    if two_judge_mode:
        ok_eval = _check_judge_health(eval_judge_url, "Eval judge")
    else:
        ok_eval = ok_sel
        print(f"     (single-judge mode — eval judge same as selection judge)")
    if not (ok_sel and ok_eval):
        print("FAIL: one or more judges unreachable. Aborting.")
        return

    if args.judge_eval_max_tokens < 200:
        print(f"WARN  --judge-eval-max-tokens={args.judge_eval_max_tokens} is low. "
              f"Recommend >=200.")

    planner    = PlannerModel(args.planner_model, device=args.device)
    all_output = {"config": vars(args), "runs": {}}
    all_stats  = {"config": vars(args), "runs": {}}

    if args.perfect_only:
        print("\nLoading perfect-only test examples...")
        examples, perfect_gt_by_qid = load_test_parquet(args.test_parquet, perfect_only=True)
        print(f"  {len(examples)} examples")
        results = evaluate_best_of_n(
            planner, examples,
            perfect_gt_by_qid=perfect_gt_by_qid,
            n=args.n, temperatures=temperature_ladder,
            selection_judge_url=args.judge_url,
            eval_judge_url=eval_judge_url,
            max_new_tokens=args.max_new_tokens,
            judge_max_tokens=args.judge_max_tokens,
            judge_eval_max_tokens=args.judge_eval_max_tokens,
            return_all_candidates=args.return_all,
        )
        stats = compute_stats(results, f"PERFECT-ONLY TEST SET  (Best-of-{args.n}, "
                                       f"temps={temperature_ladder}, "
                                       f"two_judge={two_judge_mode})")
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"]  = stats

    if args.full:
        print("\nLoading full test set...")
        examples, perfect_gt_by_qid = load_test_parquet(args.test_parquet, perfect_only=False)
        print(f"  {len(examples)} examples")
        results = evaluate_best_of_n(
            planner, examples,
            perfect_gt_by_qid=perfect_gt_by_qid,
            n=args.n, temperatures=temperature_ladder,
            selection_judge_url=args.judge_url,
            eval_judge_url=eval_judge_url,
            max_new_tokens=args.max_new_tokens,
            judge_max_tokens=args.judge_max_tokens,
            judge_eval_max_tokens=args.judge_eval_max_tokens,
            return_all_candidates=args.return_all,
        )
        stats = compute_stats(results, f"FULL TEST SET  (Best-of-{args.n}, "
                                       f"temps={temperature_ladder}, "
                                       f"two_judge={two_judge_mode})")
        all_output["runs"]["full"] = results
        all_stats["runs"]["full"]  = stats

    with open(args.output, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"Results saved to: {args.output}")

    with open(stats_output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats   saved to: {stats_output}")


if __name__ == "__main__":
    main()