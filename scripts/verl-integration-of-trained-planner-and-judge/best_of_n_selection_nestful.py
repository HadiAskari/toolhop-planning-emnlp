#!/usr/bin/env python3
"""
Best-of-N Selection and Evaluation for Planner Model

Evaluates the trained planner model using temperature-diversity Best-of-N
inference (the FORTE inference scheme):
  - Generate N candidate plans, one per temperature in a configurable ladder
  - Score all candidates via the judge server (HTTP, same as training)
  - Select the highest-scoring plan

Default temperature ladder: [0.2, 0.4, 0.6, 0.8, 1.0] for N=5.  Sampling one
candidate at each temperature gives the judge a structurally diverse pool to
choose between, rather than N near-duplicates at a single temperature.  See
the paper's Method section, "Inference: Temperature-Diversity Best-of-N".

For reproducibility / ablation, you can also run single-temperature BoN by
passing --temperature X (which expands to [X, X, ..., X]).  Pass exactly one
of --temperature or --temperatures.

Performance optimizations over naive implementation:
  1. Persistent requests.Session with connection keep-alive (~50-100ms saved per call)
  2. Persistent ThreadPoolExecutor across all examples (no pool creation overhead)
  3. Pipelined generation/scoring: generate example K+1 on GPU while scoring
     example K over HTTP (different resources, true overlap)
  4. Rich call fired asynchronously, result collected at next iteration start

Note on generation cost with the temperature ladder:
  HuggingFace `generate()` does not support per-row temperatures in a batched
  call, so temperature-diverse sampling runs N sequential forward passes
  rather than one batched pass.  This makes generation slower, but generation
  is rarely the bottleneck (judge calls dominate), and the quality gain from
  candidate diversity is worth it.  If you specifically want the
  single-temperature batched path for ablation, pass --temperature 0.7.

Two evaluation modes:
  --perfect-only   Evaluate on perfect plans subset (error_type == "none", quality_score == 100)
  --full           Evaluate on full test set including all 10 error types

STRUCTURAL EVAL CONVENTION:
  For every error_type other than "none", the GT in the dataset is a
  deliberately-flawed plan and the *correct* model behavior is to diverge
  from it (produce the perfect plan instead).  ARTIFACT_ERROR_TYPES therefore
  contains all 9 non-`none` error types: when scoring structurally, we remap
  the GT to the perfect plan for that query (looked up by query_id) so the
  model is rewarded for fixing the seeded error rather than reproducing it.

Usage:
    # Start judge server first:
    CUDA_VISIBLE_DEVICES=3 python judge_server.py --model /path/to/judge/merged --port 8001

    # Default temperature-diversity BoN-5 (FORTE main inference scheme):
    CUDA_VISIBLE_DEVICES=2 python best_of_n_selection_nestful.py \
        --planner-model ${FORTE_ROOT}/planner_rl/checkpoints_grpo_filtered_qwen7b_nestful/global_step_72/actor \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_nestful_full/test.parquet \
        --n 5 --full \
        --temperatures 0.2,0.4,0.6,0.8,1.0 \
        --judge-eval-max-tokens 300 \
        --output results/Qwen-7b-filteredGRPO-nestful/results_full.json \
        --judge_url http://localhost:8001/v1/chat/completions

    # Single-temperature BoN-5 (for ablation):
    CUDA_VISIBLE_DEVICES=4 python best_of_n_selection.py \
        --planner-model /path/to/planner \
        --test-parquet /path/to/test.parquet \
        --n 5 --full \
        --temperature 0.7 \
        --output results/run-singletemp/results_full.json \
        --judge_url http://localhost:8002/v1/chat/completions
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

from dataset_utils import resolve_dataset, dataset_label


# ── Prompt format — must exactly match SFT/RL training format ────────────────

SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)

# Error types where the GT is deliberately flawed and the correct model
# response is to produce the perfect plan instead.  When scoring structurally,
# we remap the GT to the perfect plan for the same query_id so we measure
# whether the model FIXED the seeded error, not whether it reproduced it.
#
# All 9 non-`none` error types belong here.  Previously this set only
# contained {inefficient_order, unnecessary_steps, incomplete_plan} which
# silently penalized the model for correctly fixing the other 6 error types
# (wrong_tool, parameter_typo, type_mismatch, circular_dependency,
# forward_reference, missing_dependency).
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

# Default temperature ladder for the FORTE temperature-diversity BoN scheme.
# Used when neither --temperature nor --temperatures is provided.
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


# ── Judge server client ───────────────────────────────────────────────────────

JUDGE_SERVER_URL = "http://localhost:8001/v1/chat/completions"
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

# Persistent HTTP session - reuse TCP connections to judge server.
_http_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=16,
            pool_maxsize=16,
            max_retries=0,
        )
        _http_session.mount("http://", adapter)
        _http_session.headers.update({"Connection": "keep-alive"})
    return _http_session


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


def score_plan_via_judge(query: str, plan_str: str, tools: Dict,
                          max_tokens: int = 300, retries: int = 3) -> Dict[str, Any]:
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

    session = _get_session()
    content = ""
    for attempt in range(retries):
        try:
            resp = session.post(JUDGE_SERVER_URL, json=payload, timeout=120)
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
            return annotation

        except (requests.ConnectionError, requests.Timeout) as e:
            global _http_session
            _http_session = None
            if attempt < retries - 1:
                time.sleep(2.0)
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
                    }
            except Exception:
                pass
            break

    return {
        "quality_score": 0, "success_prediction": "no",
        "reasoning": "judge call failed", "issues": [], "confidence": 0.0,
        "_full_parse": False,
    }


# ── Plan parsing ──────────────────────────────────────────────────────────────

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
                "step_id":        step_id,
                "output_variable": output_var,
                "tool_name":      tool_name,
                "parameters":     params,
            })
        except Exception:
            continue
    return steps


def normalize_value(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())


# ── Structural evaluation (vs ground truth) ───────────────────────────────────

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


# ── Planner model ─────────────────────────────────────────────────────────────

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
        """Tokenize the chat-templated prompt once; return on the right device."""
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
        """Generate exactly one plan from pre-tokenized inputs at a given temperature."""
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
        """Generate a single plan at a single temperature (kept for backward compat)."""
        inputs = self._build_prompt(question, tools)
        return self._generate_one(inputs, temperature, max_new_tokens)

    def generate_n_plans(
        self,
        question: str,
        tools: Dict[str, Any],
        temperatures: List[float],
        max_new_tokens: int = 512,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate one plan per temperature in `temperatures`.

        Two paths:

        1. All temperatures equal: use a single batched forward pass
           (input expanded to N rows) for maximum GPU efficiency.

        2. Mixed temperatures (the FORTE default ladder):
           HuggingFace generate() applies one temperature per call, so we
           run N sequential forward passes.

        Returns:
            (plans, temperatures_used)  -- both length N, same order.
        """
        n = len(temperatures)
        if n == 0:
            raise ValueError("temperatures must contain at least one value")

        inputs = self._build_prompt(question, tools)
        all_equal = all(abs(t - temperatures[0]) < 1e-9 for t in temperatures)

        # ── Fast path: batched generation when all temperatures are equal ──
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
                # Fall through to sequential path

        # ── Diversity path: one forward pass per temperature ───────────────
        plans: List[str] = []
        for t in temperatures:
            try:
                plan = self._generate_one(inputs, t, max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                plan = ""
            plans.append(plan)
        return plans, list(temperatures)


# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── Best-of-N evaluation (pipelined) ─────────────────────────────────────────

def _score_candidates_parallel(pool: ThreadPoolExecutor, question: str,
                                candidates: List[str], tools: Dict,
                                max_tokens: int) -> List[Dict]:
    futures = {
        pool.submit(score_plan_via_judge, question, plan, tools, max_tokens): idx
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
                  best_idx: int, best_plan: str, best_score: Dict,
                  gt_steps: List[Dict], structural_gt_steps: List[Dict],
                  n: int, return_all_candidates: bool, dataset: str) -> Dict:
    question = ex["question"]
    tools    = ex["tools"]

    best_steps  = parse_plan_steps(best_plan)
    struct_eval = evaluate_plan_vs_gt(best_steps, structural_gt_steps, tools=tools)

    judge_success = best_score["quality_score"] >= 80

    ref_error_type = ex["error_type"]
    ref_is_perfect = (ref_error_type == "none" and ex["quality_score"] >= 100)

    if ref_is_perfect:
        error_type_handled = judge_success
    else:
        error_type_handled = best_score["quality_score"] >= ex["quality_score"]

    judge_agrees_with_ref = (ref_is_perfect == judge_success)

    result = {
        "dataset": dataset,
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
        "judge_success":         judge_success,
        "best_judge_score":      best_score["quality_score"],
        "best_success":          best_score["success_prediction"],
        "best_confidence":       best_score["confidence"],
        "judge_full_parse":      best_score.get("_full_parse", False),
        "bon1_judge_score":      fast_scores[0]["quality_score"],
        "bon1_temperature":      cand_temperatures[0] if cand_temperatures else None,
        "all_candidate_scores":  [s["quality_score"] for s in fast_scores],
        "all_candidate_temperatures": list(cand_temperatures),
        "mean_candidate_score":  float(np.mean([s["quality_score"] for s in fast_scores])),
        "candidate_score_std":   float(np.std([s["quality_score"] for s in fast_scores])),
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
    perfect_gt_by_qid: Dict[int, str] = None, dataset: str = "toolhop",
    n: int = 5,
    temperatures: Optional[List[float]] = None,
    max_new_tokens: int = 512,
    judge_max_tokens: int = 32,
    judge_eval_max_tokens: int = 300,
    return_all_candidates: bool = False,
) -> List[Dict]:
    """
    Pipelined Best-of-N evaluation with a temperature ladder.

    `temperatures`: list of length N. Each candidate is sampled at one
    temperature from this list (in order). Default ladder when None is
    [0.2, 0.4, 0.6, 0.8, 1.0] (FORTE main inference scheme).
    """
    if temperatures is None:
        temperatures = list(DEFAULT_TEMPERATURE_LADDER)
    if len(temperatures) != n:
        raise ValueError(
            f"len(temperatures)={len(temperatures)} but n={n}; they must match"
        )

    results = []
    partial_parse_count = 0
    pool = ThreadPoolExecutor(max_workers=n + 4)

    prev_candidates:  Optional[List[str]]   = None
    prev_cand_temps:  Optional[List[float]] = None
    prev_fast_scores: Optional[List[Dict]]  = None
    prev_best_idx:    Optional[int]         = None
    prev_best_plan:   Optional[str]         = None
    prev_rich_future: Optional[Future]      = None
    prev_ex:          Optional[Dict]        = None
    prev_gt_steps:    Optional[List[Dict]]  = None
    prev_struct_gt_steps: Optional[List[Dict]] = None

    def _get_structural_gt(ex, gt_steps):
        if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                and perfect_gt_by_qid
                and ex["query_id"] in perfect_gt_by_qid):
            return parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        return gt_steps

    def _finalize_previous():
        nonlocal partial_parse_count
        if prev_rich_future is None:
            return
        best_score = prev_rich_future.result()
        if not best_score.get("_full_parse", False):
            partial_parse_count += 1
        result = _build_result(
            prev_ex, prev_candidates, prev_cand_temps, prev_fast_scores,
            prev_best_idx, prev_best_plan, best_score,
            prev_gt_steps, prev_struct_gt_steps,
            n, return_all_candidates, dataset=dataset,
        )
        results.append(result)

    pbar = tqdm(examples, desc=f"Best-of-{n} evaluation (temps={temperatures})")

    for ex_idx, ex in enumerate(pbar):
        question     = ex["question"]
        tools        = ex["tools"]
        ground_truth = ex["ground_truth"]
        gt_steps     = parse_plan_steps(ground_truth)
        struct_gt    = _get_structural_gt(ex, gt_steps)

        # Phase 1: Generate candidates (GPU, one per temperature)
        candidates, cand_temps = planner.generate_n_plans(
            question, tools,
            temperatures=temperatures,
            max_new_tokens=max_new_tokens,
        )

        # Phase 2: Finalize previous example
        _finalize_previous()

        # Phase 3: Score current candidates (HTTP, parallel)
        fast_scores = _score_candidates_parallel(
            pool, question, candidates, tools, judge_max_tokens
        )

        # Phase 4: Pick best, fire rich call async
        best_idx, best_plan = _pick_best(candidates, fast_scores)
        rich_future = pool.submit(
            score_plan_via_judge, question, best_plan, tools, judge_eval_max_tokens
        )

        prev_candidates       = candidates
        prev_cand_temps       = cand_temps
        prev_fast_scores      = fast_scores
        prev_best_idx         = best_idx
        prev_best_plan        = best_plan
        prev_rich_future      = rich_future
        prev_ex               = ex
        prev_gt_steps         = gt_steps
        prev_struct_gt_steps  = struct_gt

        if results:
            recent = results[-10:]
            avg_score = np.mean([r["best_judge_score"] for r in recent])
            success_rate = np.mean([r["judge_success"] for r in recent])
            pbar.set_postfix({
                "avg_score": f"{avg_score:.0f}",
                "success%": f"{100*success_rate:.0f}",
            })

    _finalize_previous()
    pool.shutdown(wait=False)

    if partial_parse_count > 0:
        pct = 100 * partial_parse_count / len(results) if results else 0
        print(f"\n  WARN  Partial parse rate on rich calls: {partial_parse_count}/{len(results)} ({pct:.1f}%)")
        if pct > 20:
            print("        Consider increasing --judge-eval-max-tokens (current may be too low)")

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

    # Per-temperature win statistics
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
        per_error[et] = {
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

    stats = {
        "label":               label,
        "dataset": results[0].get("dataset", "unknown"),
        "n_examples":          n,
        "n_candidates":        results[0]["n_candidates"],
        "gt_uses_nl_tools":    gt_uses_nl,
        "judge_full_parse_rate": round(float(full_parse_rate), 3),
        "accuracy": {
            "judge_success_rate":    round(float(judge_success_rate),    3),
            "error_handled_rate":    round(float(error_handled_rate),    3),
            "bon1_success_rate":     round(float(bon1_success_rate),     3),
            "bon_gain_pts":          round(bon_gain,                     2),
        },
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
        hdr = (f"  {'Error Type':28s}  {'N':>4}  {'Success%':>8}  "
               f"{'Handled%':>8}  {'Judge':>6}  {'FuncTool%':>9}  "
               f"{'Param%':>6}  {'FuncMatch%':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for et, d in per_error.items():
            print(
                f"  {et:28s}  {d['n']:>4}  "
                f"{100*d['judge_success_rate']:>8.1f}  "
                f"{100*d['error_type_handled_rate']:>8.1f}  "
                f"{d['mean_judge_score']:>6.1f}  "
                f"{100*d['functional_tool_acc']:>9.1f}  "
                f"{100*d['mean_param_accuracy']:>6.1f}  "
                f"{100*d['functional_match_rate']:>10.1f}"
            )
    print()
    return stats


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _parse_temperatures(arg: Optional[str], n: int,
                        single_temperature: Optional[float]) -> List[float]:
    """
    Resolve the temperature ladder from CLI arguments.

    Priority order:
      1. --temperatures (comma-separated list); must have length n
      2. --temperature single value, expanded to [t]*n
      3. Default ladder (DEFAULT_TEMPERATURE_LADDER) if n == 5,
         else evenly spaced ladder from 0.2 to 1.0.

    Errors if --temperatures and --temperature are both passed.
    """
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Best-of-N evaluation for planner model with temperature-diversity sampling"
    )
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--test-parquet",  required=True)
    parser.add_argument("--n", type=int, default=5,
        help="Number of candidates per query (default: 5)")
    parser.add_argument("--temperatures", type=str, default=None,
        help="Comma-separated list of temperatures, length must equal --n. "
             "Example: '0.2,0.4,0.6,0.8,1.0' (FORTE main inference scheme). "
             "If omitted and --temperature is also omitted, the default ladder "
             "[0.2,0.4,0.6,0.8,1.0] is used for n=5.")
    parser.add_argument("--temperature", type=float, default=None,
        help="Single temperature applied to all N candidates. Use this for "
             "the single-temperature BoN ablation. Mutually exclusive with --temperatures.")
    parser.add_argument("--max-new-tokens",        type=int,   default=512)
    parser.add_argument("--judge-max-tokens",      type=int,   default=32,
        help="Max tokens for fast scoring of all N candidates. Use 32.")
    parser.add_argument("--judge-eval-max-tokens", type=int,   default=300,
        help="Max tokens for rich scoring of the winner. Use >=200.")
    parser.add_argument("--output", default="best_of_n_results.json")
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--return-all", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--judge_url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--perfect-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", default="auto",
                        choices=["auto", "toolhop", "nestful"],
                        help="Dataset for labels and metadata. "
                             "'auto' detects from the parquet's "
                             "data_source field.")
    args = parser.parse_args()

    # Resolve dataset (auto-detect from parquet data_source, or explicit)
    dataset = resolve_dataset(args.dataset, args.test_parquet)
    print(f"\nDataset: {dataset_label(dataset)} ({dataset})")

    if not args.perfect_only and not args.full:
        parser.error("Specify at least one of --perfect-only or --full")

    try:
        temperature_ladder = _parse_temperatures(args.temperatures, args.n, args.temperature)
    except ValueError as e:
        parser.error(str(e))
    args.resolved_temperatures = temperature_ladder

    print(f"Inference temperature ladder (N={args.n}): {temperature_ladder}")
    if len(set(temperature_ladder)) == 1:
        print(f"  -> Single-temperature mode (T={temperature_ladder[0]}). "
              f"Will use batched generation.")
    else:
        print(f"  -> Temperature-diversity mode. Will use {args.n} sequential "
              f"forward passes per query (slower than single-temp, but produces "
              f"a structurally diverse candidate pool).")

    print(f"ARTIFACT_ERROR_TYPES (structural GT remap): "
          f"{sorted(ARTIFACT_ERROR_TYPES)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats_output = args.stats_output or args.output.replace(".json", ".stats.json")

    global JUDGE_SERVER_URL
    JUDGE_SERVER_URL = args.judge_url

    try:
        r = requests.get(JUDGE_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5)
        print(f"OK Judge server healthy: {r.json()}")
    except Exception as e:
        print(f"FAIL Judge server not reachable: {e}")
        return

    if args.judge_eval_max_tokens < 200:
        print(f"WARN  --judge-eval-max-tokens={args.judge_eval_max_tokens} is low. "
              f"Recommend >=200.")

    planner    = PlannerModel(args.planner_model, device=args.device)
    all_output = {"config": {**vars(args), "resolved_dataset": dataset}, "runs": {}}
    all_stats  = {"config": {**vars(args), "resolved_dataset": dataset}, "runs": {}}

    if args.perfect_only:
        print("\nLoading perfect-only test examples...")
        examples, perfect_gt_by_qid = load_test_parquet(args.test_parquet, perfect_only=True)
        print(f"  {len(examples)} examples")
        results = evaluate_best_of_n(
            planner, examples,
            perfect_gt_by_qid=perfect_gt_by_qid, dataset=dataset,
            n=args.n, temperatures=temperature_ladder,
            max_new_tokens=args.max_new_tokens,
            judge_max_tokens=args.judge_max_tokens,
            judge_eval_max_tokens=args.judge_eval_max_tokens,
            return_all_candidates=args.return_all,
        )
        stats = compute_stats(results, f"PERFECT-ONLY TEST SET  (Best-of-{args.n}, "
                                       f"temps={temperature_ladder}) — {dataset_label(dataset)}")
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"]  = stats

    if args.full:
        print("\nLoading full test set...")
        examples, perfect_gt_by_qid = load_test_parquet(args.test_parquet, perfect_only=False)
        print(f"  {len(examples)} examples")
        results = evaluate_best_of_n(
            planner, examples,
            perfect_gt_by_qid=perfect_gt_by_qid, dataset=dataset,
            n=args.n, temperatures=temperature_ladder,
            max_new_tokens=args.max_new_tokens,
            judge_max_tokens=args.judge_max_tokens,
            judge_eval_max_tokens=args.judge_eval_max_tokens,
            return_all_candidates=args.return_all,
        )
        stats = compute_stats(results, f"FULL TEST SET  (Best-of-{args.n}, "
                                       f"temps={temperature_ladder}) — {dataset_label(dataset)}")
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