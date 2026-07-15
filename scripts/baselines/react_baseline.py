#!/usr/bin/env python3
"""
ReAct Baseline for ToolHop Tool Planning

Implements the ReAct (Reason + Act) prompting paradigm (Yao et al., ICLR 2023)
as a baseline comparison for the PPO-trained planner.

Key differences from the trained planner:
  - NO training — prompting only (few-shot or zero-shot)
  - Interleaved Thought → Action steps instead of a single upfront plan
  - Can use any HF model (the SFT base, a larger open model, etc.)

Since ToolHop has no real execution environment, we run "offline" ReAct:
the model generates the complete Thought/Action chain in a single forward pass
without real observations between steps. This is the standard prompting baseline
used in tool-planning literature when the environment is not executable.

Two variants:
  --mode zero-shot   No examples in prompt (hardest, tests raw reasoning)
  --mode few-shot    3 hand-crafted ToolHop-style ReAct examples in prompt

Output format matches best_of_n_selection.py exactly so stats can be
compared side-by-side.

Usage:
    # Zero-shot ReAct with the SFT base model:
    python react_baseline.py \\
        --model /path/to/base/or/sft/model \\
        --test-parquet data/verl_rl_full_clean/test.parquet \\
        --mode zero-shot --full \\
        --output react_zero_shot_results.json \\
        --stats-output react_zero_shot_stats.json

    # Few-shot ReAct:
    CUDA_VISIBLE_DEVICES=2 python react_baseline.py \
        --model ${FORTE_ROOT}/planner_finetuning/checkpoints_planner_sft-llama-3b/global_step_147 \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --mode few-shot --full \
        --output react_few_shot_Llama-3B-Instruct_results.json \
        --judge_url http://localhost:8001/v1/chat/completions
        --stats-output react_few_shot_Llama-3B_stats.json

    # Score with the judge server (start it first on GPU 7):
    CUDA_VISIBLE_DEVICES=7 python judge_server.py \\
        --model /path/to/judge/merged --port 8001
"""

import json
import re
import time
import argparse
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Error types where the GT is deliberately flawed in ordering/length.
# Use perfect plan for same query_id as structural comparison target.
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


# ── Shared constants (must match SFT/RL format) ───────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)

REACT_SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans using "
    "step-by-step reasoning. Given a query and available tools, reason about "
    "each step before generating the tool call. "
    "Alternate between Thought (your reasoning) and Action (the tool call)."
)

JUDGE_SERVER_URL = "http://localhost:8004/v1/chat/completions"


# ── Few-shot ReAct examples (hand-crafted for ToolHop style) ─────────────────

FEW_SHOT_EXAMPLES = """
Example 1:

Query: What is the capital of the country where the inventor of the telephone was born?

Available Tools:
- biographical_lookup(person_name: string (required), field: string (required))
- geography_lookup(location: string (required), info_type: string (required))

Thought: I need to find the inventor of the telephone first, then find their country of birth, then look up the capital.
Step 0: {{0}} = biographical_lookup(person_name='Alexander Graham Bell', field='country_of_birth')
Thought: Now I have the country. Next I need to find its capital.
Step 1: {{1}} = geography_lookup(location={{0}}, info_type='capital_city')

---

Example 2:

Query: How many days between the birth of the US president who served during World War II and the day the war ended in Europe?

Available Tools:
- historical_event_lookup(event: string (required), field: string (required))
- person_lookup(name: string (required), field: string (required))
- date_difference(start_date: string (required), end_date: string (required), unit: string)

Thought: I need to find which US president served during World War II, then get their birth date, then find the VE Day date, then compute the difference.
Step 0: {{0}} = historical_event_lookup(event='World War II US presidency', field='president_name')
Thought: Now I have the president. I need their birth date.
Step 1: {{1}} = person_lookup(name={{0}}, field='birth_date')
Thought: Now I need the date World War II ended in Europe (VE Day).
Step 2: {{2}} = historical_event_lookup(event='VE Day', field='date')
Thought: Now I can compute the difference in days.
Step 3: {{3}} = date_difference(start_date={{1}}, end_date={{2}}, unit='days')

---

Example 3:

Query: What is the population of the city where the tallest building in the country that won the most FIFA World Cups is located?

Available Tools:
- sports_lookup(sport: string (required), query: string (required), field: string (required))
- architecture_lookup(query: string (required), country: string, field: string (required))
- geography_lookup(location: string (required), info_type: string (required))

Thought: I need to find which country has won the most FIFA World Cups.
Step 0: {{0}} = sports_lookup(sport='FIFA World Cup', query='most wins', field='country')
Thought: Now I need to find the tallest building in that country.
Step 1: {{1}} = architecture_lookup(query='tallest building', country={{0}}, field='city')
Thought: Now I need the population of that city.
Step 2: {{2}} = geography_lookup(location={{1}}, info_type='population')

---
""".strip()


# ── Prompt builders ───────────────────────────────────────────────────────────


def load_perfect_gt_from_parquet(parquet_path: str) -> dict:
    """
    Build {query_id: ground_truth_str} for error_type='none', quality>=100.
    Used to give artifact error types a fair structural comparison target.
    """
    import json as _json
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    perfect_gt: dict = {}
    for ei, rm in zip(extra_infos, reward_models):
        if isinstance(ei, str):
            ei = _json.loads(ei)
        if isinstance(rm, str):
            rm = _json.loads(rm)
        if not isinstance(ei, dict) or not isinstance(rm, dict):
            continue
        if (str(ei.get("error_type", "")) == "none"
                and int(ei.get("quality_score", 0)) >= 100):
            qid = ei.get("query_id", -1)
            gt_str = rm.get("ground_truth", "")
            if gt_str and qid not in perfect_gt:
                perfect_gt[qid] = gt_str
    print(f"  Loaded perfect GT for {len(perfect_gt)} query_ids.")
    return perfect_gt


def format_tools(tools: Dict[str, Any]) -> str:
    lines = ["Available Tools:"]
    seen = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in seen:
            seen[name] = tool_info
    for name, tool_info in seen.items():
        props = tool_info.get("parameters", {}).get("properties", {})
        required = tool_info.get("parameters", {}).get("required", [])
        params = ", ".join(
            f"{k}: {v.get('type','any')}{' (required)' if k in required else ''}"
            for k, v in props.items()
        )
        lines.append(f"- {name}({params})")
    return "\n".join(lines)


def build_zero_shot_prompt(question: str, tools: Dict[str, Any]) -> str:
    tools_str = format_tools(tools)
    return (
        f"Query: {question}\n\n"
        f"{tools_str}\n\n"
        f"Generate a step-by-step plan. For each step, first write a Thought explaining "
        f"your reasoning, then write the Action as a tool call.\n\n"
        f"Use this format:\n"
        f"Thought: <your reasoning about what to do next>\n"
        f"Step N: {{{{N}}}} = tool_name(param1=value1, ...)\n\n"
        f"Use {{{{N}}}} syntax to reference outputs from previous steps.\n\n"
        f"Begin:\n"
    )


def build_few_shot_prompt(question: str, tools: Dict[str, Any]) -> str:
    tools_str = format_tools(tools)
    return (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"---\n\n"
        f"Now solve:\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n\n"
        f"Thought: "
    )


def build_chat_messages(question: str, tools: Dict[str, Any], mode: str) -> List[Dict]:
    if mode == "zero-shot":
        user_content = build_zero_shot_prompt(question, tools)
    else:
        user_content = build_few_shot_prompt(question, tools)
    return [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── ReAct output parsing ──────────────────────────────────────────────────────

def extract_plan_from_react_output(react_text: str) -> str:
    """
    Extract just the Step N: ... lines from a ReAct-style output that
    interleaves Thought and Action lines.
    """
    step_lines = []
    for line in react_text.split("\n"):
        line = line.strip()
        if line.lower().startswith("action:"):
            line = line[len("action:"):].strip()
        if re.match(r"Step \d+:", line):
            step_lines.append(line)
    return "\n".join(step_lines)


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


# ── Structural evaluation ─────────────────────────────────────────────────────

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
    return round(len(intersection) / len(union), 3)


def normalize_value(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())


def _remap_gt_tool_name(nl_name: str, tools: Dict[str, Any]) -> str:
    """
    ToolHop GT plans use the sub_question dict key as the tool name, but the
    model is shown (and trained to output) tool_info['name'] — the API-style name.
    This maps the NL sub-question back to the API name for fair structural comparison.
    """
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

    # Remap GT NL tool names to API names when tools dict is available
    gen_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gen_steps)
    if gt_uses_nl and tools and not gen_uses_nl:
        gt_steps = [
            {**s, "tool_name": _remap_gt_tool_name(s["tool_name"], tools)}
            for s in gt_steps
        ]
        gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)

    step_count_match     = len(gen_steps) == len(gt_steps)
    correct_exact_tools  = 0
    total_functional     = 0.0
    total_params_correct = 0
    total_params         = 0
    correct_deps         = 0
    total_deps           = 0
    step_details         = []

    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt  = gt_steps[i]  if i < len(gt_steps)  else None
        detail: Dict[str, Any] = {"step_id": i}

        if gen and gt:
            exact_ok = gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower()
            detail["exact_tool_correct"] = exact_ok
            if exact_ok:
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
            detail["exact_tool_correct"]    = False
            detail["functional_tool_score"] = 0.0
            detail["param_comparison"]      = None
            detail["dependency_refs_match"] = False

        step_details.append(detail)

    n_gt           = len(gt_steps)
    exact_tool_acc = correct_exact_tools / n_gt
    functional_acc = total_functional / n_gt
    param_accuracy = total_params_correct / total_params if total_params > 0 else 0.0
    dep_accuracy   = correct_deps / total_deps if total_deps > 0 else 1.0

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
        "exact_match":              step_count_match and exact_tool_acc == 1.0 and param_accuracy == 1.0,
        "functional_match":         step_count_match and functional_acc >= 0.5 and param_accuracy >= 0.5,
        "param_only_match":         param_accuracy >= 0.5,
        "step_details":             step_details,
    }


# ── Judge client ──────────────────────────────────────────────────────────────

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


def _format_tools_for_judge(tools: Dict[str, Any]) -> str:
    if not tools:
        return ""
    lines = ["Available Tools:"]
    seen = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in seen:
            seen[name] = tool_info
    for name, tool_info in seen.items():
        props = tool_info.get("parameters", {}).get("properties", {})
        params_str = ", ".join(f"{k}: {v.get('type','any')}" for k, v in props.items())
        lines.append(f"- {name}({params_str})")
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

    content = ""
    for attempt in range(retries):
        try:
            resp = requests.post(JUDGE_SERVER_URL, json=payload, timeout=120)
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
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt < retries - 1:
                time.sleep(2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            try:
                match = re.search(r'"quality_score"\s*:\s*(\d+)', content)
                if match:
                    return {
                        "quality_score": max(0, min(100, int(match.group(1)))),
                        "success_prediction": "uncertain",
                        "reasoning": "partial parse",
                        "issues": [], "confidence": 0.5, "_full_parse": False,
                    }
            except Exception:
                pass
            break

    return {"quality_score": 0, "success_prediction": "no",
            "reasoning": "judge call failed", "issues": [], "confidence": 0.0,
            "_full_parse": False}


# ── ReAct model wrapper ───────────────────────────────────────────────────────

class ReactModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"✓ Model loaded on {self.device}")

    def generate(self, question: str, tools: Dict[str, Any],
                 mode: str = "zero-shot",
                 temperature: float = 0.0,
                 max_new_tokens: int = 768) -> Tuple[str, str]:
        messages = build_chat_messages(question, tools, mode)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw_output = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        extracted_plan = extract_plan_from_react_output(raw_output)
        return raw_output, extracted_plan


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_parquet(parquet_path: str, perfect_only: bool = False) -> List[Dict]:
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

    return examples


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_react(
    model: ReactModel,
    examples: List[Dict],
    perfect_gt_by_qid: Dict[int, str],
    mode: str = "zero-shot",
    temperature: float = 0.0,
    max_new_tokens: int = 768,
    judge_max_tokens: int = 300,
    return_raw: bool = False,
) -> List[Dict]:
    results = []
    empty_plan_count = 0

    for ex in tqdm(examples, desc=f"ReAct ({mode}) evaluation"):
        question     = ex["question"]
        tools        = ex["tools"]
        ground_truth = ex["ground_truth"]
        gt_steps     = parse_plan_steps(ground_truth)

        raw_output, extracted_plan = model.generate(
            question, tools, mode=mode,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        if not extracted_plan.strip():
            empty_plan_count += 1

        judge_ann = score_plan_via_judge(
            question, extracted_plan or "(empty plan)", tools,
            max_tokens=judge_max_tokens
        )

        gen_steps = parse_plan_steps(extracted_plan)
        # Perfect-GT fix: artifact error types have deliberately flawed GT
        if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                and ex["query_id"] in perfect_gt_by_qid):
            _struct_gt_steps = parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        else:
            _struct_gt_steps = gt_steps
        struct_eval = evaluate_plan_vs_gt(gen_steps, _struct_gt_steps, tools=tools)

        judge_success  = judge_ann["quality_score"] >= 80
        ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)

        if ref_is_perfect:
            error_type_handled = judge_success
        else:
            error_type_handled = judge_ann["quality_score"] >= ex["quality_score"]

        result = {
            "query_id":              ex["query_id"],
            "question":              question,
            "error_type":            ex["error_type"],
            "ref_quality_score":     ex["quality_score"],
            "ref_is_perfect":        ref_is_perfect,
            "ground_truth":          ground_truth,
            "generated_plan":        extracted_plan,
            "n_extracted_steps":     len(gen_steps),
            "judge_success":         judge_success,
            "judge_score":           judge_ann["quality_score"],
            "judge_success_pred":    judge_ann["success_prediction"],
            "judge_confidence":      judge_ann["confidence"],
            "judge_full_parse":      judge_ann.get("_full_parse", False),
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
            "judge_agrees_with_ref":      (ref_is_perfect == judge_success),
            "react_mode":            mode,
            "react_temperature":     temperature,
        }

        if return_raw:
            result["raw_react_output"] = raw_output

        results.append(result)

    if empty_plan_count > 0:
        print(f"\n  ⚠  {empty_plan_count}/{len(examples)} examples produced no parseable plan steps.")
        print("     Check that the model is following the Step N: {{N}} = tool(...) format.")

    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(results: List[Dict], label: str) -> Dict:
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}

    scores       = [r["judge_score"]              for r in results]
    func_tools   = [r["functional_tool_accuracy"] for r in results]
    param_accs   = [r["param_accuracy"]           for r in results]
    dep_accs     = [r["dependency_accuracy"]      for r in results]

    judge_success_rate    = float(np.mean([r["judge_success"]      for r in results]))
    error_handled_rate    = float(np.mean([r["error_type_handled"] for r in results]))
    exact_match_rate      = float(np.mean([r["exact_match"]        for r in results]))
    functional_match_rate = float(np.mean([r["functional_match"]   for r in results]))
    param_only_match_rate = float(np.mean([r["param_only_match"]   for r in results]))
    step_match_rate       = float(np.mean([r["step_count_match"]   for r in results]))
    full_parse_rate       = float(np.mean([r.get("judge_full_parse", False) for r in results]))
    empty_plan_rate       = float(np.mean([r["generated_n_steps"] == 0 for r in results]))

    error_types = sorted(set(r["error_type"] for r in results))
    per_error   = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n":                       len(sub),
            "judge_success_rate":      float(np.mean([r["judge_success"]             for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"]        for r in sub])),
            "mean_judge_score":        float(np.mean([r["judge_score"]               for r in sub])),
            "functional_tool_acc":     float(np.mean([r["functional_tool_accuracy"]  for r in sub])),
            "mean_param_accuracy":     float(np.mean([r["param_accuracy"]            for r in sub])),
            "exact_match_rate":        float(np.mean([r["exact_match"]               for r in sub])),
            "functional_match_rate":   float(np.mean([r["functional_match"]          for r in sub])),
            "param_only_match_rate":   float(np.mean([r["param_only_match"]          for r in sub])),
            "step_count_match_rate":   float(np.mean([r["step_count_match"]          for r in sub])),
        }

    success_dist = {}
    for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100 * c / n, 1)}

    stats = {
        "label":              label,
        "method":             "ReAct (prompting)",
        "react_mode":         results[0].get("react_mode", "unknown"),
        "n_examples":         n,
        "gt_uses_nl_tools":   bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(full_parse_rate, 3),
        "empty_plan_rate":    round(empty_plan_rate, 3),

        "accuracy": {
            "judge_success_rate":  round(judge_success_rate, 3),
            "error_handled_rate":  round(error_handled_rate, 3),
        },
        "judge_scores": {
            "mean":       round(float(np.mean(scores)), 2),
            "median":     round(float(np.median(scores)), 2),
            "std":        round(float(np.std(scores)), 2),
            "pct_gte_80": round(100 * sum(s >= 80  for s in scores) / n, 1),
            "pct_eq_100": round(100 * sum(s == 100 for s in scores) / n, 1),
        },
        "structural": {
            "exact_match_rate":         round(exact_match_rate, 3),
            "functional_match_rate":    round(functional_match_rate, 3),
            "param_only_match_rate":    round(param_only_match_rate, 3),
            "step_count_match_rate":    round(step_match_rate, 3),
            "mean_functional_tool_acc": round(float(np.mean(func_tools)), 3),
            "mean_param_accuracy":      round(float(np.mean(param_accs)), 3),
            "mean_dependency_accuracy": round(float(np.mean(dep_accs)), 3),
        },
        "success_prediction_dist": success_dist,
        "per_error_type":          per_error,
    }

    W = 70
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  Method    : ReAct prompting ({results[0].get('react_mode','?')}-shot)")
    print(f"  N examples: {n}")
    if stats["gt_uses_nl_tools"]:
        print(f"  ⚠  GT uses NL tool names — exact_tool_accuracy unreliable")
    if empty_plan_rate > 0.05:
        print(f"  ⚠  Empty plan rate: {100*empty_plan_rate:.1f}% — model not following format")
    if full_parse_rate < 0.9:
        print(f"  ⚠  Judge full-parse rate: {100*full_parse_rate:.0f}%")

    print(f"\n  ── Primary Accuracy ──────────────────────────────────────────")
    print(f"  Judge success (score≥80) : {100*judge_success_rate:.1f}%")
    print(f"  Error type handled       : {100*error_handled_rate:.1f}%")

    print(f"\n  ── Judge Scores ──────────────────────────────────────────────")
    print(f"  Mean / Median / Std : {stats['judge_scores']['mean']:.1f} / "
          f"{stats['judge_scores']['median']:.1f} / {stats['judge_scores']['std']:.1f}")
    print(f"  ≥80 (good)          : {stats['judge_scores']['pct_gte_80']:.1f}%")
    print(f"  =100 (perfect)      : {stats['judge_scores']['pct_eq_100']:.1f}%")

    print(f"\n  ── Structural Metrics (vs GT) ─────────────────────────────────")
    print(f"  Exact match              : {100*exact_match_rate:.1f}%")
    print(f"  Functional match         : {100*functional_match_rate:.1f}%")
    print(f"  Param-only match (≥50%) : {100*param_only_match_rate:.1f}%")
    print(f"  Step count match         : {100*step_match_rate:.1f}%")
    print(f"  Mean functional tool acc : {np.mean(func_tools):.3f}")
    print(f"  Mean param accuracy      : {np.mean(param_accs):.3f}")
    print(f"  Mean dependency accuracy : {np.mean(dep_accs):.3f}")

    print(f"\n  ── Success Prediction ────────────────────────────────────────")
    for pred, d in success_dist.items():
        print(f"    {pred:12s}: {d['count']:3d}  ({d['pct']:.1f}%)")

    if len(error_types) > 1:
        print(f"\n  ── Per Error-Type ─────────────────────────────────────────────")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ReAct baseline for ToolHop tool planning")
    parser.add_argument("--model",               required=True,
                        help="HF model path (base, SFT, or any instruction-tuned model)")
    parser.add_argument("--test-parquet",         required=True)
    parser.add_argument("--mode",                default="zero-shot",
                        choices=["zero-shot", "few-shot"],
                        help="Prompting mode (default: zero-shot)")
    parser.add_argument("--temperature",          type=float, default=0.0,
                        help="Generation temperature (0 = greedy, recommended for ReAct)")
    parser.add_argument("--max-new-tokens",       type=int,   default=768,
                        help="Max tokens to generate (more than planner — includes Thought lines)")
    parser.add_argument("--judge-max-tokens",     type=int,   default=300)
    parser.add_argument("--output",              default="react_results.json")
    parser.add_argument("--stats-output",        default=None)
    parser.add_argument("--return-raw",          action="store_true",
                        help="Save raw ReAct output (including Thought lines) in results")
    parser.add_argument("--device",             default="cuda:0")
    parser.add_argument("--judge_url",           default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--perfect-only",        action="store_true")
    parser.add_argument("--full",                action="store_true")
    parser.add_argument("--limit",               type=int, default=None,
                        help="Evaluate only first N examples (for quick sanity checks)")
    args = parser.parse_args()

    if not args.perfect_only and not args.full:
        parser.error("Specify at least one of --perfect-only or --full")

    stats_output = args.stats_output or args.output.replace(".json", ".stats.json")

    global JUDGE_SERVER_URL
    JUDGE_SERVER_URL = args.judge_url

    try:
        r = requests.get(JUDGE_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5)
        print(f"✅ Judge server healthy: {r.json()}")
    except Exception as e:
        print(f"❌ Judge server not reachable: {e}")
        return

    model      = ReactModel(args.model, device=args.device)
    all_output = {"config": vars(args), "runs": {}}
    all_stats  = {"config": vars(args), "runs": {}}

    # Load perfect GT lookup (needed for artifact error type structural comparison)
    perfect_gt_by_qid = load_perfect_gt_from_parquet(args.test_parquet)

    if args.perfect_only:
        print("\nLoading perfect-only test examples...")
        examples = load_test_parquet(args.test_parquet, perfect_only=True)
        if args.limit:
            examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_react(
            model, examples,
            perfect_gt_by_qid=perfect_gt_by_qid,
            mode=args.mode,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            judge_max_tokens=args.judge_max_tokens,
            return_raw=args.return_raw,
        )
        stats = compute_stats(results, f"PERFECT-ONLY  ReAct ({args.mode})")
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"]  = stats

    if args.full:
        print("\nLoading full test set...")
        examples = load_test_parquet(args.test_parquet, perfect_only=False)
        if args.limit:
            examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_react(
            model, examples,
            perfect_gt_by_qid=perfect_gt_by_qid,
            mode=args.mode,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            judge_max_tokens=args.judge_max_tokens,
            return_raw=args.return_raw,
        )
        stats = compute_stats(results, f"FULL TEST SET  ReAct ({args.mode})")
        all_output["runs"]["full"] = results
        all_stats["runs"]["full"]  = stats

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_output).parent.mkdir(parents=True, exist_ok=True)


    with open(args.output, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"Results saved to: {args.output}")

    with open(stats_output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats   saved to: {stats_output}")


if __name__ == "__main__":
    main()