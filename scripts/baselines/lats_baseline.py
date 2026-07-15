#!/usr/bin/env python3
"""
LATS Baseline for ToolHop Tool Planning

Implements Language Agent Tree Search (Zhou et al., ICML 2024) adapted for
offline ToolHop tool plan generation.

LATS key components (from paper):
  Selection       UCT-guided node selection
  Expansion       Generate n candidate plans from current context
  Evaluation      Score each plan with judge server (our reward signal)
  Simulation      Skip — each action = complete plan (programming LATS variant)
  Backpropagation Update node values up the tree
  Reflection      LM generates verbal critique of failed plans; stored as context
                  for future iterations

ToolHop adaptation:
  Since ToolHop has no real execution environment, we follow the programming
  variant of LATS (Section 5.2 of the paper) where:
    - Each action = a complete tool plan (not a single step)
    - Environment feedback = judge quality score (0-100) in place of test results
    - Observations = judge reasoning + score (returned at judge_eval_max_tokens)
    - Reflection threshold: plans scoring < reflect_threshold trigger LM critique
    - Self-consistency component: agreement between plans generated at same node

  The value function mirrors Eq. 2 from the paper:
    V(s) = λ * judge_score + (1 - λ) * self_consistency_score

  where self_consistency_score = fraction of plans at this node that agree on
  the core tool sequence (measured by functional_tool_match overlap).

Output format matches best_of_n_selection.py and react_baseline.py exactly.

Usage:
    # Start judge server first (needs full JSON for reflections):
    CUDA_VISIBLE_DEVICES=6 python judge_server.py \
        --model ${FORTE_ROOT}/judge_finetuning/models/judge/merged --port 8002 --batch-size 32 --batch-timeout-ms 200

    CUDA_VISIBLE_DEVICES=1 python lats_baseline.py \
        --model ${FORTE_ROOT}/planner_finetuning/checkpoints_planner_sft-llama-3b/global_step_147 \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --full \
        --n 3 --k 2 \
        --output lats_results_Llama-3B-Instruct.json \
        --stats-output lats_stats_Llama-3B-Instruct.json \
        --judge_url http://localhost:8001/v1/chat/completions \
        --checkpoint lats_checkpoint_llama3b.json
        

    # Quick sanity check (5 examples, 3 iterations):
    python lats_baseline.py \\
        --model /path/to/sft/model \\
        --test-parquet data/verl_rl_full_clean/test.parquet \\
        --full --limit 5 --n 3 --k 3 \\
        --return-raw --output lats_sanity.json
"""



#!/usr/bin/env python3
"""
LATS Baseline for ToolHop Tool Planning — FAST VERSION

Speed improvements vs original:
  1. Parallel judge scoring via ThreadPoolExecutor (n calls fire simultaneously)
  2. Batched plan generation (single GPU forward pass for n plans, like best_of_n)
  3. Checkpointing: saves after every example, resumes from last completed

Checkpointing usage:
    python lats_baseline_fast.py ... --checkpoint lats_checkpoint_full.json
    # If interrupted, rerun with same --checkpoint flag to resume

"""

import json
import math
import re
import time
import argparse
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer

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

SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)

JUDGE_SERVER_URL = "http://localhost:8002/v1/chat/completions"
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

# Persistent thread pool — shared across all judge calls
_judge_pool = ThreadPoolExecutor(max_workers=32)


def load_perfect_gt_from_parquet(parquet_path: str) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    perfect_gt: dict = {}
    for ei, rm in zip(extra_infos, reward_models):
        if isinstance(ei, str): ei = json.loads(ei)
        if isinstance(rm, str): rm = json.loads(rm)
        if not isinstance(ei, dict) or not isinstance(rm, dict): continue
        if (str(ei.get("error_type", "")) == "none" and int(ei.get("quality_score", 0)) >= 100):
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


def build_plan_prompt(question: str, tools: Dict[str, Any], reflections: List[str]) -> str:
    tools_str = format_tools(tools)
    reflection_block = ""
    if reflections:
        reflection_block = (
            "\n\nPrevious attempts at this query failed. "
            "Use the following reflections to improve your plan:\n"
            + "\n".join(f"- {r}" for r in reflections) + "\n"
        )
    return (
        f"Generate a tool execution plan to answer this query.\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n"
        f"{reflection_block}\n"
        f"Generate a step-by-step plan using the available tools. Each step should:\n"
        f"1. Call exactly one tool\n"
        f"2. Use output variables {{{{0}}}}, {{{{1}}}}, {{{{2}}}}, etc. for results\n"
        f"3. Reference previous step outputs using {{{{N}}}} syntax\n"
        f"4. Provide all required parameters\n\n"
        f"Format each step as: Step N: {{{{N}}}} = tool_name(param1=value1, param2=value2, ...)"
    )


def build_reflection_prompt(question: str, tools: Dict[str, Any],
                             plan: str, score: int, reasoning: str) -> str:
    tools_str = format_tools(tools)
    return (
        f"You attempted to generate a tool execution plan but it received a low quality score.\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n\n"
        f"Your plan:\n{plan}\n\n"
        f"Quality score: {score}/100\n"
        f"Judge reasoning: {reasoning}\n\n"
        f"In 2-3 sentences, diagnose what went wrong and describe a specific strategy "
        f"to generate a better plan. Focus on concrete tool selection, parameter, "
        f"or dependency issues. Do not rewrite the plan — just describe the fix."
    )


def _format_tools_for_judge(tools: Dict[str, Any]) -> str:
    if not tools: return ""
    lines = ["Available Tools:"]
    seen = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in seen: seen[name] = tool_info
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
                if last != -1: content = content[:last + 1]
            annotation = json.loads(content)
            annotation["quality_score"] = max(0, min(100, int(annotation.get("quality_score", 50))))
            annotation["confidence"]    = max(0.0, min(1.0, float(annotation.get("confidence", 0.5))))
            annotation["_full_parse"]   = True
            return annotation
        except (requests.ConnectionError, requests.Timeout):
            if attempt < retries - 1: time.sleep(2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            match = re.search(r'"quality_score"\s*:\s*(\d+)', content)
            if match:
                return {"quality_score": max(0, min(100, int(match.group(1)))),
                        "success_prediction": "uncertain", "reasoning": "partial parse",
                        "issues": [], "confidence": 0.5, "_full_parse": False}
            break
    return {"quality_score": 0, "success_prediction": "no",
            "reasoning": "judge call failed", "issues": [], "confidence": 0.0, "_full_parse": False}


def parse_plan_steps(plan_text: str) -> List[Dict]:
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("Step "): continue
        try:
            step_match = re.match(r"Step (\d+):", line)
            if not step_match: continue
            step_id = int(step_match.group(1))
            var_match = re.search(r"(\{\{\d+\}\})\s*=", line)
            if not var_match: continue
            output_var = var_match.group(1)
            tool_match = re.search(r"=\s*([^\(]+)\((.*)\)\s*$", line)
            if not tool_match:
                tool_match_empty = re.search(r"=\s*([^\(]+)\(\)\s*$", line)
                if tool_match_empty:
                    tool_name, params = tool_match_empty.group(1).strip(), {}
                else:
                    continue
            else:
                tool_name = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                params = {}
                if params_str:
                    param_parts, current, depth, in_str, str_char = [], "", 0, False, None
                    for ch in params_str:
                        if ch in ('"', "'") and (not in_str or ch == str_char):
                            in_str = not in_str
                            str_char = ch if in_str else None
                        if not in_str:
                            if ch in "([{": depth += 1
                            elif ch in ")]}": depth -= 1
                            elif ch == "," and depth == 0:
                                param_parts.append(current.strip())
                                current = ""
                                continue
                        current += ch
                    if current.strip(): param_parts.append(current.strip())
                    for part in param_parts:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            params[k.strip()] = v.strip()
            steps.append({"step_id": step_id, "output_variable": output_var,
                          "tool_name": tool_name, "parameters": params})
        except Exception:
            continue
    return steps


def _is_nl_tool_name(name: str) -> bool:
    return len(name.split()) > 4 or name.endswith("?")

def _functional_tool_match(gen_name: str, gt_name: str) -> float:
    STOP = {"what","is","the","of","in","a","an","and","or","to","how","many","who",
            "which","are","was","were","be","been","at","on","for","with","that",
            "this","it","its","from"}
    def keywords(s):
        words = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
        return {w for w in words if w not in STOP and len(w) > 2}
    gen_kw, gt_kw = keywords(gen_name), keywords(gt_name)
    if not gen_kw or not gt_kw: return 0.0
    return round(len(gen_kw & gt_kw) / len(gen_kw | gt_kw), 3)

def normalize_value(v): return " ".join(str(v).strip().strip("\"'").lower().split())

def _remap_gt_tool_name(nl_name, tools):
    if nl_name in tools:
        api = tools[nl_name].get("name")
        if api: return api
    for key, tool_info in tools.items():
        if nl_name in key or key in nl_name:
            api = tool_info.get("name")
            if api: return api
    return nl_name

def evaluate_plan_vs_gt(gen_steps, gt_steps, tools=None):
    empty = {"valid": False, "error": "", "step_count_match": False,
             "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
             "param_accuracy": 0.0, "dependency_accuracy": 0.0,
             "exact_match": False, "functional_match": False, "param_only_match": False,
             "gt_uses_nl_tool_names": False, "step_details": []}
    if not gen_steps: return {**empty, "error": "no steps generated"}
    if not gt_steps:  return {**empty, "error": "no ground truth steps"}
    gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)
    gen_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gen_steps)
    if gt_uses_nl and tools and not gen_uses_nl:
        gt_steps = [{**s, "tool_name": _remap_gt_tool_name(s["tool_name"], tools)} for s in gt_steps]
        gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)
    step_count_match = len(gen_steps) == len(gt_steps)
    correct_exact = total_functional = total_pc = total_p = correct_deps = total_deps = 0
    step_details = []
    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt  = gt_steps[i]  if i < len(gt_steps)  else None
        detail: Dict[str, Any] = {"step_id": i}
        if gen and gt:
            exact_ok = gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower()
            detail["exact_tool_correct"] = exact_ok
            if exact_ok: correct_exact += 1
            func = _functional_tool_match(gen["tool_name"], gt["tool_name"])
            detail["functional_tool_score"] = func
            total_functional += func
            gt_keys, gen_keys = set(gt["parameters"]), set(gen["parameters"])
            common = gt_keys & gen_keys
            incorrect = []
            for k in common:
                gv = normalize_value(gt["parameters"][k])
                dv = normalize_value(gen["parameters"][k])
                if gv == dv or gv in dv or dv in gv: total_pc += 1
                else: incorrect.append({"param": k, "generated": gen["parameters"][k], "ground_truth": gt["parameters"][k]})
            total_p += len(gt_keys)
            detail["param_comparison"] = {"total_gt_params": len(gt_keys), "correct": len(common)-len(incorrect),
                                          "missing": list(gt_keys-gen_keys), "extra": list(gen_keys-gt_keys), "incorrect": incorrect}
            gt_refs  = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            total_deps  += len(gt_refs)
            correct_deps += len(gt_refs & gen_refs)
            detail["dependency_refs_match"] = (gt_refs == gen_refs)
        else:
            detail.update({"exact_tool_correct": False, "functional_tool_score": 0.0,
                           "param_comparison": None, "dependency_refs_match": False})
        step_details.append(detail)
    n_gt = len(gt_steps)
    ea = correct_exact / n_gt
    fa = total_functional / n_gt
    pa = total_pc / total_p if total_p > 0 else 0.0
    da = correct_deps / total_deps if total_deps > 0 else 1.0
    return {"valid": True, "gt_uses_nl_tool_names": gt_uses_nl, "step_count_match": step_count_match,
            "generated_steps": len(gen_steps), "ground_truth_steps": n_gt,
            "exact_tool_accuracy": ea, "functional_tool_accuracy": fa,
            "param_accuracy": pa, "dependency_accuracy": da,
            "exact_match": step_count_match and ea == 1.0 and pa == 1.0,
            "functional_match": step_count_match and fa >= 0.5 and pa >= 0.5,
            "param_only_match": pa >= 0.5, "step_details": step_details}


class LATSNode:
    def __init__(self, reflections: List[str], parent: Optional["LATSNode"] = None):
        self.reflections = reflections
        self.parent = parent
        self.children: List["LATSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.best_plan: str = ""
        self.best_score: int = 0
        self.best_ann: Dict = {}
        self.expanded_plans: List[str] = []
        self.expanded_scores: List[int] = []

    def uct(self, exploration_weight: float = 1.0) -> float:
        if self.visits == 0: return float("inf")
        if self.parent is None or self.parent.visits == 0: return self.value
        return self.value + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def backpropagate(self, reward: float):
        node = self
        while node is not None:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent


class LATSModel:
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

    def _generate(self, messages, temperature, max_new_tokens):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
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
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_plan(self, question, tools, reflections, temperature, max_new_tokens=512):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_plan_prompt(question, tools, reflections)}]
        return self._generate(messages, temperature, max_new_tokens)

    def generate_n_plans(self, question, tools, reflections, n, max_new_tokens=512):
        """
        FAST: single batched GPU forward pass for all n plans.
        Falls back to sequential on OOM.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_plan_prompt(question, tools, reflections)}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

        input_ids      = inputs["input_ids"].expand(n, -1)
        attention_mask = inputs["attention_mask"].expand(n, -1)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            return [self.tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
                    for i in range(n)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return [self.generate_plan(question, tools, reflections, temperature=0.7,
                                       max_new_tokens=max_new_tokens) for _ in range(n)]

    def generate_reflection(self, question, tools, plan, score, reasoning, max_new_tokens=200):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_reflection_prompt(question, tools, plan, score, reasoning)}]
        return self._generate(messages, temperature=0.0, max_new_tokens=max_new_tokens)


def compute_self_consistency(plans):
    if len(plans) <= 1: return 1.0
    def tool_seq(plan):
        return [s["tool_name"].strip().lower() for s in parse_plan_steps(plan)]
    def seq_sim(a, b):
        if not a or not b: return 0.0
        sa = set(f"{i}:{t}" for i, t in enumerate(a))
        sb = set(f"{i}:{t}" for i, t in enumerate(b))
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    seqs = [tool_seq(p) for p in plans]
    pairs = [(i, j) for i in range(len(seqs)) for j in range(i+1, len(seqs))]
    return float(np.mean([seq_sim(seqs[i], seqs[j]) for i, j in pairs])) if pairs else 1.0


def lats_search(question, tools, model, n=5, k=10, exploration_weight=1.0,
                lm_weight=0.5, reflect_threshold=70, judge_max_tokens=300, max_new_tokens=512):
    root = LATSNode(reflections=[])
    all_plans, all_scores, all_annotations, all_reflections = [], [], [], []
    iterations_used = 0
    early_stop = False

    def select_node(node):
        while node.children:
            node = max(node.children, key=lambda c: c.uct(exploration_weight))
        return node

    for iteration in range(k):
        iterations_used = iteration + 1
        node = select_node(root)

        plans = model.generate_n_plans(question, tools, node.reflections, n=n, max_new_tokens=max_new_tokens)

        # FAST: parallel judge scoring
        futures = {
            _judge_pool.submit(score_plan_via_judge, question, p, tools, judge_max_tokens): i
            for i, p in enumerate(plans)
        }
        annotations = [None] * len(plans)
        for future in as_completed(futures):
            annotations[futures[future]] = future.result()
        scores = [a["quality_score"] for a in annotations]

        node.expanded_plans  = plans
        node.expanded_scores = scores
        all_plans.extend(plans)
        all_scores.extend(scores)
        all_annotations.extend(annotations)

        lm_score   = np.mean(scores) / 100.0
        sc_score   = compute_self_consistency(plans)
        node_value = lm_weight * lm_score + (1 - lm_weight) * sc_score

        best_idx = int(np.argmax(scores))
        if scores[best_idx] > node.best_score:
            node.best_score = scores[best_idx]
            node.best_plan  = plans[best_idx]
            node.best_ann   = annotations[best_idx]

        node.backpropagate(node_value)

        if scores[best_idx] >= 100:
            early_stop = True
            break

        if scores[best_idx] < reflect_threshold:
            reasoning  = annotations[best_idx].get("reasoning", "No reasoning available.")
            reflection = model.generate_reflection(question, tools, node.best_plan, node.best_score, reasoning)
            all_reflections.append(reflection)
            child = LATSNode(reflections=node.reflections + [reflection], parent=node)
            node.children.append(child)

    if not all_plans:
        final_plan, final_score = "", 0
        final_ann = {"quality_score": 0, "success_prediction": "no", "confidence": 0.0,
                     "reasoning": "no plans generated", "_full_parse": False}
    else:
        best_idx   = int(np.argmax(all_scores))
        final_score = all_scores[best_idx]
        final_plan  = all_plans[best_idx]
        final_ann   = all_annotations[best_idx]

    return {"best_plan": final_plan, "best_score": final_score, "best_annotation": final_ann,
            "all_scores": all_scores, "all_plans": all_plans, "reflections": all_reflections,
            "iterations_used": iterations_used, "early_stop": early_stop,
            "n_nodes_expanded": iterations_used, "n_plans_explored": len(all_plans)}


def load_test_parquet(parquet_path, perfect_only=False):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    examples = []
    for i in range(len(extra_infos)):
        ei = extra_infos[i]
        if isinstance(ei, str): ei = json.loads(ei)
        if not isinstance(ei, dict): ei = {}
        rm = reward_models[i]
        if isinstance(rm, str): rm = json.loads(rm)
        if not isinstance(rm, dict): rm = {}
        dj = ei.get("data_json", "{}")
        if isinstance(dj, str): dj = json.loads(dj)
        if not isinstance(dj, dict): dj = {}
        error_type    = str(ei.get("error_type", "none"))
        quality_score = int(ei.get("quality_score", 0))
        if perfect_only and not (error_type == "none" and quality_score >= 100): continue
        examples.append({"question": dj.get("question",""), "tools": dj.get("tools",{}),
                         "ground_truth": rm.get("ground_truth",""), "error_type": error_type,
                         "quality_score": quality_score, "query_id": ei.get("query_id",-1)})
    return examples


def evaluate_lats(model, examples, perfect_gt_by_qid, n=5, k=10, exploration_weight=1.0,
                  lm_weight=0.5, reflect_threshold=70, judge_max_tokens=300,
                  max_new_tokens=512, return_raw=False, checkpoint_path=None):
    # ── Checkpointing: resume from where we left off ──────────────────────────
    results = []
    completed_keys = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        completed_keys = {(r["query_id"], r["error_type"]) for r in results}
        print(f"  Resuming: {len(results)} done, {len(examples)-len(completed_keys)} remaining")

    def _save_checkpoint():
        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)

    for ex in tqdm(examples, desc=f"LATS (n={n}, k={k}) evaluation"):
        # Skip already-completed examples
        if (ex["query_id"], ex["error_type"]) in completed_keys:
            continue

        question, tools, ground_truth = ex["question"], ex["tools"], ex["ground_truth"]
        gt_steps = parse_plan_steps(ground_truth)

        search_result = lats_search(
            question=question, tools=tools, model=model,
            n=n, k=k, exploration_weight=exploration_weight,
            lm_weight=lm_weight, reflect_threshold=reflect_threshold,
            judge_max_tokens=judge_max_tokens, max_new_tokens=max_new_tokens,
        )

        best_plan, best_score, best_ann = search_result["best_plan"], search_result["best_score"], search_result["best_annotation"]
        gen_steps = parse_plan_steps(best_plan)
        struct_gt = (parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
                     if ex["error_type"] in ARTIFACT_ERROR_TYPES and ex["query_id"] in perfect_gt_by_qid
                     else gt_steps)
        struct_eval = evaluate_plan_vs_gt(gen_steps, struct_gt, tools=tools)

        judge_success  = best_score >= 80
        ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)
        error_type_handled = judge_success if ref_is_perfect else best_score >= ex["quality_score"]

        result = {
            "query_id": ex["query_id"], "question": question,
            "error_type": ex["error_type"], "ref_quality_score": ex["quality_score"],
            "ref_is_perfect": ref_is_perfect, "ground_truth": ground_truth,
            "best_plan": best_plan, "n_extracted_steps": len(gen_steps),
            "judge_success": judge_success, "judge_score": best_score,
            "judge_success_pred": best_ann.get("success_prediction","uncertain"),
            "judge_confidence": best_ann.get("confidence", 0.5),
            "judge_full_parse": best_ann.get("_full_parse", False),
            "iterations_used": search_result["iterations_used"],
            "early_stop": search_result["early_stop"],
            "n_plans_explored": search_result["n_plans_explored"],
            "n_reflections": len(search_result["reflections"]),
            "all_scores": search_result["all_scores"],
            "mean_explored_score": float(np.mean(search_result["all_scores"])),
            "gt_uses_nl_tool_names": struct_eval["gt_uses_nl_tool_names"],
            "exact_match": struct_eval["exact_match"],
            "functional_match": struct_eval["functional_match"],
            "param_only_match": struct_eval["param_only_match"],
            "step_count_match": struct_eval["step_count_match"],
            "generated_n_steps": struct_eval.get("generated_steps", 0),
            "gt_n_steps": struct_eval.get("ground_truth_steps", len(gt_steps)),
            "exact_tool_accuracy": struct_eval["exact_tool_accuracy"],
            "functional_tool_accuracy": struct_eval["functional_tool_accuracy"],
            "param_accuracy": struct_eval["param_accuracy"],
            "dependency_accuracy": struct_eval["dependency_accuracy"],
            "error_type_handled": error_type_handled,
            "judge_agrees_with_ref": (ref_is_perfect == judge_success),
        }
        if return_raw:
            result["reflections"] = search_result["reflections"]
            result["all_plans"]   = search_result["all_plans"]

        results.append(result)
        _save_checkpoint()  # Save after every example for safe resume

    return results


def compute_stats(results, label, n=5, k=10):
    total = len(results)
    if total == 0: return {"label": label, "n": 0}
    scores     = [r["judge_score"]              for r in results]
    func_tools = [r["functional_tool_accuracy"] for r in results]
    param_accs = [r["param_accuracy"]           for r in results]
    dep_accs   = [r["dependency_accuracy"]      for r in results]
    judge_success_rate    = float(np.mean([r["judge_success"]      for r in results]))
    error_handled_rate    = float(np.mean([r["error_type_handled"] for r in results]))
    exact_match_rate      = float(np.mean([r["exact_match"]        for r in results]))
    functional_match_rate = float(np.mean([r["functional_match"]   for r in results]))
    param_only_match_rate = float(np.mean([r["param_only_match"]   for r in results]))
    step_match_rate       = float(np.mean([r["step_count_match"]   for r in results]))
    full_parse_rate       = float(np.mean([r.get("judge_full_parse", False) for r in results]))
    mean_iters       = float(np.mean([r["iterations_used"]  for r in results]))
    mean_plans       = float(np.mean([r["n_plans_explored"] for r in results]))
    mean_reflections = float(np.mean([r["n_reflections"]    for r in results]))
    early_stop_rate  = float(np.mean([r["early_stop"]       for r in results]))
    error_types = sorted(set(r["error_type"] for r in results))
    per_error = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n": len(sub),
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
    for pred in ["yes","likely_yes","uncertain","likely_no","no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100*c/total, 1)}
    stats = {
        "label": label, "method": "LATS (MCTS + reflection)",
        "n_expanded": n, "k_iterations": k, "n_examples": total,
        "gt_uses_nl_tools": bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(full_parse_rate, 3),
        "accuracy": {"judge_success_rate": round(judge_success_rate, 3),
                     "error_handled_rate": round(error_handled_rate, 3)},
        "judge_scores": {"mean": round(float(np.mean(scores)), 2),
                         "median": round(float(np.median(scores)), 2),
                         "std": round(float(np.std(scores)), 2),
                         "pct_gte_80": round(100*sum(s>=80 for s in scores)/total, 1),
                         "pct_eq_100": round(100*sum(s==100 for s in scores)/total, 1)},
        "structural": {"exact_match_rate": round(exact_match_rate, 3),
                       "functional_match_rate": round(functional_match_rate, 3),
                       "param_only_match_rate": round(param_only_match_rate, 3),
                       "step_count_match_rate": round(step_match_rate, 3),
                       "mean_functional_tool_acc": round(float(np.mean(func_tools)), 3),
                       "mean_param_accuracy": round(float(np.mean(param_accs)), 3),
                       "mean_dependency_accuracy": round(float(np.mean(dep_accs)), 3)},
        "lats_search_stats": {"mean_iterations_used": round(mean_iters, 2),
                               "mean_plans_explored": round(mean_plans, 2),
                               "mean_reflections": round(mean_reflections, 2),
                               "early_stop_rate": round(early_stop_rate, 3),
                               "budget_utilization": round(mean_iters/k, 3)},
        "success_prediction_dist": success_dist,
        "per_error_type": per_error,
    }
    W = 70
    print(f"\n{'='*W}\n  {label}\n{'='*W}")
    print(f"  Method: LATS (n={n} expansions, k={k} iterations)  |  N={total}")
    if full_parse_rate < 0.9:
        print(f"  ⚠  Judge full-parse rate: {100*full_parse_rate:.0f}%")
    print(f"\n  ── Primary Accuracy ──────────────────────────────────────────")
    print(f"  Judge success (≥80) : {100*judge_success_rate:.1f}%")
    print(f"  Error handled       : {100*error_handled_rate:.1f}%")
    print(f"\n  ── Judge Scores ──────────────────────────────────────────────")
    print(f"  Mean/Median/Std : {stats['judge_scores']['mean']:.1f} / {stats['judge_scores']['median']:.1f} / {stats['judge_scores']['std']:.1f}")
    print(f"  ≥80 : {stats['judge_scores']['pct_gte_80']:.1f}%   =100 : {stats['judge_scores']['pct_eq_100']:.1f}%")
    print(f"\n  ── LATS Search Stats ─────────────────────────────────────────")
    print(f"  Mean iters : {mean_iters:.1f}/{k}   Mean plans : {mean_plans:.1f}   Reflections : {mean_reflections:.2f}   Early stop : {100*early_stop_rate:.1f}%")
    print(f"\n  ── Structural Metrics ────────────────────────────────────────")
    print(f"  Exact match : {100*exact_match_rate:.1f}%   Functional match : {100*functional_match_rate:.1f}%   Param match : {100*param_only_match_rate:.1f}%   Step match : {100*step_match_rate:.1f}%")
    print(f"  FuncTool acc : {np.mean(func_tools):.3f}   Param acc : {np.mean(param_accs):.3f}   Dep acc : {np.mean(dep_accs):.3f}")
    if len(error_types) > 1:
        print(f"\n  ── Per Error-Type ─────────────────────────────────────────────")
        hdr = f"  {'Error Type':28s}  {'N':>4}  {'Success%':>8}  {'Handled%':>8}  {'Judge':>6}  {'FuncTool%':>9}  {'Param%':>6}  {'ExMatch%':>9}  {'FuncMatch%':>10}"
        print(hdr)
        print("  " + "-"*(len(hdr)-2))
        for et, d in per_error.items():
            print(f"  {et:28s}  {d['n']:>4}  {100*d['judge_success_rate']:>8.1f}  {100*d['error_type_handled_rate']:>8.1f}  "
                  f"{d['mean_judge_score']:>6.1f}  {100*d['functional_tool_acc']:>9.1f}  {100*d['mean_param_accuracy']:>6.1f}  "
                  f"{100*d['exact_match_rate']:>9.1f}  {100*d['functional_match_rate']:>10.1f}")
    print()
    return stats


def main():
    parser = argparse.ArgumentParser(description="LATS baseline for ToolHop (fast version)")
    parser.add_argument("--model",             required=True)
    parser.add_argument("--test-parquet",       required=True)
    parser.add_argument("--n",                  type=int,   default=5)
    parser.add_argument("--k",                  type=int,   default=10)
    parser.add_argument("--exploration-weight", type=float, default=1.0)
    parser.add_argument("--lm-weight",          type=float, default=0.5)
    parser.add_argument("--reflect-threshold",  type=int,   default=70)
    parser.add_argument("--max-new-tokens",     type=int,   default=512)
    parser.add_argument("--judge-max-tokens",   type=int,   default=300)
    parser.add_argument("--output",             default="lats_results.json")
    parser.add_argument("--stats-output",       default=None)
    parser.add_argument("--return-raw",         action="store_true")
    parser.add_argument("--device",             default="cuda:0")
    parser.add_argument("--judge_url",          default="http://localhost:8002/v1/chat/completions")
    parser.add_argument("--perfect-only",       action="store_true")
    parser.add_argument("--full",               action="store_true")
    parser.add_argument("--limit",              type=int, default=None)
    parser.add_argument("--checkpoint",         default=None,
                        help="Checkpoint JSON path for safe resume. On restart, "
                             "completed examples are skipped automatically.")
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
        print(f"❌ Judge server not reachable: {e}"); return

    print(f"\nLATS budget: n={args.n} × k={args.k} = {args.n*args.k} judge calls/example")
    print(f"Speed vs original: ~{args.n}x faster judge scoring (parallel) + batched generation")

    model      = LATSModel(args.model, device=args.device)
    all_output = {"config": vars(args), "runs": {}}
    all_stats  = {"config": vars(args), "runs": {}}
    perfect_gt_by_qid = load_perfect_gt_from_parquet(args.test_parquet)

    if args.perfect_only:
        print("\nLoading perfect-only examples...")
        examples = load_test_parquet(args.test_parquet, perfect_only=True)
        if args.limit: examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        ckpt = args.checkpoint.replace(".json","_perfect.json") if args.checkpoint else None
        results = evaluate_lats(model, examples, perfect_gt_by_qid,
                                n=args.n, k=args.k, exploration_weight=args.exploration_weight,
                                lm_weight=args.lm_weight, reflect_threshold=args.reflect_threshold,
                                judge_max_tokens=args.judge_max_tokens, max_new_tokens=args.max_new_tokens,
                                return_raw=args.return_raw, checkpoint_path=ckpt)
        stats = compute_stats(results, f"PERFECT-ONLY LATS (n={args.n}, k={args.k})", n=args.n, k=args.k)
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"]  = stats

    if args.full:
        print("\nLoading full test set...")
        examples = load_test_parquet(args.test_parquet, perfect_only=False)
        if args.limit: examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        ckpt = args.checkpoint.replace(".json","_full.json") if args.checkpoint else None
        results = evaluate_lats(model, examples, perfect_gt_by_qid,
                                n=args.n, k=args.k, exploration_weight=args.exploration_weight,
                                lm_weight=args.lm_weight, reflect_threshold=args.reflect_threshold,
                                judge_max_tokens=args.judge_max_tokens, max_new_tokens=args.max_new_tokens,
                                return_raw=args.return_raw, checkpoint_path=ckpt)
        stats = compute_stats(results, f"FULL TEST SET LATS (n={args.n}, k={args.k})", n=args.n, k=args.k)
        all_output["runs"]["full"] = results
        all_stats["runs"]["full"]  = stats

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f: json.dump(all_output, f, indent=2)
    print(f"Results → {args.output}")
    with open(stats_output, "w") as f: json.dump(all_stats, f, indent=2)
    print(f"Stats   → {stats_output}")


if __name__ == "__main__":
    main()