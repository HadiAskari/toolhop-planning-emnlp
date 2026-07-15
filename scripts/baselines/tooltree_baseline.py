#!/usr/bin/env python3
"""
ToolTree Baseline for ToolHop Tool Planning

Implements ToolTree (Yang et al., ICLR 2026) — an MCTS-inspired planning paradigm
with dual LLM evaluation and bidirectional pruning. Adapted for ToolHop's
offline (no-execution) setting.

This implementation MIRRORS THE RELEASED CODEBASE rather than the paper's
Equation 1 wherever the two diverge. The README's reported numbers come from
the code, not the equation, so matching the code reproduces the algorithm
that produced ToolTree's published results.

DOCUMENTED CODE-VS-PAPER DIVERGENCES (kept faithful to the code):
  • UCT formula: paper Eq. 1 is
        Q(s,a) + λ · rpre(s,a) · sqrt( ln N(s) / N(s,a) )
    but src/mcts/node.py:best_child computes
        Q(s,a) + λ · (rpre or 1e-6) · sqrt(ln(N_parent + 1)) / (1 + N(s,a)) + jitter
    (PUCT-style: visit count linear in denominator, not under sqrt; prior
    floored at 1e-6; tiny random jitter for tiebreaking).
  • Pre-pruning: src/mcts/pruning.py:pre_prune filters candidates by τ_pre
    threshold only — there is NO top-K filter after pre-evaluation. Top-K
    is enforced at candidate-generation time via the planner prompt.
  • After expansion: src/mcts/tree_search.py:search picks new_children[0]
    (first survivor in iteration order) as the rollout target, not the
    highest-rpre child.
  • Early stopping: src/mcts/tree_search.py:_check_early_stop computes
        max(window[1:]) - window[0] < δ
    not window[-1] - window[0] (i.e. allows for intermediate peaks).

ALGORITHM (paper §3):
  Each MCTS rollout cycles through:
    1. Selection      — descend the tree via prior-augmented PUCT
    2. Pre-evaluation — judge scores rpre(s,a) for candidate next steps
    3. Expansion      — instantiate children that pass τ_pre
    4. Execution      — commit the selected tool call (offline: append to plan)
    5. Post-evaluation — judge scores rpost(s,a) on the resulting partial plan
    6. Backprop       — propagate rpost up to and INCLUDING the root
  After Rmax rollouts (or Q convergence), extract best path by greedy max-Q.

Bidirectional pruning:
    • Pre-pruning:  skip child creation if rpre(s,a) < τ_pre
    • Post-pruning: mark `is_expandable = False` if rpost(s,a) < τ_post
      (selection will skip non-expandable children but their q values
       still inform best-trajectory extraction)

OFFLINE ADAPTATION (key adjustment from paper):
ToolHop has no execution environment, so rpost reuses the cached rpre score
on each node (same partial-plan input → same judge output). Each (state,
action) pair triggers at most one judge call. The dual-eval design still
drives different MCTS behavior via:
  (1) rpre filters branches BEFORE they enter the tree (during expansion);
  (2) rpost uses a STRICTER threshold (0.4 vs 0.3) for post-pruning, so a
      branch that passed pre-prune can still be marked non-expandable;
  (3) rpre acts as a static prior in UCT exploration;
  (4) rpost drives Q-value updates during backprop.
A future extension could simulate hypothetical tool outputs via a small LLM
call to differentiate rpre/rpost more meaningfully (see TODO in _post_evaluate).

CANDIDATE GENERATION:
For efficiency, we generate K candidate next steps in a SINGLE planner call
(rather than K independent samples). The planner is prompted to produce K
DIFFERENT alternatives that do not repeat tools already in the partial plan.

OUTPUT FORMAT:
Matches best_of_n_selection.py / alpha_umi_baseline.py / tool_planner_baseline.py
so all baselines can be compared side-by-side via compute_stats().

KNOWN DATASET ARTIFACT (ToolHop):
Ground truth tool names are natural-language sub-questions.
exact_tool_accuracy will be ~0; use functional_tool_accuracy instead.

Usage:
    # Start judge server first (use a healthy batch size to absorb concurrent load):
    CUDA_VISIBLE_DEVICES=5 python judge_server.py \
        --model /path/to/judge/merged --port 8005 \
        --batch-size 32 --batch-timeout-ms 200

    # Run ToolTree with parallelism (RECOMMENDED — much faster than sequential):
    CUDA_VISIBLE_DEVICES=2 python tooltree_baseline.py \
        --planner-model meta-llama/Llama-3.2-3B-Instruct \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --full \
        --num-workers 4 --judge-concurrency 4 \
        --judge_url http://localhost:8003/v1/chat/completions \
        --output tooltree_results.json

    # Ablate post-evaluation pruning (already default 0.0 in offline mode):
    python tooltree_baseline.py ... --tau-post 0.0

    # Cheaper search (fewer rollouts):
    python tooltree_baseline.py ... --max-rollouts 5

    # Sequential fallback for debugging:
    python tooltree_baseline.py ... --num-workers 1

PARALLELISM:
  • --num-workers N: process N queries concurrently in the outer loop.
    The planner GPU is serialized via a Lock; while one thread holds the
    GPU, the others fire judge HTTP calls — hides judge wall-time behind
    planner work.
  • --judge-concurrency K: fan out the K pre-eval judge calls within an
    expansion. Peak concurrent judge requests = num_workers × K.
  • If your judge server can't handle the concurrent load (timeouts, 500s),
    lower these or bump the judge's --batch-size / --batch-timeout-ms.
"""

import json
import re
import time
import math
import random
import argparse
import requests
import threading
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Constants (shared with other baselines)
# ─────────────────────────────────────────────────────────────────────────────

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

# Planner system prompt for candidate generation
PLANNER_SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query, the available tools, and a partial plan so far, you "
    "generate alternative next steps to extend the plan."
)

# Defaults (matching paper §B.4 hyperparameters where reasonable;
# Rmax reduced from 60 to 10 for offline tool-plan budget parity.
# τ_post defaults to 0.0 because in our offline adaptation rpost reuses rpre's
# value — applying a stricter post-prune threshold (paper's 0.4) on the same
# signal creates a "valley" where candidates with rpre ∈ [τ_pre, τ_post) pass
# pre-prune but are immediately killed post-prune, wasting expansions.)
DEFAULT_RMAX = 10
DEFAULT_LAMBDA = 1.4
DEFAULT_TAU_PRE = 0.3
DEFAULT_TAU_POST = 0.0
DEFAULT_N_CANDIDATES = 2
DEFAULT_MAX_DEPTH = 6
DEFAULT_EARLY_STOP_PATIENCE = 5
DEFAULT_EARLY_STOP_DELTA = 0.01

# Concurrency. Outer (queries) and inner (judge fan-out per expansion).
# The planner GPU serializes via a Lock; judge calls fan out concurrently.
DEFAULT_NUM_WORKERS = 4         # parallel queries in the outer eval loop
DEFAULT_JUDGE_CONCURRENCY = 4   # parallel judge requests per expansion

# Token marker the planner uses to signal "the plan is complete"
COMPLETE_MARKER = "COMPLETE"


# ─────────────────────────────────────────────────────────────────────────────
# Format helpers (copied from existing baselines for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def load_perfect_gt_from_parquet(parquet_path: str) -> dict:
    """Build {query_id: ground_truth_str} for error_type='none', quality>=100.
    Used to give artifact error types a fair structural comparison target."""
    import json as _json
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos = table.column("extra_info").to_pylist()
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
    seen: Dict[str, Any] = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in seen:
            seen[name] = tool_info
    for name, tool_info in seen.items():
        props = tool_info.get("parameters", {}).get("properties", {})
        required = tool_info.get("parameters", {}).get("required", [])
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}{' (required)' if k in required else ''}"
            for k, v in props.items()
        )
        lines.append(f"- {name}({params})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plan parsing
# ─────────────────────────────────────────────────────────────────────────────

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
                    params: Dict[str, str] = {}
                else:
                    continue
            else:
                tool_name = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                params = {}
                if params_str:
                    param_parts: List[str] = []
                    current = ""
                    depth = 0
                    in_str = False
                    str_char: Optional[str] = None
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
                "step_id": step_id,
                "output_variable": output_var,
                "tool_name": tool_name,
                "parameters": params,
            })
        except Exception:
            continue
    return steps


def _extract_step_line(raw: str) -> Optional[str]:
    """From an LLM output, extract the first valid Step line and normalize it
    to canonical `Step N: {{N}} = tool_name(args)` form.

    Accepts (in order of preference):
      1. Strict: `Step N: {{N}} = tool_name(args)` — passes through unchanged
      2. Strict empty:  `Step N: {{N}} = tool_name()`
      3. Named var: `Step N: anyvar = tool_name(args)` → rewrite to {{N}}
      4. No var:    `Step N: tool_name(args)` → inject {{N}} =

    Llama-3-3B and other smaller instruct models intermittently emit forms
    3 and 4 instead of the canonical {{N}} = form. Without this leniency
    those candidates are dropped and the planner appears to fail entirely."""
    # Identifier (Python-ish): start with letter/_, then letters/digits/_
    _IDENT = r"[A-Za-z_][A-Za-z_0-9]*"

    # Strict canonical: Step N: {{N}} = tool(...)
    _STRICT = re.compile(
        rf"^Step (\d+):\s*\{{\{{\d+\}}\}}\s*=\s*({_IDENT})\((.*)\)\s*$"
    )
    _STRICT_EMPTY = re.compile(
        rf"^Step (\d+):\s*\{{\{{\d+\}}\}}\s*=\s*({_IDENT})\(\)\s*$"
    )
    # Lenient: Step N: anyvar = tool(...)
    _NAMED_VAR = re.compile(
        rf"^Step (\d+):\s*{_IDENT}\s*=\s*({_IDENT})\((.*)\)\s*$"
    )
    _NAMED_VAR_EMPTY = re.compile(
        rf"^Step (\d+):\s*{_IDENT}\s*=\s*({_IDENT})\(\)\s*$"
    )
    # Lenient: Step N: tool(...)
    _NO_VAR = re.compile(
        rf"^Step (\d+):\s*({_IDENT})\((.*)\)\s*$"
    )
    _NO_VAR_EMPTY = re.compile(
        rf"^Step (\d+):\s*({_IDENT})\(\)\s*$"
    )

    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("Step "):
            continue

        # Try strict patterns first — pass through unchanged for fidelity
        m = _STRICT.match(line)
        if m:
            return line
        m = _STRICT_EMPTY.match(line)
        if m:
            return line

        # Lenient with-args: rewrite to canonical form
        for pat in (_NAMED_VAR, _NO_VAR):
            m = pat.match(line)
            if m:
                step_id, tool, args = m.group(1), m.group(2), m.group(3)
                return f"Step {step_id}: {{{{{step_id}}}}} = {tool}({args})"

        # Lenient no-args: rewrite
        for pat in (_NAMED_VAR_EMPTY, _NO_VAR_EMPTY):
            m = pat.match(line)
            if m:
                step_id, tool = m.group(1), m.group(2)
                return f"Step {step_id}: {{{{{step_id}}}}} = {tool}()"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Structural evaluation (copied)
# ─────────────────────────────────────────────────────────────────────────────

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
    gt_kw = keywords(gt_name)
    if not gen_kw or not gt_kw:
        return 0.0
    intersection = gen_kw & gt_kw
    union = gen_kw | gt_kw
    return round(len(intersection) / len(union), 3)


def normalize_value(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())


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
    total_functional = 0.0
    total_params_correct = 0
    total_params = 0
    correct_deps = 0
    total_deps = 0
    step_details: List[Dict] = []

    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt = gt_steps[i] if i < len(gt_steps) else None
        detail: Dict[str, Any] = {"step_id": i}

        if gen and gt:
            exact_ok = gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower()
            detail["exact_tool_correct"] = exact_ok
            if exact_ok:
                correct_exact_tools += 1

            func_score = _functional_tool_match(gen["tool_name"], gt["tool_name"])
            detail["functional_tool_score"] = func_score
            total_functional += func_score

            gt_keys = set(gt["parameters"].keys())
            gen_keys = set(gen["parameters"].keys())
            common = gt_keys & gen_keys
            incorrect: List[Dict] = []
            for k in common:
                gt_v = normalize_value(gt["parameters"][k])
                gen_v = normalize_value(gen["parameters"][k])
                if gt_v == gen_v or gt_v in gen_v or gen_v in gt_v:
                    total_params_correct += 1
                else:
                    incorrect.append({
                        "param": k,
                        "generated": gen["parameters"][k],
                        "ground_truth": gt["parameters"][k],
                    })
            total_params += len(gt_keys)
            detail["param_comparison"] = {
                "total_gt_params": len(gt_keys),
                "correct": len(common) - len(incorrect),
                "missing": list(gt_keys - gen_keys),
                "extra": list(gen_keys - gt_keys),
                "incorrect": incorrect,
            }

            gt_refs = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            total_deps += len(gt_refs)
            correct_deps += len(gt_refs & gen_refs)
            detail["dependency_refs_match"] = (gt_refs == gen_refs)
        else:
            detail["exact_tool_correct"] = False
            detail["functional_tool_score"] = 0.0
            detail["param_comparison"] = None
            detail["dependency_refs_match"] = False

        step_details.append(detail)

    n_gt = len(gt_steps)
    exact_tool_acc = correct_exact_tools / n_gt
    functional_acc = total_functional / n_gt
    param_accuracy = total_params_correct / total_params if total_params > 0 else 0.0
    dep_accuracy = correct_deps / total_deps if total_deps > 0 else 1.0

    return {
        "valid": True,
        "gt_uses_nl_tool_names": gt_uses_nl,
        "step_count_match": step_count_match,
        "generated_steps": len(gen_steps),
        "ground_truth_steps": n_gt,
        "exact_tool_accuracy": exact_tool_acc,
        "functional_tool_accuracy": functional_acc,
        "param_accuracy": param_accuracy,
        "dependency_accuracy": dep_accuracy,
        "exact_match": step_count_match and exact_tool_acc == 1.0 and param_accuracy == 1.0,
        "functional_match": step_count_match and functional_acc >= 0.5 and param_accuracy >= 0.5,
        "param_only_match": param_accuracy >= 0.5,
        "step_details": step_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Judge client (copied)
# ─────────────────────────────────────────────────────────────────────────────

def _format_tools_for_judge(tools: Dict[str, Any]) -> str:
    if not tools:
        return ""
    lines = ["Available Tools:"]
    seen: Dict[str, Any] = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in seen:
            seen[name] = tool_info
    for name, tool_info in seen.items():
        props = tool_info.get("parameters", {}).get("properties", {})
        params_str = ", ".join(f"{k}: {v.get('type','any')}" for k, v in props.items())
        lines.append(f"- {name}({params_str})")
    return "\n".join(lines)


def score_plan_via_judge(
    query: str,
    plan_str: str,
    tools: Dict,
    max_tokens: int = 300,
    retries: int = 3,
) -> Dict[str, Any]:
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
            {"role": "user", "content": user_content},
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
                content = content[content.find("```json") + 7: content.rfind("```")].strip()
            elif "```" in content:
                content = content[content.find("```") + 3: content.rfind("```")].strip()
            if not content.endswith("}"):
                last = content.rfind("}")
                if last != -1:
                    content = content[:last + 1]
            annotation = json.loads(content)
            annotation["quality_score"] = max(0, min(100, int(annotation.get("quality_score", 50))))
            annotation["confidence"] = max(0.0, min(1.0, float(annotation.get("confidence", 0.5))))
            annotation["_full_parse"] = True
            return annotation
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt < retries - 1:
                time.sleep(2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            try:
                m = re.search(r'"quality_score"\s*:\s*(\d+)', content)
                if m:
                    return {
                        "quality_score": max(0, min(100, int(m.group(1)))),
                        "success_prediction": "uncertain",
                        "reasoning": "partial parse",
                        "issues": [], "confidence": 0.5, "_full_parse": False,
                    }
            except Exception:
                pass
            break

    return {
        "quality_score": 0, "success_prediction": "no",
        "reasoning": "judge call failed", "issues": [], "confidence": 0.0,
        "_full_parse": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Planner LLM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PlannerModel:
    """Thin wrapper around a HuggingFace causal LM for candidate generation.

    Thread-safe: a single Lock serializes access to the GPU so multiple
    worker threads can call .generate() concurrently from the outer
    parallel-queries loop without crashing on concurrent CUDA access.
    Judge calls happen OUTSIDE this lock, so threads waiting on the planner
    don't block each other's HTTP calls to the judge."""

    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading planner from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._gpu_lock = threading.Lock()
        print(f"  ✓ Planner loaded on {self.device}")

    def generate(
        self,
        user_content: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenize outside the GPU lock (it's CPU work)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        # Only the GPU forward pass needs serialization
        with self._gpu_lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 1e-6),
                    do_sample=temperature > 0,
                    top_p=0.9 if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        # Decode outside the lock too
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_prompt(
    question: str,
    tools: Dict[str, Any],
    partial_plan_steps: List[str],
    n_candidates: int,
    next_step_idx: int,
) -> str:
    tools_str = format_tools(tools)
    if partial_plan_steps:
        partial_str = "\n".join(partial_plan_steps)
    else:
        partial_str = "(empty — this is the first step)"

    return (
        f"Generate {n_candidates} DIFFERENT candidate next steps for the partial plan below. "
        f"Each candidate should use a different tool or take a different approach.\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n\n"
        f"Partial plan so far ({len(partial_plan_steps)} steps):\n"
        f"{partial_str}\n\n"
        f"Generate {n_candidates} alternative next steps. The next step index is "
        f"{next_step_idx} (so use output variable {{{{{next_step_idx}}}}}).\n\n"
        f"Format:\n"
        f"Candidate 1: Step {next_step_idx}: {{{{{next_step_idx}}}}} = tool_name(arg=value, ...)\n"
        f"Candidate 2: Step {next_step_idx}: {{{{{next_step_idx}}}}} = tool_name(arg=value, ...)\n"
        f"...\n\n"
        f"Rules:\n"
        f"  - Use {{{{N}}}} variables for intermediate results.\n"
        f"  - Reference prior outputs (e.g. {{{{0}}}}, {{{{1}}}}) with {{{{N}}}} in parameter values.\n"
        f"  - Each candidate must be a valid Step line.\n"
        f"  - Do NOT repeat a tool that already appears in the partial plan unless "
        f"its arguments are meaningfully different. Prefer the smallest set of "
        f"distinct tool calls that solves the query.\n"
        f"  - If the plan APPEARS COMPLETE (all sub-questions in the query have been "
        f"addressed by the steps above), output only:\n"
        f"      Candidate 1: {COMPLETE_MARKER}\n\n"
        f"Generate the candidates now:"
    )


def _step_body(line: str) -> str:
    """Extract the `tool_name(args)` portion of a Step line for dedup.
    Returns lowercased+whitespace-collapsed string, or '' on parse failure."""
    m = re.search(r"=\s*(.+?)\s*$", line.strip())
    if not m:
        return ""
    body = m.group(1)
    # Collapse whitespace, lowercase tool name + args for loose matching
    return re.sub(r"\s+", " ", body.strip().lower())


def parse_candidates(
    raw: str,
    n_expected: int,
    prior_step_lines: Optional[List[str]] = None,
) -> List[str]:
    """Parse N candidates from the planner output.
    Returns either ["COMPLETE"] or a list of Step lines.
    Skips malformed candidates AND candidates whose tool_name(args) matches a
    prior step in the partial plan."""
    # Check for COMPLETE marker
    if COMPLETE_MARKER in raw.upper():
        # Was COMPLETE explicitly emitted at the start of a candidate?
        for line in raw.split("\n"):
            stripped = line.strip()
            if re.match(rf"Candidate\s*\d+\s*:\s*{COMPLETE_MARKER}\b", stripped, re.IGNORECASE):
                return [COMPLETE_MARKER]

    # Build dedup set from prior trajectory (tool_name(args) bodies)
    prior_bodies = set()
    if prior_step_lines:
        for line in prior_step_lines:
            body = _step_body(line)
            if body:
                prior_bodies.add(body)

    candidates: List[str] = []
    seen_bodies = set()
    # Split on "Candidate N:" markers
    parts = re.split(r"Candidate\s*\d+\s*:\s*", raw)
    # parts[0] is the preamble (may be empty)
    for part in parts[1:]:
        step_line = _extract_step_line(part)
        if step_line is None:
            continue
        body = _step_body(step_line)
        if not body or body in prior_bodies or body in seen_bodies:
            continue
        seen_bodies.add(body)
        candidates.append(step_line)

    # Fallback: if regex split didn't find candidates, try a line-by-line scan
    if not candidates:
        for line in raw.split("\n"):
            step_line = _extract_step_line(line)
            if step_line is None:
                continue
            body = _step_body(step_line)
            if not body or body in prior_bodies or body in seen_bodies:
                continue
            seen_bodies.add(body)
            candidates.append(step_line)

    return candidates[:n_expected]


# ─────────────────────────────────────────────────────────────────────────────
# ToolTree node + agent
# ─────────────────────────────────────────────────────────────────────────────

class ToolTreeNode:
    """A node in the ToolTree search tree representing a partial plan state.

    Mirrors the reference implementation's MCTSNode (src/mcts/node.py):
      - is_leaf() == (no children)
      - is_terminal: set if depth >= max_depth, planner emits COMPLETE, or
        expansion finds no candidates that pass pre-pruning
      - is_expandable: True by default; set False by post-pruning
        (paper §3.2: rpost(s,a) < τ_post marks edge non-expandable)
    """

    __slots__ = (
        "parent", "step_str", "depth",
        "children",
        "rpre_cached",      # rpre(parent_state, action_to_here), set at creation
        "q",                # mean rpost from rollouts through this edge
        "n_visits",         # N(parent_state, action_to_here)
        "is_terminal",
        "is_expandable",    # False after post-pruning (paper's "marked non-expandable")
        "_cached_score",    # cached judge score on this node's full plan
    )

    def __init__(self, parent=None, step_str=None, depth=0, rpre_cached=0.0):
        self.parent = parent
        self.step_str = step_str
        self.depth = depth
        self.children: List["ToolTreeNode"] = []
        self.rpre_cached = rpre_cached
        self.q = 0.0
        self.n_visits = 0
        self.is_terminal = False
        self.is_expandable = True
        self._cached_score: Optional[float] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_partial_plan_steps(self) -> List[str]:
        steps: List[str] = []
        node = self
        while node.parent is not None:
            if node.step_str:
                steps.append(node.step_str)
            node = node.parent
        return list(reversed(steps))

    def get_plan_str(self) -> str:
        return "\n".join(self.get_partial_plan_steps())


class ToolTreeAgent:
    """ToolTree MCTS agent (offline adaptation).

    Each call to .run(question, tools) executes Rmax rollouts of MCTS-with-dual-eval
    and returns the highest-Q trajectory as a plan string.
    """

    def __init__(
        self,
        planner: PlannerModel,
        max_rollouts: int = DEFAULT_RMAX,
        lambda_: float = DEFAULT_LAMBDA,
        tau_pre: float = DEFAULT_TAU_PRE,
        tau_post: float = DEFAULT_TAU_POST,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
        early_stop_delta: float = DEFAULT_EARLY_STOP_DELTA,
        candidate_temperature: float = 0.7,
        judge_max_tokens: int = 200,
        judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    ):
        self.planner = planner
        self.max_rollouts = max_rollouts
        self.lambda_ = lambda_
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.candidate_temperature = candidate_temperature
        self.judge_max_tokens = judge_max_tokens
        self.judge_concurrency = judge_concurrency

    # ── UCT ───────────────────────────────────────────────────────────────
    def _uct(self, parent: ToolTreeNode, child: ToolTreeNode) -> float:
        """Prior-augmented selection score.

        NOTE: This matches the reference codebase's MCTSNode.best_child, which
        DIVERGES from paper Eq. 1. The paper writes
            Q(s,a) + λ · rpre(s,a) · sqrt( ln N(s) / N(s,a) )
        but the released code computes
            Q(s,a) + λ · (rpre or 1e-6) · sqrt(ln(N_parent+1)) / (1 + N(s,a))  + jitter
        i.e. PUCT-style (visit count in denominator linearly, not under sqrt),
        with prior floored at 1e-6 and a tiny random jitter to break ties.
        We match the code, not the equation — that's what produced the
        paper's reported numbers.
        """
        parent_visits = max(parent.n_visits, 1)
        log_parent = math.log(parent_visits + 1)
        prior = child.rpre_cached if child.rpre_cached > 0 else 1e-6
        exploit = child.q
        explore = (
            self.lambda_
            * prior
            * math.sqrt(log_parent)
            / (1 + child.n_visits)
        )
        jitter = random.random() * 1e-9
        return exploit + explore + jitter

    # ── Selection ────────────────────────────────────────────────────────
    def _select(self, root: ToolTreeNode) -> ToolTreeNode:
        """Descend the tree via UCT until reaching a leaf, a terminal node,
        or a node whose children are all non-expandable. Mirrors the
        reference's ToolTreeSearch._select."""
        current = root
        while not current.is_leaf():
            expandable = [c for c in current.children if c.is_expandable]
            if not expandable:
                return current
            current = max(expandable, key=lambda c: self._uct(current, c))
            if current.is_terminal or not current.is_expandable:
                return current
        return current

    # ── Expansion ────────────────────────────────────────────────────────
    def _expand(
        self,
        node: ToolTreeNode,
        question: str,
        tools: Dict[str, Any],
    ) -> List[ToolTreeNode]:
        """Generate candidates, pre-evaluate, pre-prune by τ_pre (threshold ONLY
        — top-K is enforced at candidate generation, matching the reference).
        Returns the list of newly created children (may be empty)."""
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return []

        partial_steps = node.get_partial_plan_steps()
        next_step_idx = len(partial_steps)

        prompt = build_candidate_prompt(
            question=question,
            tools=tools,
            partial_plan_steps=partial_steps,
            n_candidates=self.n_candidates,
            next_step_idx=next_step_idx,
        )
        raw = self.planner.generate(
            user_content=prompt,
            temperature=self.candidate_temperature,
            max_new_tokens=256,   # 3 candidates fit easily in ~150 tokens;
                                   # 256 leaves headroom without burning budget
        )
        candidates = parse_candidates(
            raw,
            self.n_candidates,
            prior_step_lines=partial_steps,
        )

        # Handle "COMPLETE" signal
        if candidates and candidates[0] == COMPLETE_MARKER:
            node.is_terminal = True
            return []

        # If parser found nothing usable, mark terminal so we don't keep retrying
        if not candidates:
            node.is_terminal = True
            return []

        # Pre-evaluate each candidate IN PARALLEL (judge is HTTP, fan out).
        # Sequential here was a major bottleneck — N candidates × ~2s/judge
        # = 2N seconds of wall-time per expansion. With ThreadPoolExecutor
        # they fire concurrently and the judge server's batching handles it.
        if len(candidates) > 1:
            workers = min(len(candidates), self.judge_concurrency)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rpres = list(pool.map(
                    lambda c: self._pre_evaluate(
                        parent_node=node,
                        candidate_step_str=c,
                        question=question,
                        tools=tools,
                    ),
                    candidates,
                ))
        else:
            rpres = [self._pre_evaluate(
                parent_node=node,
                candidate_step_str=candidates[0],
                question=question,
                tools=tools,
            )]

        # Pre-prune by τ_pre threshold (no top-K filter — top-K is at gen time)
        new_children: List[ToolTreeNode] = []
        for cand_str, rpre in zip(candidates, rpres):
            if rpre < self.tau_pre:
                continue
            child = ToolTreeNode(
                parent=node,
                step_str=cand_str,
                depth=node.depth + 1,
                rpre_cached=rpre,
            )
            # Offline adaptation: rpost == rpre for the same partial-plan input.
            # Cache it on the child so _post_evaluate doesn't re-call the judge.
            child._cached_score = rpre
            node.children.append(child)
            new_children.append(child)

        # If all candidates were pre-pruned, the node has no children. Selection
        # will return here again next rollout; if temperature > 0, regenerated
        # candidates may pass. To bound waste, mark terminal here.
        if not new_children:
            node.is_terminal = True

        return new_children

    # ── Pre-evaluation ────────────────────────────────────────────────────
    def _pre_evaluate(
        self,
        parent_node: ToolTreeNode,
        candidate_step_str: str,
        question: str,
        tools: Dict[str, Any],
    ) -> float:
        """rpre(s, a): score the partial plan with the candidate step appended."""
        partial_steps = parent_node.get_partial_plan_steps()
        full_steps = partial_steps + [candidate_step_str]
        plan_str = "\n".join(full_steps)

        if not plan_str.strip():
            return 0.0

        ann = score_plan_via_judge(
            question, plan_str, tools, max_tokens=self.judge_max_tokens
        )
        return ann["quality_score"] / 100.0

    # ── Post-evaluation ──────────────────────────────────────────────────
    def _post_evaluate(
        self,
        node: ToolTreeNode,
        question: str,
        tools: Dict[str, Any],
    ) -> float:
        """rpost(s, a): score the partial plan ending at this node.

        Offline adaptation: if rpre was already computed for this node during
        expansion (same partial plan), reuse the cached score. Otherwise,
        compute fresh."""
        if node._cached_score is not None:
            return node._cached_score
        if node.parent is None:
            return 0.0  # root — no plan yet
        plan_str = node.get_plan_str()
        if not plan_str.strip():
            return 0.0
        ann = score_plan_via_judge(
            question, plan_str, tools, max_tokens=self.judge_max_tokens
        )
        score = ann["quality_score"] / 100.0
        node._cached_score = score
        return score

    # ── Backprop ─────────────────────────────────────────────────────────
    def _backprop(self, node: ToolTreeNode, reward: float) -> None:
        """Walk from `node` up to and INCLUDING root, updating Q and N.
        Matches the reference's ToolTreeSearch._backpropagate (which loops
        while current is not None, so the root is updated too)."""
        cur: Optional[ToolTreeNode] = node
        while cur is not None:
            cur.n_visits += 1
            cur.q += (reward - cur.q) / cur.n_visits
            cur = cur.parent

    # ── One rollout ──────────────────────────────────────────────────────
    def _run_rollout(
        self,
        root: ToolTreeNode,
        question: str,
        tools: Dict[str, Any],
    ) -> float:
        # 1. Selection
        leaf = self._select(root)

        # Force terminal if at max depth (matches reference)
        if leaf.depth >= self.max_depth:
            leaf.is_terminal = True

        # 2. Expansion + target pick. Reference takes new_children[0] (first
        # survivor in iteration order, which is the planner's first candidate
        # that passed pre-pruning).
        if not leaf.is_terminal:
            new_children = self._expand(leaf, question, tools)
            target = new_children[0] if new_children else leaf
        else:
            target = leaf

        # 3. Post-evaluation + 4. Post-pruning + 5. Backprop
        if target is not root and target.step_str is not None:
            rpost = self._post_evaluate(target, question, tools)
            # Post-prune (paper §3.2: rpost < τ_post → non-expandable)
            if rpost < self.tau_post:
                target.is_expandable = False
            self._backprop(target, rpost)
            return rpost
        else:
            # Root or empty expansion — propagate a zero reward up.
            self._backprop(target, 0.0)
            return 0.0

    # ── Best path extraction ────────────────────────────────────────────
    def _extract_best_plan(self, root: ToolTreeNode) -> Tuple[str, ToolTreeNode]:
        """Greedy max-q descent from root, matching the reference's
        ToolTreeSearch._get_best_trajectory. Non-expandable children remain
        candidates (their q_value still encodes a signal)."""
        node = root
        while node.children:
            best = max(
                node.children,
                key=lambda c: (c.q, c.n_visits, c.rpre_cached),
            )
            node = best
            if best.is_terminal or not best.children:
                break
        return node.get_plan_str(), node

    # ── Public entry ────────────────────────────────────────────────────
    def run(
        self,
        question: str,
        tools: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Rmax MCTS rollouts and return the best trajectory.

        Returns dict with: plan_str, n_rollouts_used, max_q, tree_size,
        best_leaf_depth, rollout_history (list of rpost per rollout).
        """
        root = ToolTreeNode(parent=None, step_str=None, depth=0, rpre_cached=0.0)

        best_q_history: List[float] = []
        rollout_history: List[float] = []

        for r in range(self.max_rollouts):
            rpost = self._run_rollout(root, question, tools)
            rollout_history.append(rpost)

            # Track best Q at the root's children (mirrors reference)
            best_q_now = max((c.q for c in root.children), default=0.0)
            best_q_history.append(best_q_now)

            # Early stopping: best Q over recent window has not improved by > delta
            # over the patience-length window's baseline. Matches reference exactly.
            if len(best_q_history) > self.early_stop_patience:
                window = best_q_history[-(self.early_stop_patience + 1):]
                baseline = window[0]
                best_recent = max(window[1:])
                if (best_recent - baseline) < self.early_stop_delta:
                    break

        plan_str, best_leaf = self._extract_best_plan(root)
        return {
            "plan_str": plan_str,
            "n_rollouts_used": len(rollout_history),
            "max_q": max((c.q for c in root.children), default=0.0),
            "tree_size": sum(1 for _ in self._all_nodes(root)),
            "best_leaf_depth": best_leaf.depth,
            "rollout_history": rollout_history,
        }

    @staticmethod
    def _all_nodes(root: ToolTreeNode):
        """Generator over all nodes in the tree."""
        stack = [root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (copied)
# ─────────────────────────────────────────────────────────────────────────────

def load_test_parquet(parquet_path: str, perfect_only: bool = False) -> List[Dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    examples: List[Dict] = []

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

        error_type = str(extra_info.get("error_type", "none"))
        quality_score = int(extra_info.get("quality_score", 0))

        if perfect_only and not (error_type == "none" and quality_score >= 100):
            continue

        examples.append({
            "question": data_json.get("question", ""),
            "tools": data_json.get("tools", {}),
            "ground_truth": reward_model.get("ground_truth", ""),
            "error_type": error_type,
            "quality_score": quality_score,
            "query_id": extra_info.get("query_id", -1),
        })

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tool_tree(
    agent: ToolTreeAgent,
    examples: List[Dict],
    perfect_gt_by_qid: Dict[int, str],
    judge_max_tokens_final: int = 300,
    return_tree_meta: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> List[Dict]:
    """Evaluate ToolTree over a list of examples in PARALLEL.

    With num_workers>1, multiple queries run concurrently. The planner GPU
    is serialized via PlannerModel._gpu_lock, but judge HTTP calls fan out,
    hiding most of the judge wall-time behind planner work. Empirically,
    Qwen-3B (~7s/planner) sees ~3-4x speedup with num_workers=4; Llama-3B
    sees less because its planner calls are already fast."""

    def process_one(idx_ex: Tuple[int, Dict]) -> Tuple[int, Dict]:
        idx, ex = idx_ex
        question = ex["question"]
        tools = ex["tools"]
        ground_truth = ex["ground_truth"]
        gt_steps = parse_plan_steps(ground_truth)

        try:
            tree_result = agent.run(question, tools)
            plan_str = tree_result["plan_str"]

            # Short-circuit empty plans. The trained judge has been observed to
            # return quality_score=100 for "(empty plan)" inputs (presumably
            # because it can't find any errors), which silently inflates
            # reported success rates. An empty plan is a hard failure.
            if not plan_str.strip():
                judge_ann = {
                    "quality_score": 0,
                    "success_prediction": "no",
                    "reasoning": "no plan generated (planner failed to produce parseable candidates)",
                    "issues": [{"type": "empty_plan", "severity": "critical"}],
                    "confidence": 1.0,
                    "_full_parse": True,
                }
            else:
                judge_ann = score_plan_via_judge(
                    question, plan_str, tools,
                    max_tokens=judge_max_tokens_final,
                )

            gen_steps = parse_plan_steps(plan_str)
            if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                    and ex["query_id"] in perfect_gt_by_qid):
                _struct_gt_steps = parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
            else:
                _struct_gt_steps = gt_steps
            struct_eval = evaluate_plan_vs_gt(gen_steps, _struct_gt_steps, tools=tools)

            judge_success = judge_ann["quality_score"] >= 80
            ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)
            error_type_handled = (
                judge_success
                if ref_is_perfect
                else judge_ann["quality_score"] >= ex["quality_score"]
            )

            result: Dict[str, Any] = {
                "query_id": ex["query_id"],
                "question": question,
                "error_type": ex["error_type"],
                "ref_quality_score": ex["quality_score"],
                "ref_is_perfect": ref_is_perfect,

                "ground_truth": ground_truth,
                "generated_plan": plan_str,

                # ToolTree metadata
                "n_rollouts_used": tree_result["n_rollouts_used"],
                "max_q":           tree_result["max_q"],
                "tree_size":       tree_result["tree_size"],
                "best_leaf_depth": tree_result["best_leaf_depth"],

                # Judge scores
                "judge_success": judge_success,
                "judge_score": judge_ann["quality_score"],
                "judge_success_pred": judge_ann["success_prediction"],
                "judge_confidence": judge_ann["confidence"],
                "judge_full_parse": judge_ann.get("_full_parse", False),

                # Structural metrics
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

                # Reference agreement
                "error_type_handled": error_type_handled,
                "judge_agrees_with_ref": (ref_is_perfect == judge_success),

                "method": "ToolTree",
            }
            if return_tree_meta:
                result["rollout_history"] = tree_result["rollout_history"]
            return idx, result
        except Exception as e:
            # Don't crash the whole run on one bad query
            return idx, {
                "query_id": ex.get("query_id", -1),
                "question": question,
                "error_type": ex.get("error_type", "none"),
                "ref_quality_score": ex.get("quality_score", 0),
                "ref_is_perfect": False,
                "ground_truth": ground_truth,
                "generated_plan": "",
                "n_rollouts_used": 0, "max_q": 0.0,
                "tree_size": 0, "best_leaf_depth": 0,
                "judge_success": False, "judge_score": 0,
                "judge_success_pred": "no", "judge_confidence": 0.0,
                "judge_full_parse": False,
                "gt_uses_nl_tool_names": False,
                "exact_match": False, "functional_match": False,
                "param_only_match": False, "step_count_match": False,
                "generated_n_steps": 0, "gt_n_steps": len(gt_steps),
                "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
                "param_accuracy": 0.0, "dependency_accuracy": 0.0,
                "error_type_handled": False, "judge_agrees_with_ref": False,
                "method": "ToolTree",
                "_error": f"{type(e).__name__}: {e}",
            }

    results: List[Optional[Dict]] = [None] * len(examples)

    if num_workers <= 1:
        # Sequential fallback (useful for debugging)
        for idx, ex in enumerate(tqdm(examples, desc="ToolTree (sequential)")):
            _, result = process_one((idx, ex))
            results[idx] = result
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(process_one, (i, ex)): i
                       for i, ex in enumerate(examples)}
            with tqdm(total=len(examples),
                      desc=f"ToolTree (parallel, {num_workers} workers)") as pbar:
                for fut in as_completed(futures):
                    idx, result = fut.result()
                    results[idx] = result
                    pbar.update(1)

    empty_plan_count = sum(1 for r in results
                           if r is not None and r["generated_n_steps"] == 0)
    error_count = sum(1 for r in results if r is not None and "_error" in r)

    if empty_plan_count > 0:
        print(f"\n  ⚠  {empty_plan_count}/{len(examples)} examples produced no parseable plan steps.")
    if error_count > 0:
        print(f"  ⚠  {error_count}/{len(examples)} examples failed with an exception "
              f"(see _error in their result entry).")

    return [r for r in results if r is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Statistics (matching other baselines)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(results: List[Dict], label: str) -> Dict:
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}

    scores = [r["judge_score"] for r in results]
    func_tools = [r["functional_tool_accuracy"] for r in results]
    param_accs = [r["param_accuracy"] for r in results]
    dep_accs = [r["dependency_accuracy"] for r in results]

    judge_success_rate = float(np.mean([r["judge_success"] for r in results]))
    error_handled_rate = float(np.mean([r["error_type_handled"] for r in results]))
    exact_match_rate = float(np.mean([r["exact_match"] for r in results]))
    functional_match_rate = float(np.mean([r["functional_match"] for r in results]))
    param_only_match_rate = float(np.mean([r["param_only_match"] for r in results]))
    step_match_rate = float(np.mean([r["step_count_match"] for r in results]))
    full_parse_rate = float(np.mean([r.get("judge_full_parse", False) for r in results]))
    empty_plan_rate = float(np.mean([r["generated_n_steps"] == 0 for r in results]))
    mean_rollouts = float(np.mean([r["n_rollouts_used"] for r in results]))
    mean_tree_size = float(np.mean([r["tree_size"] for r in results]))
    mean_depth = float(np.mean([r["best_leaf_depth"] for r in results]))

    error_types = sorted(set(r["error_type"] for r in results))
    per_error: Dict[str, Any] = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n": len(sub),
            "judge_success_rate": float(np.mean([r["judge_success"] for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"] for r in sub])),
            "mean_judge_score": float(np.mean([r["judge_score"] for r in sub])),
            "functional_tool_acc": float(np.mean([r["functional_tool_accuracy"] for r in sub])),
            "mean_param_accuracy": float(np.mean([r["param_accuracy"] for r in sub])),
            "exact_match_rate": float(np.mean([r["exact_match"] for r in sub])),
            "functional_match_rate": float(np.mean([r["functional_match"] for r in sub])),
            "param_only_match_rate": float(np.mean([r["param_only_match"] for r in sub])),
            "step_count_match_rate": float(np.mean([r["step_count_match"] for r in sub])),
        }

    success_dist: Dict[str, Any] = {}
    for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100 * c / n, 1)}

    stats = {
        "label": label,
        "method": "ToolTree (offline)",
        "n_examples": n,
        "gt_uses_nl_tools": bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(full_parse_rate, 3),
        "empty_plan_rate": round(empty_plan_rate, 3),

        "tree_stats": {
            "mean_rollouts_used": round(mean_rollouts, 2),
            "mean_tree_size": round(mean_tree_size, 2),
            "mean_best_leaf_depth": round(mean_depth, 2),
        },

        "accuracy": {
            "judge_success_rate": round(judge_success_rate, 3),
            "error_handled_rate": round(error_handled_rate, 3),
        },
        "judge_scores": {
            "mean": round(float(np.mean(scores)), 2),
            "median": round(float(np.median(scores)), 2),
            "std": round(float(np.std(scores)), 2),
            "pct_gte_80": round(100 * sum(s >= 80 for s in scores) / n, 1),
            "pct_eq_100": round(100 * sum(s == 100 for s in scores) / n, 1),
        },
        "structural": {
            "exact_match_rate": round(exact_match_rate, 3),
            "functional_match_rate": round(functional_match_rate, 3),
            "param_only_match_rate": round(param_only_match_rate, 3),
            "step_count_match_rate": round(step_match_rate, 3),
            "mean_functional_tool_acc": round(float(np.mean(func_tools)), 3),
            "mean_param_accuracy": round(float(np.mean(param_accs)), 3),
            "mean_dependency_accuracy": round(float(np.mean(dep_accs)), 3),
        },
        "success_prediction_dist": success_dist,
        "per_error_type": per_error,
    }

    W = 70
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  Method : ToolTree (offline, Yang et al. ICLR 2026)")
    print(f"  N      : {n}")
    print(f"  Tree   : avg {mean_rollouts:.1f} rollouts, {mean_tree_size:.1f} nodes, "
          f"best-leaf depth {mean_depth:.1f}")
    if stats["gt_uses_nl_tools"]:
        print(f"  ⚠  GT uses NL tool names — exact_tool_accuracy ~0")
    if empty_plan_rate > 0.05:
        print(f"  ⚠  Empty plan rate: {100*empty_plan_rate:.1f}%")
    if full_parse_rate < 0.9:
        print(f"  ⚠  Judge full-parse rate: {100*full_parse_rate:.0f}%")

    print(f"\n  ── Primary Accuracy ──────────────────────────────────────────")
    print(f"  Judge success (score≥80) : {100*judge_success_rate:.1f}%")
    print(f"  Error type handled       : {100*error_handled_rate:.1f}%")

    print(f"\n  ── Judge Scores ──────────────────────────────────────────────")
    print(f"  Mean / Median / Std : {stats['judge_scores']['mean']:.1f} / "
          f"{stats['judge_scores']['median']:.1f} / {stats['judge_scores']['std']:.1f}")
    print(f"  ≥80: {stats['judge_scores']['pct_gte_80']:.1f}%  |  "
          f"=100: {stats['judge_scores']['pct_eq_100']:.1f}%")

    print(f"\n  ── Structural Metrics ─────────────────────────────────────────")
    print(f"  Exact match: {100*exact_match_rate:.1f}%  "
          f"Functional match: {100*functional_match_rate:.1f}%  "
          f"Param-only: {100*param_only_match_rate:.1f}%")
    print(f"  Step count match: {100*step_match_rate:.1f}%  "
          f"Func tool acc: {np.mean(func_tools):.3f}  "
          f"Param acc: {np.mean(param_accs):.3f}")

    if len(error_types) > 1:
        print(f"\n  ── Per Error-Type ─────────────────────────────────────────────")
        for et, d in per_error.items():
            print(f"  {et:28s}  N={d['n']:>4}  "
                  f"Success={100*d['judge_success_rate']:.1f}%  "
                  f"Judge={d['mean_judge_score']:.1f}  "
                  f"FuncMatch={100*d['functional_match_rate']:.1f}%")
    print()
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ToolTree baseline (offline) for ToolHop tool planning"
    )
    parser.add_argument("--planner-model", required=True,
                        help="HF model path for the planner (generates candidate next steps)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--perfect-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    # MCTS hyperparameters
    parser.add_argument("--max-rollouts", type=int, default=DEFAULT_RMAX,
                        help="Max MCTS rollouts per query (paper §B.4 default: 60; "
                             f"reduced to {DEFAULT_RMAX} for offline budget parity)")
    parser.add_argument("--lambda-uct", type=float, default=DEFAULT_LAMBDA,
                        help="UCT exploration constant λ (paper default: 1.4)")
    parser.add_argument("--tau-pre", type=float, default=DEFAULT_TAU_PRE,
                        help="Pre-pruning threshold τ_pre (paper default: 0.3)")
    parser.add_argument("--tau-post", type=float, default=DEFAULT_TAU_POST,
                        help="Post-pruning threshold τ_post (paper default: 0.4)")
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES,
                        help="Top-K candidates to generate per expansion. The "
                             "reference enforces K here (at generation time), not "
                             "via a post-pre-prune filter.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="Max plan depth")
    parser.add_argument("--early-stop-patience", type=int,
                        default=DEFAULT_EARLY_STOP_PATIENCE,
                        help="Stop if best Q doesn't improve over N rollouts")
    parser.add_argument("--early-stop-delta", type=float,
                        default=DEFAULT_EARLY_STOP_DELTA,
                        help="Q-improvement threshold for early stopping")
    parser.add_argument("--candidate-temperature", type=float, default=0.7,
                        help="Sampling temperature for candidate generation")

    # Concurrency
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Parallel queries in the outer eval loop. Planner GPU "
                             "serializes via a lock; judge HTTP calls fan out. "
                             "Default 4 hides judge latency behind planner work.")
    parser.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY,
                        help="Parallel judge requests fired per expansion "
                             "(K pre-evals at once). Default 4.")

    # Judge
    parser.add_argument("--judge-max-tokens", type=int, default=200,
                        help="Max tokens for in-tree judge calls (rpre/rpost)")
    parser.add_argument("--judge-max-tokens-final", type=int, default=300,
                        help="Max tokens for the final reported judge score")
    parser.add_argument("--judge_url", default="http://localhost:8001/v1/chat/completions")

    # Output
    parser.add_argument("--output", default="tooltree_results.json")
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--return-tree-meta", action="store_true",
                        help="Save rollout_history per example (debug)")

    args = parser.parse_args()

    if not args.perfect_only and not args.full:
        parser.error("Specify at least one of --perfect-only or --full")

    stats_output = args.stats_output or args.output.replace(".json", ".stats.json")
    global JUDGE_SERVER_URL
    JUDGE_SERVER_URL = args.judge_url

    # Health check judge
    try:
        r = requests.get(
            JUDGE_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5
        )
        print(f"✅ Judge server healthy: {r.json()}")
    except Exception as e:
        print(f"❌ Judge server not reachable: {e}")
        return

    # Load planner
    print("\nLoading planner...")
    planner = PlannerModel(args.planner_model, device=args.device)

    # Build agent
    agent = ToolTreeAgent(
        planner=planner,
        max_rollouts=args.max_rollouts,
        lambda_=args.lambda_uct,
        tau_pre=args.tau_pre,
        tau_post=args.tau_post,
        n_candidates=args.n_candidates,
        max_depth=args.max_depth,
        early_stop_patience=args.early_stop_patience,
        early_stop_delta=args.early_stop_delta,
        candidate_temperature=args.candidate_temperature,
        judge_max_tokens=args.judge_max_tokens,
        judge_concurrency=args.judge_concurrency,
    )

    # Perfect GT lookup
    perfect_gt_by_qid = load_perfect_gt_from_parquet(args.test_parquet)

    all_output: Dict[str, Any] = {"config": vars(args), "runs": {}}
    all_stats: Dict[str, Any] = {"config": vars(args), "runs": {}}

    if args.perfect_only:
        print("\nLoading perfect-only test examples...")
        examples = load_test_parquet(args.test_parquet, perfect_only=True)
        if args.limit:
            examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_tool_tree(
            agent, examples,
            perfect_gt_by_qid=perfect_gt_by_qid,
            judge_max_tokens_final=args.judge_max_tokens_final,
            return_tree_meta=args.return_tree_meta,
            num_workers=args.num_workers,
        )
        stats = compute_stats(results, "PERFECT-ONLY  ToolTree")
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"] = stats

    if args.full:
        print("\nLoading full test set...")
        examples = load_test_parquet(args.test_parquet, perfect_only=False)
        if args.limit:
            examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_tool_tree(
            agent, examples,
            perfect_gt_by_qid=perfect_gt_by_qid,
            judge_max_tokens_final=args.judge_max_tokens_final,
            return_tree_meta=args.return_tree_meta,
            num_workers=args.num_workers,
        )
        stats = compute_stats(results, "FULL TEST SET  ToolTree")
        all_output["runs"]["full"] = results
        all_stats["runs"]["full"] = stats

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"Results saved → {args.output}")

    with open(stats_output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats   saved → {stats_output}")


if __name__ == "__main__":
    main()