#!/usr/bin/env python3
"""
GPT (closed-source frontier) baseline for FORTE comparison.

Evaluates an OpenAI model with the EXACT SAME prompt format, judge server,
and structural eval as the FORTE-trained planners. Produces apples-to-apples
reference numbers for the main results table.

Single-shot greedy generation (no Best-of-N). The asymmetry vs FORTE-BoN5
is intentional and should be disclosed in the paper: FORTE uses inference-
time selection over 5 candidates; the closed-source reference is single-
shot to keep API spend reasonable. Frame the comparison as
"FORTE-BoN5 matches/beats frontier-greedy."

Output JSON schema mirrors best_of_n_selection_*.py with n_candidates=1
so the same downstream stats / comparison scripts work without changes.

Usage:
    # Pilot, 100 ToolHop examples, GPT-5.1-Nano:
    python evaluate_gpt_baseline.py \
        --api-key  \
        --model gpt-5.5 \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --full --limit 100 \
        --max-output-tokens 2048 \
        --output results/gpt-5.5/results_toolhop.json \
        --judge_url http://localhost:8001/v1/chat/completions

    # NESTFUL pilot:
    python evaluate_gpt_baseline.py \
        --api-key  \
        --model gpt-5.5 \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_nestful_full/test.parquet \
        --full --limit 100 \
        --max-output-tokens 2048 \
        --output results/gpt-5.5/results_nestful.json \
        --judge_url http://localhost:8002/v1/chat/completions

    # If pilot looks good, drop --limit for the full 1000-example run.
"""
 
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
 
import numpy as np
import requests
import requests.adapters
from tqdm import tqdm
from openai import OpenAI
 
from dataset_utils import resolve_dataset, dataset_label
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PROMPT FORMAT — must exactly match FORTE planner inference (best_of_n_*)
# ══════════════════════════════════════════════════════════════════════════════
 
SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)
 
 
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
        f"3. Reference previous step outputs using {{{{N}}}} - never substitute a "
        f"hardcoded value for an output that comes from a prior step\n"
        f"4. Use the exact parameter names shown in the tool signatures above\n"
        f"5. Provide all required parameters\n\n"
        f"Generate only the steps the query requires - no redundant steps, "
        f"no missing steps.\n\n"
        f"Format each step as: Step N: {{{{N}}}} = tool_name(param1=value1, "
        f"param2=value2, ...)"
    )
 
 
ARTIFACT_ERROR_TYPES = {
    "circular_dependency", "forward_reference", "incomplete_plan",
    "inefficient_order", "missing_dependency", "parameter_typo",
    "type_mismatch", "unnecessary_steps", "wrong_tool",
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# JUDGE CLIENT — copied verbatim from best_of_n_selection_*.py
# ══════════════════════════════════════════════════════════════════════════════
 
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
 
_http_session: Optional[requests.Session] = None
 
 
def _get_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=8, pool_maxsize=8, max_retries=0,
        )
        _http_session.mount("http://", adapter)
        _http_session.headers.update({"Connection": "keep-alive"})
    return _http_session
 
 
def _format_tools_for_judge(tools: Dict[str, Any]) -> str:
    if not tools:
        return ""
    lines = ["Available Tools:"]
    unique = {}
    for sub_q, info in tools.items():
        name = info.get("name", sub_q)
        if name not in unique:
            unique[name] = info
    for name, info in unique.items():
        props = info.get("parameters", {}).get("properties", {})
        ps = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {name}({ps})")
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
                content = content[content.find("```json") + 7: content.rfind("```")].strip()
            elif "```" in content:
                content = content[content.find("```") + 3: content.rfind("```")].strip()
            if not content.endswith("}"):
                last = content.rfind("}")
                if last != -1:
                    content = content[:last + 1]
            ann = json.loads(content)
            ann["quality_score"] = max(0, min(100, int(ann.get("quality_score", 50))))
            ann["confidence"] = max(0.0, min(1.0, float(ann.get("confidence", 0.5))))
            ann["_full_parse"] = True
            return ann
        except (requests.ConnectionError, requests.Timeout):
            global _http_session
            _http_session = None
            if attempt < retries - 1:
                time.sleep(2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            m = re.search(r'"quality_score"\s*:\s*(\d+)', content)
            if m:
                return {
                    "quality_score": max(0, min(100, int(m.group(1)))),
                    "success_prediction": "uncertain",
                    "reasoning": "partial parse",
                    "issues": [], "confidence": 0.5,
                    "_full_parse": False,
                }
            break
    return {
        "quality_score": 0, "success_prediction": "no",
        "reasoning": "judge call failed", "issues": [],
        "confidence": 0.0, "_full_parse": False,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PARSING + STRUCTURAL EVAL — copied verbatim
# ══════════════════════════════════════════════════════════════════════════════
 
def _is_nl_tool_name(name: str) -> bool:
    return len(name.split()) > 4 or name.endswith("?")
 
 
def _functional_tool_match(gen_name: str, gt_name: str) -> float:
    STOP = {"what", "is", "the", "of", "in", "a", "an", "and", "or", "to",
            "how", "many", "who", "which", "are", "was", "were", "be", "been",
            "at", "on", "for", "with", "that", "this", "it", "its", "from"}
    def kw(s: str):
        return {w for w in re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
                if w not in STOP and len(w) > 2}
    g, r = kw(gen_name), kw(gt_name)
    if not g or not r:
        return 0.0
    return round(len(g & r) / len(g | r), 3)
 
 
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
                    param_parts, current, depth, in_str, str_char = [], "", 0, False, None
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
            steps.append({"step_id": step_id, "output_variable": output_var,
                          "tool_name": tool_name, "parameters": params})
        except Exception:
            continue
    return steps
 
 
def normalize_value(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())
 
 
def _remap_gt_tool_name(nl_name: str, tools: Dict[str, Any]) -> str:
    if nl_name in tools:
        api_name = tools[nl_name].get("name")
        if api_name:
            return api_name
    for key, info in tools.items():
        if nl_name in key or key in nl_name:
            api_name = info.get("name")
            if api_name:
                return api_name
    return nl_name
 
 
def evaluate_plan_vs_gt(gen_steps, gt_steps, tools=None):
    empty = {"valid": False, "error": "",
             "step_count_match": False,
             "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
             "param_accuracy": 0.0, "dependency_accuracy": 0.0,
             "exact_match": False, "functional_match": False,
             "param_only_match": False, "gt_uses_nl_tool_names": False}
    if not gen_steps:
        return {**empty, "error": "no steps generated"}
    if not gt_steps:
        return {**empty, "error": "no ground truth steps"}
    gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)
    gen_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gen_steps)
    if gt_uses_nl and tools and not gen_uses_nl:
        gt_steps = [{**s, "tool_name": _remap_gt_tool_name(s["tool_name"], tools)}
                    for s in gt_steps]
        gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)
    step_count_match = len(gen_steps) == len(gt_steps)
    ce, tf, tpc, tp, cd, td = 0, 0.0, 0, 0, 0, 0
    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt = gt_steps[i] if i < len(gt_steps) else None
        if gen and gt:
            if gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower():
                ce += 1
            tf += _functional_tool_match(gen["tool_name"], gt["tool_name"])
            gt_keys = set(gt["parameters"].keys())
            gen_keys = set(gen["parameters"].keys())
            for k in (gt_keys & gen_keys):
                gv = normalize_value(gt["parameters"][k])
                dv = normalize_value(gen["parameters"][k])
                if gv == dv or gv in dv or dv in gv:
                    tpc += 1
            tp += len(gt_keys)
            gt_refs = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            td += len(gt_refs)
            cd += len(gt_refs & gen_refs)
    n_gt = len(gt_steps)
    ea = ce / n_gt
    fa = tf / n_gt
    pa = tpc / tp if tp > 0 else 0.0
    da = cd / td if td > 0 else 1.0
    return {
        "valid": True, "gt_uses_nl_tool_names": gt_uses_nl,
        "step_count_match": step_count_match,
        "generated_steps": len(gen_steps), "ground_truth_steps": n_gt,
        "exact_tool_accuracy": ea, "functional_tool_accuracy": fa,
        "param_accuracy": pa, "dependency_accuracy": da,
        "exact_match": step_count_match and ea == 1.0 and pa == 1.0,
        "functional_match": step_count_match and fa >= 0.5 and pa >= 0.5,
        "param_only_match": pa >= 0.5,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — copied verbatim
# ══════════════════════════════════════════════════════════════════════════════
 
def load_test_parquet(parquet_path: str, perfect_only: bool = False):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    examples = []
    for i in range(len(extra_infos)):
        ei = extra_infos[i]
        if isinstance(ei, str):
            ei = json.loads(ei)
        if not isinstance(ei, dict):
            ei = {}
        rm = reward_models[i]
        if isinstance(rm, str):
            rm = json.loads(rm)
        if not isinstance(rm, dict):
            rm = {}
        dj = ei.get("data_json", "{}")
        if isinstance(dj, str):
            dj = json.loads(dj)
        if not isinstance(dj, dict):
            dj = {}
        et = str(ei.get("error_type", "none"))
        qs = int(ei.get("quality_score", 0))
        if perfect_only and not (et == "none" and qs >= 100):
            continue
        examples.append({
            "question": dj.get("question", ""),
            "tools": dj.get("tools", {}),
            "ground_truth": rm.get("ground_truth", ""),
            "error_type": et, "quality_score": qs,
            "query_id": ei.get("query_id", -1),
        })
    perfect_gt = {
        ex["query_id"]: ex["ground_truth"]
        for ex in examples
        if ex["error_type"] == "none" and ex["quality_score"] >= 100
    }
    return examples, perfect_gt
 
 
# ══════════════════════════════════════════════════════════════════════════════
# GPT CLIENT
# ══════════════════════════════════════════════════════════════════════════════
 
def _clean_response(text: str) -> str:
    """Strip markdown code fences. Step-line filtering is handled by the parser."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
 
 
def call_gpt(client, system: str, user: str, model: str,
             temperature: float, max_tokens: int,
             max_retries: int = 4) -> tuple:
    """
    OpenAI Responses API call with automatic fallback for unsupported params.
 
    GPT-5 family models reject `temperature` (they are deterministic by design)
    and benefit from `reasoning={"effort": "minimal"}` to keep hidden reasoning
    from eating the output-token budget. Both params are tried, with automatic
    fall-back on the model-specific "unsupported parameter" error.
 
    Returns (text, error_str).
    """
    prompt = f"{system}\n\n{user}"
 
    # Param-inclusion flags. We attempt the call with them set, and drop on
    # the corresponding 400 / unsupported-parameter error.
    use_temperature = True
    use_reasoning_minimal = "gpt-5" in model.lower()  # only GPT-5 supports this
 
    last_err = ""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "input": prompt,
                "max_output_tokens": max_tokens,
                "text": {"format": {"type": "text"}},
            }
            if use_temperature:
                kwargs["temperature"] = temperature
            if use_reasoning_minimal:
                kwargs["reasoning"] = {"effort": "minimal"}
 
            resp = client.responses.create(**kwargs)
            return resp.output_text.strip(), ""
 
        except Exception as e:
            last_err = str(e)
            low = last_err.lower()
 
            # Auto-drop unsupported params and retry IMMEDIATELY (no backoff,
            # no failed-attempt charge).
            dropped = False
            if use_temperature and "temperature" in low and \
                    ("unsupported" in low or "not supported" in low):
                use_temperature = False
                dropped = True
            if use_reasoning_minimal and "reasoning" in low and \
                    ("unsupported" in low or "not supported" in low):
                use_reasoning_minimal = False
                dropped = True
            if dropped:
                continue
 
            # Normal retry logic for transient errors.
            if "rate" in low or "429" in low:
                time.sleep(min(2 ** attempt, 30))
            elif "timeout" in low:
                time.sleep(min(2 ** attempt, 10))
            elif attempt < max_retries - 1:
                time.sleep(1.0)
            else:
                return "", last_err
    return "", f"Max retries exceeded: {last_err}"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# EVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════
 
def evaluate_gpt(client, examples, perfect_gt_by_qid, dataset, args):
    results = []
    empty_count = 0
    err_count = 0
    pbar = tqdm(examples, desc=f"GPT eval ({args.model})")
    for ex in pbar:
        question = ex["question"]
        tools = ex["tools"]
        gt = ex["ground_truth"]
        gt_steps = parse_plan_steps(gt)
 
        raw, gpt_err = call_gpt(
            client, SYSTEM_PROMPT, build_user_message(question, tools),
            model=args.model, temperature=args.temperature,
            max_tokens=args.max_output_tokens,
        )
        if gpt_err:
            err_count += 1
        generated_plan = _clean_response(raw) if raw else ""
        if not generated_plan:
            empty_count += 1
 
        judge_ann = score_plan_via_judge(
            question, generated_plan or "(empty plan)", tools,
            max_tokens=args.judge_max_tokens,
        )
 
        if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                and ex["query_id"] in perfect_gt_by_qid):
            struct_gt = parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        else:
            struct_gt = gt_steps
        gen_steps = parse_plan_steps(generated_plan)
        struct = evaluate_plan_vs_gt(gen_steps, struct_gt, tools=tools)
 
        judge_success = judge_ann["quality_score"] >= 80
        ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)
        if ref_is_perfect:
            error_type_handled = judge_success
        else:
            error_type_handled = judge_ann["quality_score"] >= ex["quality_score"]
 
        results.append({
            "dataset": dataset,
            "query_id": ex["query_id"],
            "question": question,
            "error_type": ex["error_type"],
            "ref_quality_score": ex["quality_score"],
            "ref_is_perfect": ref_is_perfect,
            "ground_truth": gt,
            "generated_plan": generated_plan,
            "raw_response": raw,
            "gpt_error": gpt_err,
            "judge_success": judge_success,
            "judge_score": judge_ann["quality_score"],
            "judge_success_pred": judge_ann["success_prediction"],
            "judge_confidence": judge_ann["confidence"],
            "judge_full_parse": judge_ann.get("_full_parse", False),
            "gt_uses_nl_tool_names": struct["gt_uses_nl_tool_names"],
            "exact_match": struct["exact_match"],
            "functional_match": struct["functional_match"],
            "param_only_match": struct["param_only_match"],
            "step_count_match": struct["step_count_match"],
            "generated_n_steps": struct.get("generated_steps", 0),
            "gt_n_steps": struct.get("ground_truth_steps", len(gt_steps)),
            "exact_tool_accuracy": struct["exact_tool_accuracy"],
            "functional_tool_accuracy": struct["functional_tool_accuracy"],
            "param_accuracy": struct["param_accuracy"],
            "dependency_accuracy": struct["dependency_accuracy"],
            "error_type_handled": error_type_handled,
            "judge_agrees_with_ref": (ref_is_perfect == judge_success),
            "method": f"GPT-{args.model}",
        })
 
        if len(results) >= 5:
            recent = results[-10:]
            pbar.set_postfix({
                "JSR": f"{100*np.mean([r['judge_success'] for r in recent]):.0f}",
                "score": f"{np.mean([r['judge_score'] for r in recent]):.0f}",
                "errs": err_count,
            })
        time.sleep(args.sleep_time)
 
    if err_count:
        print(f"\n  WARN  GPT API errors: {err_count}/{len(results)}")
    if empty_count:
        print(f"  WARN  Empty responses: {empty_count}/{len(results)}")
    return results
 
 
# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════
 
def compute_stats(results, label, model_name):
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}
    scores = [r["judge_score"] for r in results]
    func_tools = [r["functional_tool_accuracy"] for r in results]
    param_accs = [r["param_accuracy"] for r in results]
    dep_accs = [r["dependency_accuracy"] for r in results]
 
    judge_sr = float(np.mean([r["judge_success"] for r in results]))
    err_hr = float(np.mean([r["error_type_handled"] for r in results]))
    exact_mr = float(np.mean([r["exact_match"] for r in results]))
    func_mr = float(np.mean([r["functional_match"] for r in results]))
    param_omr = float(np.mean([r["param_only_match"] for r in results]))
    step_mr = float(np.mean([r["step_count_match"] for r in results]))
    fp_rate = float(np.mean([r.get("judge_full_parse", False) for r in results]))
    empty_rate = float(np.mean([r["generated_n_steps"] == 0 for r in results]))
    gpt_err_rate = float(np.mean([bool(r.get("gpt_error", "")) for r in results]))
 
    error_types = sorted(set(r["error_type"] for r in results))
    per_error = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n": len(sub),
            "judge_success_rate": float(np.mean([r["judge_success"] for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"] for r in sub])),
            "mean_judge_score": float(np.mean([r["judge_score"] for r in sub])),
            "functional_tool_acc": float(np.mean([r["functional_tool_accuracy"] for r in sub])),
            "mean_param_accuracy": float(np.mean([r["param_accuracy"] for r in sub])),
            "mean_dependency_accuracy": float(np.mean([r["dependency_accuracy"] for r in sub])),
            "exact_match_rate": float(np.mean([r["exact_match"] for r in sub])),
            "functional_match_rate": float(np.mean([r["functional_match"] for r in sub])),
            "param_only_match_rate": float(np.mean([r["param_only_match"] for r in sub])),
            "step_count_match_rate": float(np.mean([r["step_count_match"] for r in sub])),
        }
    success_dist = {}
    for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100 * c / n, 1)}
 
    stats = {
        "label": label,
        "method": f"GPT-{model_name} (greedy, single-shot)",
        "dataset": results[0].get("dataset", "unknown"),
        "n_examples": n,
        "gt_uses_nl_tools": bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(fp_rate, 3),
        "empty_plan_rate": round(empty_rate, 3),
        "gpt_error_rate": round(gpt_err_rate, 3),
        "accuracy": {
            "judge_success_rate": round(judge_sr, 3),
            "error_handled_rate": round(err_hr, 3),
        },
        "judge_scores": {
            "mean": round(float(np.mean(scores)), 2),
            "median": round(float(np.median(scores)), 2),
            "std": round(float(np.std(scores)), 2),
            "pct_gte_80": round(100 * sum(s >= 80 for s in scores) / n, 1),
            "pct_eq_100": round(100 * sum(s == 100 for s in scores) / n, 1),
        },
        "structural": {
            "exact_match_rate": round(exact_mr, 3),
            "functional_match_rate": round(func_mr, 3),
            "param_only_match_rate": round(param_omr, 3),
            "step_count_match_rate": round(step_mr, 3),
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
    print(f"  Method   : GPT-{model_name} (greedy, single-shot)")
    print(f"  N        : {n}")
    if gpt_err_rate > 0:
        print(f"  WARN  GPT API error rate : {100*gpt_err_rate:.1f}%")
    if empty_rate > 0.05:
        print(f"  WARN  Empty plan rate    : {100*empty_rate:.1f}%")
    if fp_rate < 0.9:
        print(f"  WARN  Judge full-parse   : {100*fp_rate:.0f}%")
    print(f"\n  -- Paper-table metrics (JSR / FM / PA / DA) --")
    print(f"    JSR : {100*judge_sr:.1f}%   FM  : {100*func_mr:.1f}%")
    print(f"    PA  : {100*np.mean(param_accs):.1f}%   DA  : {100*np.mean(dep_accs):.1f}%")
    print(f"\n  -- Secondary --")
    print(f"  Step count match : {100*step_mr:.1f}%")
    print(f"  Error handled    : {100*err_hr:.1f}%")
    print(f"  FuncTool acc     : {np.mean(func_tools):.3f}")
    print(f"  Judge mean score : {stats['judge_scores']['mean']:.1f}")
    print(f"  >=80 / =100      : {stats['judge_scores']['pct_gte_80']:.1f}% / "
          f"{stats['judge_scores']['pct_eq_100']:.1f}%")
    if len(error_types) > 1:
        print(f"\n  -- Per error-type --")
        for et, d in per_error.items():
            print(f"    {et:24s}  N={d['n']:>3}  "
                  f"JSR={100*d['judge_success_rate']:>5.1f}%  "
                  f"FM={100*d['functional_match_rate']:>5.1f}%  "
                  f"PA={100*d['mean_param_accuracy']:>5.1f}%")
    print()
    return stats
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(
        description="GPT (closed-source) baseline for FORTE paper comparison"
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-5.1-nano",
                        help="OpenAI model name")
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--dataset", default="auto",
                        choices=["auto", "toolhop", "nestful"],
                        help="Dataset for labels and metadata. 'auto' detects "
                             "from the parquet's data_source field.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Greedy by default (0.0). Match FORTE-BoN1 if "
                             "you want noise (0.2).")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="GPT-5 family burns some output tokens on hidden "
                             "reasoning even with effort=minimal. 1024 leaves "
                             "headroom; bump to 2048 if you see truncation.")
    parser.add_argument("--judge-max-tokens", type=int, default=300)
    parser.add_argument("--judge_url",
                        default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on examples (e.g. --limit 100 for pilot)")
    parser.add_argument("--sleep-time", type=float, default=0.05,
                        help="Pause between API calls (seconds)")
    parser.add_argument("--perfect-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", default="gpt_baseline_results.json")
    parser.add_argument("--stats-output", default=None)
    args = parser.parse_args()
 
    if not args.api_key:
        parser.error("Provide --api-key or set $OPENAI_API_KEY")
    if not args.perfect_only and not args.full:
        parser.error("Specify --perfect-only or --full")
 
    dataset = resolve_dataset(args.dataset, args.test_parquet)
    print(f"\nDataset : {dataset_label(dataset)} ({dataset})")
    print(f"Model   : {args.model}")
    print(f"Temp    : {args.temperature}")
    if args.limit:
        print(f"Limit   : first {args.limit} examples (pilot mode)")
 
    global JUDGE_SERVER_URL
    JUDGE_SERVER_URL = args.judge_url
    try:
        r = requests.get(
            JUDGE_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5
        )
        print(f"OK  Judge server healthy: {r.json()}")
    except Exception as e:
        print(f"FAIL Judge server not reachable: {e}")
        return
 
    client = OpenAI(api_key=args.api_key)
 
    stats_output = args.stats_output or args.output.replace(".json", ".stats.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_output).parent.mkdir(parents=True, exist_ok=True)
 
    all_output = {"config": {**vars(args), "resolved_dataset": dataset}, "runs": {}}
    all_stats = {"config": {**vars(args), "resolved_dataset": dataset}, "runs": {}}
 
    if args.perfect_only:
        examples, pgt = load_test_parquet(args.test_parquet, perfect_only=True)
        if args.limit:
            examples = examples[:args.limit]
        print(f"\nPerfect-only examples: {len(examples)}")
        results = evaluate_gpt(client, examples, pgt, dataset, args)
        stats = compute_stats(
            results,
            f"PERFECT-ONLY  GPT-{args.model} -- {dataset_label(dataset)}",
            args.model,
        )
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"] = stats
 
    if args.full:
        examples, pgt = load_test_parquet(args.test_parquet, perfect_only=False)
        if args.limit:
            examples = examples[:args.limit]
        print(f"\nFull test examples: {len(examples)}")
        results = evaluate_gpt(client, examples, pgt, dataset, args)
        stats = compute_stats(
            results,
            f"FULL  GPT-{args.model} -- {dataset_label(dataset)}",
            args.model,
        )
        all_output["runs"]["full"] = results
        all_stats["runs"]["full"] = stats
 
    with open(args.output, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"\nResults -> {args.output}")
    with open(stats_output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats   -> {stats_output}")
 
 
if __name__ == "__main__":
    main()