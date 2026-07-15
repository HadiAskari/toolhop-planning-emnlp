"""
Shared utilities for the FORTE rebuttal experiments (E1-E6).

Every function in the "VENDORED" sections below is copied verbatim from the
code that produced the paper's numbers:

  - parse_plan_steps / evaluate_plan_vs_gt / normalize_value / helpers:
      scripts/verl-integration-of-trained-planner-and-judge/best_of_n_selection_toolhop.py
  - judge client (JUDGE_SYSTEM_PROMPT, _format_tools_for_judge, score_plan_via_judge):
      same file
  - load_test_parquet: same file
  - reindex_tools_by_api_name: toolhop_plan_generator.py

Do NOT "improve" these functions: they must stay byte-compatible with the
metrics reported in the paper so rebuttal numbers are directly comparable.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # requests is only needed for judge-server calls (E2-E5), not for E1/E6
    import requests
except ImportError:  # pragma: no cover
    requests = None

# ══════════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION
#
# Local layout   : <repo>/ToolHop.json, <repo>/scripts/<subdirs>
# Lab-server     : ${FORTE_ROOT}/<subdirs>
#                  (i.e. the scripts/ prefix is absent on the server)
# Override with  : FORTE_ROOT env var or --repo-root on any CLI.
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_ROOTS = [
    os.environ.get("FORTE_ROOT", ""),
    str(Path(__file__).resolve().parents[2]),          # <repo> (rebuttal-experiments/common/ -> repo root)
    ".",
]

# Relative candidate locations for each well-known data file, tried in order
# under every root.  Covers both the local-git layout and the server layout.
_DATA_FILE_CANDIDATES = {
    "ToolHop.json": [
        "ToolHop.json", "scripts/ToolHop.json",
    ],
    "toolhop_annotated": [
        "toolhop_annotated_v1_remapped.json",
        "scripts/toolhop_annotated_v1_remapped.json",
    ],
    "canonical_splits_toolhop": [
        "canonical_splits.json", "scripts/canonical_splits.json",
    ],
    "canonical_splits_nestful": [
        "canonical_splits_nestful.json", "scripts/canonical_splits_nestful.json",
    ],
    "nestful_data": [
        "nestful_data.jsonl",
        "NESTFUL/data/nestful_data.jsonl",
        "scripts/NESTFUL/data/nestful_data.jsonl",
    ],
    "nestful_annotated": [
        "NESTFUL/data/nestful_annotated_combined.json",
        "scripts/NESTFUL/data/nestful_annotated_combined.json",
        "nestful_annotated_combined.json",
    ],
    "toolhop_plan_generator": [
        "toolhop_plan_generator.py", "scripts/toolhop_plan_generator.py",
    ],
    "nestful_annotator": [
        "nestful_annotator.py", "scripts/nestful_annotator.py",
    ],
}


def find_data_file(key: str, explicit_path: Optional[str] = None,
                   repo_root: Optional[str] = None) -> str:
    """Locate a well-known data file across local/server layouts."""
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Explicit path does not exist: {explicit_path}")

    roots = ([repo_root] if repo_root else []) + [r for r in _DEFAULT_ROOTS if r]
    tried = []
    for root in roots:
        for rel in _DATA_FILE_CANDIDATES[key]:
            p = os.path.join(root, rel)
            tried.append(p)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"Could not locate data file '{key}'. Tried:\n  " + "\n  ".join(tried)
        + "\nSet FORTE_ROOT or pass an explicit path."
    )


def import_module_from_file(path: str, module_name: str):
    """Import e.g. toolhop_plan_generator.py / nestful_annotator.py by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# VENDORED: constants  (best_of_n_selection_toolhop.py)
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_ERROR_TYPES = {
    "circular_dependency", "forward_reference", "incomplete_plan",
    "inefficient_order", "missing_dependency", "parameter_typo",
    "type_mismatch", "unnecessary_steps", "wrong_tool",
}

# Fixed candidate order inside the annotated corpora: index qid*10+k
ANNOTATED_ERROR_ORDER = [
    "none", "type_mismatch", "missing_dependency", "wrong_tool",
    "parameter_typo", "circular_dependency", "inefficient_order",
    "incomplete_plan", "unnecessary_steps", "forward_reference",
]

DEFAULT_TEMPERATURE_LADDER: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]


# ══════════════════════════════════════════════════════════════════════════════
# VENDORED: planner prompt  (best_of_n_selection_toolhop.py)
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
        f"3. Reference previous step outputs using {{{{N}}}} - never substitute a hardcoded value for an output that comes from a prior step\n"
        f"4. Use the exact parameter names shown in the tool signatures above\n"
        f"5. Provide all required parameters\n\n"
        f"Generate only the steps the query requires - no redundant steps, no missing steps.\n\n"
        f"Format each step as: Step N: {{{{N}}}} = tool_name(param1=value1, param2=value2, ...)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# VENDORED: judge server client  (best_of_n_selection_toolhop.py)
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

_http_session = None


def set_judge_url(url: str) -> None:
    global JUDGE_SERVER_URL
    JUDGE_SERVER_URL = url


def _get_session():
    global _http_session
    if requests is None:
        raise ImportError("The 'requests' package is required for judge-server calls")
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
    last_err = ""
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

        except requests.HTTPError as e:
            body = e.response.text[:200] if e.response is not None else ""
            last_err = f"HTTP {e.response.status_code if e.response is not None else '?'}: {body}"
            if attempt < retries - 1:
                time.sleep(2.0)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = f"{type(e).__name__}: {e}"
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
            last_err = f"unparseable judge response: {content[:120]!r}"
            break

    return {
        "quality_score": 0, "success_prediction": "no",
        "reasoning": f"judge call failed ({last_err or 'unknown error'})",
        "issues": [], "confidence": 0.0,
        "_full_parse": False, "_call_failed": True,
    }


def batch_score_plans(items: List[Dict[str, Any]], max_tokens: int = 32,
                      workers: int = 8, desc: str = "judge",
                      max_failed_frac: float = 0.05) -> List[Dict[str, Any]]:
    """
    Score many plans against the judge server concurrently.

    items: list of {"query": str, "plan_str": str, "tools": dict}
    Returns annotations in the same order as `items`.

    Raises RuntimeError if more than `max_failed_frac` of the calls hard-fail
    (HTTP error / connection error / unparseable response): a failed call is
    recorded as score 0, so past that fraction the aggregate stats are
    meaningless and must not be reported.
    """
    results: List[Optional[Dict]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(score_plan_via_judge, it["query"], it["plan_str"],
                        it["tools"], max_tokens): idx
            for idx, it in enumerate(items)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:  # never lose the slot
                results[idx] = {
                    "quality_score": 0, "success_prediction": "no",
                    "reasoning": f"scoring exception: {e}", "issues": [],
                    "confidence": 0.0, "_full_parse": False,
                    "_call_failed": True,
                }
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"  [{desc}] scored {done}/{len(items)}", flush=True)

    n_failed = sum(1 for r in results if r and r.get("_call_failed"))
    if n_failed:
        frac = n_failed / len(items)
        example = next(r["reasoning"] for r in results if r and r.get("_call_failed"))
        msg = (f"{n_failed}/{len(items)} ({frac:.1%}) judge calls failed; "
               f"first failure: {example}")
        if frac > max_failed_frac:
            raise RuntimeError(
                f"[{desc}] {msg}\nFailed calls score 0, so these results are "
                f"invalid. Check the judge server (GPU OOM? wrong URL?) and rerun.")
        print(f"  [{desc}] WARNING: {msg}", flush=True)
    return results  # type: ignore


def check_judge_server(url: Optional[str] = None) -> bool:
    """Ping the judge server /health endpoint."""
    target = (url or JUDGE_SERVER_URL).replace("/v1/chat/completions", "/health")
    try:
        r = requests.get(target, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# VENDORED: plan parsing + structural metrics  (best_of_n_selection_toolhop.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# VENDORED: test parquet loader  (best_of_n_selection_toolhop.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# VENDORED: tools reindexing  (toolhop_plan_generator.py)
# ══════════════════════════════════════════════════════════════════════════════

def reindex_tools_by_api_name(tools_raw: Dict) -> Dict[str, Dict]:
    reindexed: Dict[str, Dict] = {}
    for nl_key, tool_spec in tools_raw.items():
        api_name = tool_spec.get("name") if isinstance(tool_spec, dict) else None
        if not api_name:
            sanitized = re.sub(r"[^a-z0-9_]+", "_", nl_key.lower()).strip("_")
            api_name = sanitized[:60] if sanitized else "unnamed_tool"
        if api_name not in reindexed:
            reindexed[api_name] = tool_spec
    return reindexed


# ══════════════════════════════════════════════════════════════════════════════
# Annotated-corpus helpers (new code, matches formats verified in the corpora)
# ══════════════════════════════════════════════════════════════════════════════

def format_steps_as_plan_string(steps: List[Dict[str, Any]]) -> str:
    """
    Serialize corpus-style step dicts to the canonical plan string.
    Matches LLMJudgeAnnotator._format_plan / Step.__str__ exactly:
        Step {step_id}: {output_variable} = {tool_name}(k=repr(v), ...)
    """
    lines = []
    for s in steps:
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in s["parameters"].items())
        lines.append(f"Step {s['step_id']}: {s['output_variable']} = {s['tool_name']}({params_str})")
    return "\n".join(lines)


def load_annotated_corpus(dataset: str, path: Optional[str] = None,
                          repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Load the full annotated contrastive corpus for 'toolhop' or 'nestful'."""
    key = "toolhop_annotated" if dataset == "toolhop" else "nestful_annotated"
    p = find_data_file(key, explicit_path=path, repo_root=repo_root)
    with open(p) as f:
        corpus = json.load(f)
    print(f"[data] loaded annotated corpus: {p} "
          f"({len(corpus['data'])} plans, {corpus['metadata'].get('n_queries', '?')} queries)")
    return corpus


def load_canonical_splits(dataset: str, path: Optional[str] = None,
                          repo_root: Optional[str] = None) -> Dict[str, Any]:
    key = ("canonical_splits_toolhop" if dataset == "toolhop"
           else "canonical_splits_nestful")
    p = find_data_file(key, explicit_path=path, repo_root=repo_root)
    with open(p) as f:
        return json.load(f)


def corpus_records_by_qid(corpus: Dict[str, Any]) -> Dict[int, List[Dict]]:
    """Group the flat annotated data list into {query_id: [10 candidate records]}."""
    by_qid: Dict[int, List[Dict]] = {}
    for rec in corpus["data"]:
        by_qid.setdefault(rec["query_id"], []).append(rec)
    return by_qid


def gold_record(records: List[Dict]) -> Optional[Dict]:
    """The gold (error_type == 'none') record among a query's candidates."""
    for rec in records:
        if rec["plan"].get("error_type") == "none":
            return rec
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Small stats helpers (no sklearn/scipy dependency)
# ══════════════════════════════════════════════════════════════════════════════

def mann_whitney_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """
    AUC for 'positive class scores higher than negative class scores',
    computed as the normalized Mann-Whitney U statistic with tie correction.
    """
    if not pos_scores or not neg_scores:
        return float("nan")
    combined = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    combined.sort(key=lambda x: x[0])
    # average ranks with ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, combined) if lab == 1)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def cohens_kappa(labels_a: List[Any], labels_b: List[Any]) -> float:
    """Cohen's kappa for two annotators over identical items."""
    assert len(labels_a) == len(labels_b) and labels_a
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b), key=str)
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for a in labels_a if a == c) / n
        pb = sum(1 for b in labels_b if b == c) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[out] wrote {path}")
