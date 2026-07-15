#!/usr/bin/env python3
r"""
Robust plan parser + rescue script for Tool-Planner / planner outputs.

THE BUG
-------
`parse_plan_steps` in tool_planner_baseline.py / best_of_n_selection.py drops
every step line that lacks an explicit `{{N}} =` variable assignment:

    var_match = re.search(r"(\{\{\d+\}\})\s*=", line)
    if not var_match: continue   # <-- silent skip, no log

On NESTFUL with smaller models the generator commonly emits:

    Step 0: unique_common_values(list1=[1,2,3], list2=[4,5,6])
    Step 1: swap_by_index(values={{0}}, i=0, j=1)

i.e. the first step omits the variable assignment. `_extract_step_lines` still
keeps those lines (its regex is `\s*Step \d+:`), the judge still scores the
plan, but `parse_plan_steps` returns `[]`. That makes generated_n_steps=0,
marks the example "empty", and zeroes every structural metric.

For Tool-Planner-3B on NESTFUL the visible effect is:
    empty_plan_rate          = 0.889   <-- bogus (parse failure, not empty)
    mean_param_accuracy      = 0.098   <-- bogus (zero-rowed by parse failure)
    mean_dependency_accuracy = 0.108   <-- bogus
    judge_success_rate       = 0.845   <-- real signal, unaffected

OTHER RECOVERABLE PATTERNS
--------------------------
The robust parser also handles, in decreasing order of likelihood:
  1. Missing {{N}} = assignment (the main culprit; covered above)
  2. Alt variable syntax: {0} = tool(...), var0 = tool(...), result_0 = tool(...)
  3. Markdown decorations on the step line: **Step 0:**, ### Step 0
  4. Alt step keywords: Action 0:, Call 0:, Operation 0:
  5. Alt punctuation: Step 0), Step 0 -, Step 0 ->
  6. Code-fence wrapping around the whole plan
  7. Reasoning / preamble text interleaved between steps

CAVEAT
------
Cases (3)-(7) above are only recoverable if the original `_extract_step_lines`
already kept the line. With the current code, anything not matching
`\s*Step \d+:` is dropped BEFORE the saved generated_plan, so we cannot recover
it from the results JSON. For those, you need to re-run inference with the
robust extractor (also provided below).

USAGE
-----
    # 1. Inspect what's broken in a results file:
    python rescue_plans.py diagnose --input tool_planner_qwen-3B-results.json

    # 2. Re-parse and write a corrected results JSON + stats:
    python rescue_plans.py rescue \
        --input  ${FORTE_ROOT}/baselines/tool_planner_Qwen2.5-3B-Instruct_results.json \
        --output ${FORTE_ROOT}/baselines/tool_planner_Qwen2.5-3B-Instruct_results.rescued.json

    # 3. Self-test:
    python rescue_plans.py test

PATCHING THE PIPELINE
---------------------
For future runs, replace `parse_plan_steps` and `_extract_step_lines` in
tool_planner_baseline.py / best_of_n_selection.py with `parse_plan_steps_robust`
and `extract_step_lines_robust` from this file (drop-in compatible).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS (mirrored from tool_planner_baseline.py / best_of_n_selection.py)
# ════════════════════════════════════════════════════════════════════════════

ARTIFACT_ERROR_TYPES = {
    "circular_dependency", "forward_reference", "incomplete_plan",
    "inefficient_order", "missing_dependency", "parameter_typo",
    "type_mismatch", "unnecessary_steps", "wrong_tool",
}

_STOP = {
    "what", "is", "the", "of", "in", "a", "an", "and", "or", "to", "how",
    "many", "who", "which", "are", "was", "were", "be", "been", "at", "on",
    "for", "with", "that", "this", "it", "its", "from",
}


# ════════════════════════════════════════════════════════════════════════════
# ROBUST EXTRACT + PARSE
# ════════════════════════════════════════════════════════════════════════════

# Step header. Tolerates:
#   "Step 0:", "**Step 0:**", "### Step 0", "  - Step 0)",
#   "Action 1:", "Call 2 -", "Operation 0:", "Plan 0:", "Stage 1:",
#   case-insensitive on the keyword.
_STEP_HEADER = re.compile(
    r"""^
    [\s>*#`\-•·]*                                  # leading bullets / md noise
    (?:\*+)?                                       # optional bold open
    (?:\#+\s*)?                                    # optional md header
    (?:`)?                                         # optional inline code open
    (?:Step|Action|Call|Operation|Plan|Stage)
    \s+
    (\d+)                                          # step number  -> group(1)
    \s*
    [:.)\-—–>]                                     # punctuation after number
    \s*
    (?:`)?                                         # optional inline code close
    (?:\*+)?                                       # optional bold close
    \s*
    (.*)$                                          # rest of the line -> group(2)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Assignment patterns, in canonical and various recoverable forms.
#   {{0}}=     <-- canonical
#   {0}=       <-- single brace
#   var0=, var_0=, result0=, result_0=, output0=, out0=, r0=, x0=, step0=, s0=
_ASSIGN_PATTERN = re.compile(
    r"""^
    (
        \{\{\s*\d+\s*\}\}                          # {{0}}
      | \{\s*\d+\s*\}                              # {0}
      | (?:var|result|output|out|r|x|step|s)_?\d+  # var0, result_0, etc.
    )
    \s*=\s*
    (.+)$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Final tool-call pattern: name(args)
_TOOL_CALL = re.compile(r"^([^(]+)\((.*)\)\s*$")
_TOOL_CALL_UNCLOSED = re.compile(r"^([^(]+)\((.*)$")  # fallback w/o closing paren


def _strip_code_fences(text: str) -> str:
    """Remove ``` and ```lang fences so step lines inside them are reachable."""
    if not text:
        return ""
    text = re.sub(r"```[a-zA-Z0-9_+\-]*\n?", "", text)
    text = text.replace("```", "")
    return text


def _normalize_var(raw_var: str, fallback_step_id: int) -> str:
    """Normalize any variable token to canonical {{N}} form."""
    m = re.search(r"\d+", raw_var)
    n = int(m.group()) if m else fallback_step_id
    return f"{{{{{n}}}}}"


def _normalize_ref_in_value(v: str) -> str:
    """
    Inside parameter values, normalize ref tokens to {{N}} so that the
    downstream dependency_accuracy metric (which greps for {{N}}) works.
    """
    if not isinstance(v, str):
        return v
    # {0} -> {{0}}
    v = re.sub(r"(?<!\{)\{(\s*\d+\s*)\}(?!\})", r"{{\1}}", v)
    # var0, result_0, etc. when used as a bare ref (whole-string match)
    m = re.fullmatch(
        r"\s*(?:var|result|output|out|r|x|step|s)_?(\d+)\s*",
        v,
        re.IGNORECASE,
    )
    if m:
        return f"{{{{{int(m.group(1))}}}}}"
    return v


def _split_params(params_str: str) -> Dict[str, str]:
    """Depth-aware split on top-level commas, then split each piece on the
    first top-level `=`. Returns {} for empty input."""
    params: Dict[str, str] = {}
    if not params_str.strip():
        return params

    parts: List[str] = []
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
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
                continue
        current += ch
    if current.strip():
        parts.append(current.strip())

    for part in parts:
        depth = 0
        in_str = False
        str_char = None
        eq_idx = -1
        for i, ch in enumerate(part):
            if ch in ('"', "'") and (not in_str or ch == str_char):
                in_str = not in_str
                str_char = ch if in_str else None
            if not in_str:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)
                elif ch == "=" and depth == 0:
                    eq_idx = i
                    break
        if eq_idx > 0:
            k = part[:eq_idx].strip()
            v = part[eq_idx + 1 :].strip()
            v = _normalize_ref_in_value(v)
            params[k] = v
    return params


def extract_step_lines_robust(raw_text: str) -> str:
    """
    Drop-in replacement for `_extract_step_lines` in tool_planner_baseline.py.

    Keeps every line that the robust header regex recognizes as a step header,
    after stripping code fences. Strips markdown bullet/header decorations and
    renormalizes the keyword to "Step N:" so downstream parsing is uniform.
    """
    raw_text = _strip_code_fences(raw_text or "")
    out_lines: List[str] = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _STEP_HEADER.match(line)
        if not m:
            continue
        n = int(m.group(1))
        rest = m.group(2).strip()
        rest = re.sub(r"[\s*`]+$", "", rest).strip()
        out_lines.append(f"Step {n}: {rest}")
    return "\n".join(out_lines)


def parse_plan_steps_robust(
    plan_text: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Drop-in replacement for `parse_plan_steps`.

    Returns (steps, diag). `steps` has the same shape as the original:
        [{"step_id": int, "output_variable": str,
          "tool_name": str, "parameters": Dict[str,str]}, ...]

    `diag` is a small dict counting which recovery rules fired -- handy for
    figuring out what's actually going on in your generations:
        lines_total                       -- non-blank lines seen
        lines_with_step_header            -- matched the step header regex
        lines_strict_match                -- canonical {{N}} = tool(...)
        lines_recovered_no_assignment     -- "Step N: tool(...)" (no `{{N}} =`)
        lines_recovered_alt_var_syntax    -- "Step N: var0 = tool(...)" etc.
        lines_skipped_no_tool_call        -- header found but no `name(args)`
    """
    diag = {
        "lines_total": 0,
        "lines_with_step_header": 0,
        "lines_strict_match": 0,
        "lines_recovered_no_assignment": 0,
        "lines_recovered_alt_var_syntax": 0,
        "lines_skipped_no_tool_call": 0,
    }
    steps: List[Dict[str, Any]] = []
    plan_text = _strip_code_fences(plan_text or "")

    for raw_line in plan_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        diag["lines_total"] += 1

        header = _STEP_HEADER.match(line)
        if not header:
            continue
        diag["lines_with_step_header"] += 1

        step_id = int(header.group(1))
        rest = header.group(2).strip()
        rest = re.sub(r"[\s*`]+$", "", rest).strip()

        assign = _ASSIGN_PATTERN.match(rest)
        if assign:
            raw_var = assign.group(1)
            tool_call = assign.group(2).strip()
            output_var = _normalize_var(raw_var, step_id)
            if raw_var.lstrip().startswith("{{"):
                diag["lines_strict_match"] += 1
            else:
                diag["lines_recovered_alt_var_syntax"] += 1
        else:
            tool_call = rest
            output_var = f"{{{{{step_id}}}}}"
            diag["lines_recovered_no_assignment"] += 1

        # Strip a trailing semicolon / period sometimes added by the model
        tool_call = tool_call.rstrip(";.").strip()

        m = _TOOL_CALL.match(tool_call)
        if not m:
            m = _TOOL_CALL_UNCLOSED.match(tool_call)
            if not m:
                diag["lines_skipped_no_tool_call"] += 1
                continue
        tool_name = m.group(1).strip()
        params = _split_params(m.group(2).rstrip(")").strip())

        steps.append({
            "step_id": step_id,
            "output_variable": output_var,
            "tool_name": tool_name,
            "parameters": params,
        })

    return steps, diag


# ════════════════════════════════════════════════════════════════════════════
# STRUCTURAL EVAL (faithful reimplementation of evaluate_plan_vs_gt)
# ════════════════════════════════════════════════════════════════════════════

def _kw(s: str) -> set:
    return {
        w for w in re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
        if w not in _STOP and len(w) > 2
    }


def _functional_tool_match(g: str, r: str) -> float:
    gk, rk = _kw(g), _kw(r)
    if not gk or not rk:
        return 0.0
    return round(len(gk & rk) / len(gk | rk), 3)


def _norm(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())


def evaluate_plan_vs_gt(
    gen_steps: List[Dict], gt_steps: List[Dict]
) -> Dict[str, Any]:
    """Mirror of evaluate_plan_vs_gt in tool_planner_baseline.py."""
    empty = {
        "valid": False, "step_count_match": False,
        "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
        "param_accuracy": 0.0, "dependency_accuracy": 1.0,
        "exact_match": False, "functional_match": False,
        "param_only_match": False,
        "generated_steps": len(gen_steps), "ground_truth_steps": len(gt_steps),
    }
    if not gen_steps or not gt_steps:
        return empty

    step_count_match = len(gen_steps) == len(gt_steps)
    ce, tf, tpc, tp, cd, td = 0, 0.0, 0, 0, 0, 0
    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt = gt_steps[i] if i < len(gt_steps) else None
        if not (gen and gt):
            continue
        if gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower():
            ce += 1
        tf += _functional_tool_match(gen["tool_name"], gt["tool_name"])
        gt_keys = set(gt["parameters"].keys())
        gen_keys = set(gen["parameters"].keys())
        for k in gt_keys & gen_keys:
            gv = _norm(gt["parameters"][k])
            dv = _norm(gen["parameters"][k])
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
        "valid": True,
        "step_count_match": step_count_match,
        "generated_steps": len(gen_steps),
        "ground_truth_steps": n_gt,
        "exact_tool_accuracy": ea,
        "functional_tool_accuracy": fa,
        "param_accuracy": pa,
        "dependency_accuracy": da,
        "exact_match": step_count_match and ea == 1.0 and pa == 1.0,
        "functional_match": step_count_match and fa >= 0.5 and pa >= 0.5,
        "param_only_match": pa >= 0.5,
    }


# ════════════════════════════════════════════════════════════════════════════
# RESCUE PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def build_perfect_gt_lookup(results: List[Dict]) -> Dict[int, str]:
    """For each query_id, find the 'none' error_type record and treat its
    ground_truth as the perfect plan (used for structural eval on the 9
    artifact error types, matching the original pipeline behavior)."""
    lookup: Dict[int, str] = {}
    for r in results:
        if r.get("error_type") == "none" and int(r.get("ref_quality_score", 0)) >= 100:
            qid = r.get("query_id")
            if qid is not None and qid not in lookup:
                lookup[qid] = r.get("ground_truth", "")
    return lookup


def rescue_results(
    results: List[Dict[str, Any]],
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Re-parse every record's `generated_plan` with the robust parser and
    recompute structural metrics. Returns (updated_results, summary).
    """
    perfect_gt = build_perfect_gt_lookup(results)
    if verbose:
        print(f"  Built perfect-GT lookup for {len(perfect_gt)} query_ids.")

    diag_total: Dict[str, int] = defaultdict(int)
    n_changed_steps = 0
    n_recovered_from_empty = 0
    n_still_empty = 0

    for r in results:
        gp = r.get("generated_plan", "")
        gt_raw = r.get("ground_truth", "")

        old_n_steps = int(r.get("generated_n_steps", 0))

        gen_steps, diag = parse_plan_steps_robust(gp)
        for k, v in diag.items():
            diag_total[k] += v

        # Structural GT: remap to perfect GT for any non-`none` error type
        # if a perfect GT exists for this query_id (matches original pipeline).
        err_type = r.get("error_type", "none")
        qid = r.get("query_id")
        if err_type in ARTIFACT_ERROR_TYPES and qid in perfect_gt:
            gt_steps, _ = parse_plan_steps_robust(perfect_gt[qid])
        else:
            gt_steps, _ = parse_plan_steps_robust(gt_raw)

        ev = evaluate_plan_vs_gt(gen_steps, gt_steps)

        new_n_steps = ev["generated_steps"]
        if new_n_steps != old_n_steps:
            n_changed_steps += 1
        if old_n_steps == 0 and new_n_steps > 0:
            n_recovered_from_empty += 1
        if new_n_steps == 0:
            n_still_empty += 1

        # Write updated fields back into the record (in place).
        r["generated_n_steps"] = new_n_steps
        r["gt_n_steps"] = ev["ground_truth_steps"]
        r["exact_tool_accuracy"] = ev["exact_tool_accuracy"]
        r["functional_tool_accuracy"] = ev["functional_tool_accuracy"]
        r["param_accuracy"] = ev["param_accuracy"]
        r["dependency_accuracy"] = ev["dependency_accuracy"]
        r["exact_match"] = ev["exact_match"]
        r["functional_match"] = ev["functional_match"]
        r["param_only_match"] = ev["param_only_match"]
        r["step_count_match"] = ev["step_count_match"]

    summary = {
        "n_records": len(results),
        "n_step_count_changed": n_changed_steps,
        "n_recovered_from_empty": n_recovered_from_empty,
        "n_still_empty": n_still_empty,
        "robust_parser_diagnostics": dict(diag_total),
    }
    return results, summary


def compute_aggregate_stats(
    results: List[Dict[str, Any]], label: str
) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}

    def _mean(key, cast=float):
        return float(np.mean([cast(r[key]) for r in results]))

    judge_scores = [int(r.get("judge_score", 0)) for r in results]
    func_tools = [r["functional_tool_accuracy"] for r in results]
    param_accs = [r["param_accuracy"] for r in results]
    dep_accs = [r["dependency_accuracy"] for r in results]

    error_types = sorted({r.get("error_type", "none") for r in results})
    per_error: Dict[str, Dict[str, float]] = {}
    for et in error_types:
        sub = [r for r in results if r.get("error_type") == et]
        per_error[et] = {
            "n": len(sub),
            "judge_success_rate": _mean_bool(sub, "judge_success"),
            "mean_judge_score": float(np.mean([r.get("judge_score", 0) for r in sub])),
            "functional_tool_acc": float(np.mean([r["functional_tool_accuracy"] for r in sub])),
            "mean_param_accuracy": float(np.mean([r["param_accuracy"] for r in sub])),
            "mean_dependency_accuracy": float(np.mean([r["dependency_accuracy"] for r in sub])),
            "exact_match_rate": _mean_bool(sub, "exact_match"),
            "functional_match_rate": _mean_bool(sub, "functional_match"),
            "param_only_match_rate": _mean_bool(sub, "param_only_match"),
            "step_count_match_rate": _mean_bool(sub, "step_count_match"),
        }

    return {
        "label": label,
        "n_examples": n,
        "empty_plan_rate": float(np.mean([r["generated_n_steps"] == 0 for r in results])),
        "accuracy": {
            "judge_success_rate": _mean_bool(results, "judge_success"),
            "error_handled_rate": _mean_bool(results, "error_type_handled"),
        },
        "judge_scores": {
            "mean": float(np.mean(judge_scores)),
            "median": float(np.median(judge_scores)),
            "pct_gte_80": round(100 * sum(s >= 80 for s in judge_scores) / n, 1),
            "pct_eq_100": round(100 * sum(s == 100 for s in judge_scores) / n, 1),
        },
        "structural": {
            "exact_match_rate": _mean_bool(results, "exact_match"),
            "functional_match_rate": _mean_bool(results, "functional_match"),
            "param_only_match_rate": _mean_bool(results, "param_only_match"),
            "step_count_match_rate": _mean_bool(results, "step_count_match"),
            "mean_functional_tool_acc": float(np.mean(func_tools)),
            "mean_param_accuracy": float(np.mean(param_accs)),
            "mean_dependency_accuracy": float(np.mean(dep_accs)),
        },
        "per_error_type": per_error,
    }


def _mean_bool(records, key) -> float:
    return float(np.mean([bool(r.get(key, False)) for r in records]))


def print_aggregate(label: str, s: Dict[str, Any]) -> None:
    print(f"\n  === {label} ===")
    print(f"    N examples:               {s['n_examples']}")
    print(f"    Empty plan rate:          {100*s['empty_plan_rate']:.1f}%")
    print(f"    Judge success rate:       {100*s['accuracy']['judge_success_rate']:.1f}%")
    print(f"    Error-type handled rate:  {100*s['accuracy']['error_handled_rate']:.1f}%")
    print(f"    Exact match rate:         {100*s['structural']['exact_match_rate']:.1f}%")
    print(f"    Functional match rate:    {100*s['structural']['functional_match_rate']:.1f}%")
    print(f"    Param-only match rate:    {100*s['structural']['param_only_match_rate']:.1f}%")
    print(f"    Step count match rate:    {100*s['structural']['step_count_match_rate']:.1f}%")
    print(f"    Mean functional tool acc: {s['structural']['mean_functional_tool_acc']:.3f}")
    print(f"    Mean param accuracy:      {s['structural']['mean_param_accuracy']:.3f}")
    print(f"    Mean dependency acc:      {s['structural']['mean_dependency_accuracy']:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _iter_run_results(bundle: Dict[str, Any]):
    """Yield (run_name, results_list) for every run that's a list of records."""
    runs = bundle.get("runs", {})
    for run_name, run_data in runs.items():
        if isinstance(run_data, list):
            yield run_name, run_data


def cmd_diagnose(args: argparse.Namespace) -> None:
    with open(args.input) as f:
        bundle = json.load(f)

    found = False
    for run_name, run_data in _iter_run_results(bundle):
        found = True
        print(f"\n=== {run_name}  ({len(run_data)} examples) ===")

        n_total = len(run_data)
        n_empty_old = 0
        n_empty_with_judge_ok = 0
        n_recovered = 0
        recovered_samples: List[Tuple[Any, Any, str, int]] = []
        still_broken_samples: List[Tuple[Any, Any, str]] = []
        diag_total: Counter = Counter()

        for r in run_data:
            gp = r.get("generated_plan", "")
            judge_ok = bool(r.get("judge_success"))
            old_n_steps = int(r.get("generated_n_steps", 0))

            new_steps, diag = parse_plan_steps_robust(gp)
            for k, v in diag.items():
                diag_total[k] += v

            if old_n_steps == 0:
                n_empty_old += 1
                if judge_ok:
                    n_empty_with_judge_ok += 1
                if new_steps:
                    n_recovered += 1
                    if len(recovered_samples) < 3:
                        recovered_samples.append(
                            (r.get("query_id"), r.get("error_type"), gp, len(new_steps))
                        )
                else:
                    if len(still_broken_samples) < 3 and gp.strip():
                        still_broken_samples.append(
                            (r.get("query_id"), r.get("error_type"), gp)
                        )

        print(f"  Records flagged empty by old parser:      "
              f"{n_empty_old}/{n_total} ({100*n_empty_old/max(n_total,1):.1f}%)")
        print(f"  ... AND given judge_success by judge:     "
              f"{n_empty_with_judge_ok}/{max(n_empty_old,1)} "
              f"({100*n_empty_with_judge_ok/max(n_empty_old,1):.1f}%)  "
              f"<-- these are the suspicious ones")
        print(f"  Recoverable by robust parser:             "
              f"{n_recovered}/{max(n_empty_old,1)} "
              f"({100*n_recovered/max(n_empty_old,1):.1f}%)")
        print(f"\n  Robust-parser diagnostics across all {n_total} records:")
        for k in [
            "lines_total", "lines_with_step_header",
            "lines_strict_match", "lines_recovered_no_assignment",
            "lines_recovered_alt_var_syntax", "lines_skipped_no_tool_call",
        ]:
            print(f"    {k:38s} {diag_total[k]}")

        if recovered_samples:
            print(f"\n  Sample RECOVERED cases (old parser saw empty, robust gets N steps):")
            for qid, et, gp, n_steps in recovered_samples:
                preview = gp.replace("\n", " | ")[:180]
                print(f"    qid={qid} et={et}: recovered {n_steps} steps")
                print(f"      generated_plan[:180]: {preview!r}")

        if still_broken_samples:
            print(f"\n  Sample STILL-BROKEN cases (saved generated_plan has content"
                  f" but robust parser also can't recover):")
            for qid, et, gp in still_broken_samples:
                preview = gp.replace("\n", " | ")[:180]
                print(f"    qid={qid} et={et}")
                print(f"      generated_plan[:180]: {preview!r}")
            print(f"    ^ these can only be recovered by re-running inference with"
                  f"\n      extract_step_lines_robust + parse_plan_steps_robust.")

    if not found:
        print("ERROR: no run list found in the bundle. Expected bundle['runs'][name] = [records...]")
        sys.exit(1)


def cmd_rescue(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    with in_path.open() as f:
        bundle = json.load(f)

    found = False
    new_stats_bundle: Dict[str, Any] = {"config": bundle.get("config", {}), "runs": {}}

    for run_name, run_data in _iter_run_results(bundle):
        found = True
        print(f"\nRescuing run: {run_name}  ({len(run_data)} records)")

        rescued, summary = rescue_results(run_data, verbose=True)
        print(f"  Records with step-count change:    {summary['n_step_count_changed']}")
        print(f"  Records recovered from old-empty:  {summary['n_recovered_from_empty']}")
        print(f"  Records still empty after rescue:  {summary['n_still_empty']}")
        diag = summary["robust_parser_diagnostics"]
        print(f"  Robust-parser line tallies:")
        for k, v in diag.items():
            print(f"    {k:38s} {v}")

        agg = compute_aggregate_stats(rescued, label=f"RESCUED  {run_name}")
        new_stats_bundle["runs"][run_name] = agg
        print_aggregate(f"RESCUED  {run_name}", agg)

    if not found:
        print("ERROR: no run list found in the bundle.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nRescued results written to: {out_path}")

    stats_path = (
        Path(args.stats_output)
        if args.stats_output
        else out_path.with_suffix(".stats.json")
    )
    with stats_path.open("w") as f:
        json.dump(new_stats_bundle, f, indent=2)
    print(f"Rescued stats   written to: {stats_path}")


def cmd_test(args: argparse.Namespace) -> None:
    print("Running self-tests...")

    # 1. Clean format (the way query_id=5 looks in the saved JSON).
    good = (
        "Step 0: {{0}} = unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "Step 1: {{1}} = swap_by_index(values={{0}}, i=0, j=1)"
    )
    steps, diag = parse_plan_steps_robust(good)
    assert len(steps) == 2, f"clean: got {len(steps)} steps"
    assert steps[0]["tool_name"] == "unique_common_values"
    assert steps[0]["output_variable"] == "{{0}}"
    assert steps[1]["parameters"]["values"] == "{{0}}"
    assert diag["lines_strict_match"] == 2
    print("  PASS  clean canonical format")

    # 2. The primary NESTFUL failure mode: no {{N}} = assignment.
    no_assign = (
        "Step 0: unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "Step 1: swap_by_index(values={{0}}, i=0, j=1)"
    )
    steps, diag = parse_plan_steps_robust(no_assign)
    assert len(steps) == 2, f"no_assign: got {len(steps)} steps"
    assert steps[0]["output_variable"] == "{{0}}"
    assert steps[1]["output_variable"] == "{{1}}"
    assert steps[0]["tool_name"] == "unique_common_values"
    assert diag["lines_recovered_no_assignment"] == 2
    print("  PASS  no-assignment recovery (the main bug)")

    # 3. Alt variable syntax.
    alt_var = (
        "Step 0: var0 = unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "Step 1: var1 = swap_by_index(values=var0, i=0, j=1)"
    )
    steps, diag = parse_plan_steps_robust(alt_var)
    assert len(steps) == 2
    assert steps[0]["output_variable"] == "{{0}}"
    assert steps[1]["output_variable"] == "{{1}}"
    assert steps[1]["parameters"]["values"] == "{{0}}", steps[1]["parameters"]
    assert diag["lines_recovered_alt_var_syntax"] == 2
    print("  PASS  alt variable syntax (var0 -> {{0}})")

    # 4. Single-brace.
    single_brace = (
        "Step 0: {0} = unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "Step 1: {1} = swap_by_index(values={0}, i=0, j=1)"
    )
    steps, _ = parse_plan_steps_robust(single_brace)
    assert len(steps) == 2
    assert steps[0]["output_variable"] == "{{0}}"
    assert steps[1]["parameters"]["values"] == "{{0}}"
    print("  PASS  single-brace syntax ({0} -> {{0}})")

    # 5. Markdown decorations.
    md = (
        "**Step 0:** {{0}} = unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "**Step 1:** {{1}} = swap_by_index(values={{0}}, i=0, j=1)"
    )
    steps, _ = parse_plan_steps_robust(md)
    assert len(steps) == 2
    print("  PASS  markdown-decorated step headers")

    # 6. Alt step keyword.
    alt_kw = (
        "Action 0: {{0}} = unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "Action 1: {{1}} = swap_by_index(values={{0}}, i=0, j=1)"
    )
    steps, _ = parse_plan_steps_robust(alt_kw)
    assert len(steps) == 2
    print("  PASS  alt step keyword (Action 0:)")

    # 7. Reasoning interleaved.
    interleaved = (
        "First, I need to find the common values.\n"
        "Step 0: {{0}} = unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "Then swap them.\n"
        "Step 1: {{1}} = swap_by_index(values={{0}}, i=0, j=1)\n"
        "That should do it."
    )
    steps, _ = parse_plan_steps_robust(interleaved)
    assert len(steps) == 2
    print("  PASS  reasoning interleaved between steps")

    # 8. Code fence.
    fenced = (
        "```python\n"
        "Step 0: {{0}} = unique_common_values(list1=[1, 2, 3], list2=[4, 5, 6])\n"
        "Step 1: {{1}} = swap_by_index(values={{0}}, i=0, j=1)\n"
        "```"
    )
    steps, _ = parse_plan_steps_robust(fenced)
    assert len(steps) == 2
    print("  PASS  code-fence wrapped plan")

    # 9. End-to-end: structural eval on a no-assignment plan vs canonical GT.
    gt_text = (
        "Step 0: {{0}} = unique_common_values(list1=[1, 2, 3, 4, 5], list2=[4, 5, 6, 7, 8])\n"
        "Step 1: {{1}} = swap_by_index(values={{0}}, i=0, j=1)"
    )
    gen_text_broken = (
        "Step 0: unique_common_values(list1=[1, 2, 3, 4, 5], list2=[4, 5, 6, 7, 8])\n"
        "Step 1: swap_by_index(values={{0}}, i=0, j=1)"
    )
    gt_steps, _ = parse_plan_steps_robust(gt_text)
    gen_steps, _ = parse_plan_steps_robust(gen_text_broken)
    ev = evaluate_plan_vs_gt(gen_steps, gt_steps)
    assert ev["param_accuracy"] == 1.0, ev
    assert ev["functional_match"] is True, ev
    assert ev["dependency_accuracy"] == 1.0, ev
    print("  PASS  end-to-end eval: broken plan -> param_acc 1.0, func_match True")

    print("\nAll self-tests passed.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("diagnose", help="Inspect parse failures in a results JSON.")
    pd.add_argument("--input", required=True, help="Path to results JSON.")
    pd.set_defaults(func=cmd_diagnose)

    pr = sub.add_parser("rescue", help="Re-parse + recompute metrics; write fixed JSON.")
    pr.add_argument("--input", required=True, help="Path to results JSON.")
    pr.add_argument("--output", required=True, help="Path to write rescued results.")
    pr.add_argument("--stats-output", default=None,
                    help="Path to write rescued stats (default: <output>.stats.json).")
    pr.set_defaults(func=cmd_rescue)

    pt = sub.add_parser("test", help="Run built-in self-tests.")
    pt.set_defaults(func=cmd_test)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()