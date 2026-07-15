#!/usr/bin/env python3
"""
reparse_alpha_umi.py — Recompute structural metrics & stats from existing
α-UMi JSON outputs using the FIXED parser (tolerant of missing "{{N}} =").

WHY THIS EXISTS:
  The original α-UMi parser required every Step line to contain
      Step N: {{N}} = tool_name(args)
  Llama-3.2-3B (and to a lesser extent other models) emits
      Step N: tool_name(args)
  without the "{{N}} =" assignment, so the parser silently dropped every
  line and forced generated_n_steps=0, exact_match=0, functional_match=0,
  param_accuracy=0, dependency_accuracy=0 — even when the tool name and
  arguments were correct.

  The fixed parser synthesizes "{{step_id}} =" from the step number when
  it's missing. Same downstream eval logic; metrics now reflect reality.

WHAT THIS SCRIPT DOES:
  - Loads an α-UMi results JSON (the file with per-example entries, NOT
    the .stats.json).
  - Re-parses every entry's `generated_plan` and `ground_truth` with the
    fixed parser.
  - Recomputes structural metrics (exact_match, functional_match,
    param_accuracy, dependency_accuracy, etc.) and writes them back into
    each entry.
  - Recomputes the aggregate stats block.
  - Writes BOTH a new results JSON (with corrected per-entry metrics) and
    a new stats JSON (the file you paste into your paper table).
  - Judge scores are NOT recomputed — they came from the judge server at
    inference time and don't depend on the parser.

USAGE:
  # Minimal — uses each entry's own ground_truth, including for artifact
  # error types (slightly more pessimistic than the original eval but
  # avoids needing a parquet path).
  python reparse_alpha_umi.py --input alpha_umi_Llama-3B_results.json

  # Recommended — matches the original eval methodology by remapping
  # artifact error types to perfect GT.
    python reparse_alpha_umi.py \
      --input ${FORTE_ROOT}/baselines_nestful/qwen7b/alpha_umi_qwen-7B-Instruct_results.json \
      --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_nestful_full/test.parquet

  # Batch mode (glob):
  for f in results/*.json; do
      python reparse_alpha_umi.py --input "$f" --test-parquet ...
  done

OUTPUT FILES (by default, alongside the input):
  <input_stem>_reparsed.json        — corrected per-example results
  <input_stem>_reparsed.stats.json  — corrected aggregate stats
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── ARTIFACT_ERROR_TYPES (matches alpha_umi_baseline.py) ─────────────────────

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


# ── parse_plan_steps — FIXED VERSION (lenient on missing {{N}} =) ───────────

def parse_plan_steps(plan_text: str) -> List[Dict]:
    """
    Parse Step-N lines from a plan into structured dicts.

    Accepts BOTH:
      - Canonical:  Step N: {{N}} = tool_name(args)
      - Lenient:    Step N: tool_name(args)     ← synthesizes {{N}} =

    Mirrors the parser in alpha_umi_baseline.py exactly.
    """
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
            if var_match:
                output_var = var_match.group(1)
                tool_match = re.search(r"=\s*([^\(]+)\((.*)\)\s*$", line)
                tool_match_empty = re.search(r"=\s*([^\(]+)\(\)\s*$", line)
            else:
                output_var = f"{{{{{step_id}}}}}"
                tool_match = re.search(
                    r"^Step \d+:\s*([a-zA-Z_]\w*)\((.*)\)\s*$", line
                )
                tool_match_empty = re.search(
                    r"^Step \d+:\s*([a-zA-Z_]\w*)\(\)\s*$", line
                )

            if not tool_match:
                if tool_match_empty:
                    tool_name = tool_match_empty.group(1).strip()
                    params: Dict[str, str] = {}
                else:
                    continue
            else:
                tool_name = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                params: Dict[str, str] = {}
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


# ── Structural eval helpers (mirrored from alpha_umi_baseline.py) ───────────

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


def evaluate_plan_vs_gt(gen_steps: List[Dict], gt_steps: List[Dict]) -> Dict:
    """
    Structural eval. Note: we drop the `tools` argument here because the input
    JSON doesn't store the per-example tools dict. NL-tool-name remapping is
    skipped — this only matters when gt_uses_nl_tool_names=True (ToolHop GT
    that uses sub-question keys as tool names). The α-UMi JSONs we've seen
    have gt_uses_nl_tool_names=False everywhere, so this is a no-op.
    """
    empty = {
        "valid": False, "error": "",
        "step_count_match": False,
        "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
        "param_accuracy": 0.0, "dependency_accuracy": 0.0,
        "exact_match": False, "functional_match": False, "param_only_match": False,
        "gt_uses_nl_tool_names": False,
        "generated_steps": 0, "ground_truth_steps": len(gt_steps),
    }
    if not gen_steps:
        return {**empty, "error": "no steps generated"}
    if not gt_steps:
        return {**empty, "error": "no ground truth steps"}

    gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)

    step_count_match = len(gen_steps) == len(gt_steps)
    correct_exact_tools = 0
    total_functional = 0.0
    total_params_correct = 0
    total_params = 0
    correct_deps = 0
    total_deps = 0

    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt = gt_steps[i] if i < len(gt_steps) else None

        if gen and gt:
            exact_ok = gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower()
            if exact_ok:
                correct_exact_tools += 1

            total_functional += _functional_tool_match(gen["tool_name"], gt["tool_name"])

            gt_keys = set(gt["parameters"].keys())
            gen_keys = set(gen["parameters"].keys())
            common = gt_keys & gen_keys
            for k in common:
                gt_v = normalize_value(gt["parameters"][k])
                gen_v = normalize_value(gen["parameters"][k])
                if gt_v == gen_v or gt_v in gen_v or gen_v in gt_v:
                    total_params_correct += 1
            total_params += len(gt_keys)

            gt_refs = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            total_deps += len(gt_refs)
            correct_deps += len(gt_refs & gen_refs)

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
    }


# ── Perfect-GT loader (optional, for artifact error type remapping) ─────────

def load_perfect_gt_from_parquet(parquet_path: str) -> Dict[int, str]:
    """Build {query_id: ground_truth_str} for error_type='none', quality>=100."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("⚠  pyarrow not installed; cannot load parquet. "
              "Falling back to entry-supplied ground_truth.", file=sys.stderr)
        return {}

    table = pq.read_table(parquet_path)
    extra_infos = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    perfect_gt: Dict[int, str] = {}
    for ei, rm in zip(extra_infos, reward_models):
        if isinstance(ei, str):
            ei = json.loads(ei)
        if isinstance(rm, str):
            rm = json.loads(rm)
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


# ── Re-parse one run (list of result entries) ───────────────────────────────

def reparse_one_run(
    results: List[Dict],
    perfect_gt_by_qid: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """
    Walk results, recompute structural metrics for each entry, return the
    modified list (in-place mutation). Judge fields are left untouched.
    """
    perfect_gt_by_qid = perfect_gt_by_qid or {}
    n_recovered = 0   # entries whose generated_n_steps went from 0 → >0

    for entry in results:
        gt_steps = parse_plan_steps(entry.get("ground_truth", ""))
        gen_steps = parse_plan_steps(entry.get("generated_plan", ""))

        # Artifact error types: use perfect GT for structural comparison
        if (entry.get("error_type") in ARTIFACT_ERROR_TYPES
                and entry.get("query_id") in perfect_gt_by_qid):
            struct_gt_steps = parse_plan_steps(perfect_gt_by_qid[entry["query_id"]])
        else:
            struct_gt_steps = gt_steps

        old_gen_n = entry.get("generated_n_steps", 0)
        struct_eval = evaluate_plan_vs_gt(gen_steps, struct_gt_steps)

        if old_gen_n == 0 and struct_eval["generated_steps"] > 0:
            n_recovered += 1

        # Update structural fields (preserve everything else)
        entry["gt_uses_nl_tool_names"] = struct_eval["gt_uses_nl_tool_names"]
        entry["exact_match"]           = struct_eval["exact_match"]
        entry["functional_match"]      = struct_eval["functional_match"]
        entry["param_only_match"]      = struct_eval["param_only_match"]
        entry["step_count_match"]      = struct_eval["step_count_match"]
        entry["generated_n_steps"]     = struct_eval["generated_steps"]
        entry["gt_n_steps"]            = struct_eval["ground_truth_steps"]
        entry["exact_tool_accuracy"]   = struct_eval["exact_tool_accuracy"]
        entry["functional_tool_accuracy"] = struct_eval["functional_tool_accuracy"]
        entry["param_accuracy"]        = struct_eval["param_accuracy"]
        entry["dependency_accuracy"]   = struct_eval["dependency_accuracy"]

    print(f"  ✓ Recovered {n_recovered}/{len(results)} entries from "
          f"generated_n_steps=0 → >0 (parser fix took effect).")
    return results


# ── Stats (mirrored from alpha_umi_baseline.py compute_stats) ───────────────

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

    error_types = sorted(set(r["error_type"] for r in results))
    per_error: Dict[str, Any] = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n": len(sub),
            "judge_success_rate":      float(np.mean([r["judge_success"]            for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"]       for r in sub])),
            "mean_judge_score":        float(np.mean([r["judge_score"]              for r in sub])),
            "functional_tool_acc":     float(np.mean([r["functional_tool_accuracy"] for r in sub])),
            "mean_param_accuracy":     float(np.mean([r["param_accuracy"]           for r in sub])),
            "exact_match_rate":        float(np.mean([r["exact_match"]              for r in sub])),
            "functional_match_rate":   float(np.mean([r["functional_match"]         for r in sub])),
            "param_only_match_rate":   float(np.mean([r["param_only_match"]         for r in sub])),
            "step_count_match_rate":   float(np.mean([r["step_count_match"]         for r in sub])),
        }

    success_dist: Dict[str, Any] = {}
    for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100 * c / n, 1)}

    stats = {
        "label": label,
        "method": "α-UMi (prompting) — reparsed",
        "alpha_umi_mode": results[0].get("alpha_umi_mode", "unknown"),
        "n_examples": n,
        "gt_uses_nl_tools": bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(full_parse_rate, 3),
        "empty_plan_rate": round(empty_plan_rate, 3),

        "accuracy": {
            "judge_success_rate": round(judge_success_rate, 3),
            "error_handled_rate": round(error_handled_rate, 3),
        },
        "judge_scores": {
            "mean":       round(float(np.mean(scores)), 2),
            "median":     round(float(np.median(scores)), 2),
            "std":        round(float(np.std(scores)), 2),
            "pct_gte_80": round(100 * sum(s >= 80 for s in scores) / n, 1),
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
        "per_error_type": per_error,
    }

    # ── Print summary ────────────────────────────────────────────────────────
    W = 70
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  N examples: {n}")
    print(f"  Empty plan rate (post-reparse): {100*empty_plan_rate:.1f}%")
    print(f"\n  ── Primary Accuracy (paper table metrics) ─────────────────────")
    print(f"  JSR (judge success ≥80) : {100*judge_success_rate:5.1f}%")
    print(f"  FM  (functional match)  : {100*functional_match_rate:5.1f}%")
    print(f"  PA  (mean param acc)    : {100*np.mean(param_accs):5.1f}%")
    print(f"  DA  (mean dep acc)      : {100*np.mean(dep_accs):5.1f}%")
    print(f"\n  ── Other metrics ──────────────────────────────────────────────")
    print(f"  Error type handled       : {100*error_handled_rate:.1f}%")
    print(f"  Exact match              : {100*exact_match_rate:.1f}%")
    print(f"  Param-only match (≥50%)  : {100*param_only_match_rate:.1f}%")
    print(f"  Step count match         : {100*step_match_rate:.1f}%")
    print(f"  Mean functional tool acc : {np.mean(func_tools):.3f}")
    print()
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Recompute α-UMi structural metrics & stats from existing JSON "
                    "outputs using the fixed parser (handles missing '{{N}} =').",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Input α-UMi results JSON (the one with per-example entries).")
    ap.add_argument("--test-parquet", default=None,
                    help="Optional: test set parquet path. If provided, artifact "
                         "error types (type_mismatch, wrong_tool, etc.) are scored "
                         "against the perfect GT for the same query_id, matching "
                         "the original eval methodology. Without this flag, the "
                         "entry's own ground_truth is used for all error types.")
    ap.add_argument("--output-results", default=None,
                    help="Output results JSON path. "
                         "Default: <input_stem>_reparsed.json (same directory).")
    ap.add_argument("--output-stats", default=None,
                    help="Output stats JSON path. "
                         "Default: <input_stem>_reparsed.stats.json (same directory).")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        ap.error(f"Input file not found: {in_path}")

    if args.output_results is None:
        args.output_results = str(in_path.with_name(f"{in_path.stem}_reparsed.json"))
    if args.output_stats is None:
        args.output_stats = str(in_path.with_name(f"{in_path.stem}_reparsed.stats.json"))

    print(f"Loading {in_path}...")
    with open(in_path) as f:
        data = json.load(f)

    if "runs" not in data:
        ap.error(f"{in_path} has no 'runs' key — is this the right file? "
                 f"(Should be the per-example results JSON, not the .stats.json.)")

    # Optional: load perfect GT for artifact error type remapping
    perfect_gt_by_qid: Dict[int, str] = {}
    if args.test_parquet:
        print(f"\nLoading perfect GT from {args.test_parquet}...")
        perfect_gt_by_qid = load_perfect_gt_from_parquet(args.test_parquet)
    else:
        print("\n⚠  No --test-parquet provided. Using entry-supplied ground_truth "
              "for artifact error types. This is slightly more pessimistic than "
              "the original eval (which remapped to perfect GT for those types).")

    # Re-parse each run
    new_results: Dict[str, Any] = {"config": data.get("config", {}), "runs": {}}
    new_stats:   Dict[str, Any] = {"config": data.get("config", {}), "runs": {}}

    for run_name, run_results in data["runs"].items():
        print(f"\nRe-parsing run '{run_name}' ({len(run_results)} entries)...")
        reparsed = reparse_one_run(run_results, perfect_gt_by_qid)
        new_results["runs"][run_name] = reparsed

        # Detect dataset from filename if possible (cosmetic only)
        dataset_hint = ""
        in_lower = str(in_path).lower()
        if "nestful" in in_lower:
            dataset_hint = " [NESTFUL]"
        elif "toolhop" in in_lower:
            dataset_hint = " [ToolHop]"

        label = f"{run_name.upper()}  α-UMi (reparsed){dataset_hint}"
        stats = compute_stats(reparsed, label)
        new_stats["runs"][run_name] = stats

    # Write outputs
    Path(args.output_results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_stats).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_results, "w") as f:
        json.dump(new_results, f, indent=2)
    print(f"\nReparsed results → {args.output_results}")

    with open(args.output_stats, "w") as f:
        json.dump(new_stats, f, indent=2)
    print(f"Reparsed stats   → {args.output_stats}")


if __name__ == "__main__":
    main()