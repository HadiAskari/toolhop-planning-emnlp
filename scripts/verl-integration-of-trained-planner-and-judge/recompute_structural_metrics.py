#!/usr/bin/env python3
"""
recompute_structural_metrics.py

Re-evaluate structural metrics on already-generated Best-of-N results JSONs
without re-running inference or judge calls.

Why this exists:
  The previous version of best_of_n_selection.py had ARTIFACT_ERROR_TYPES =
  {inefficient_order, unnecessary_steps, incomplete_plan} -- only 3 of the 9
  non-`none` error types.  For the other 6 (wrong_tool, parameter_typo,
  type_mismatch, circular_dependency, forward_reference, missing_dependency)
  the structural eval was comparing the model's plan against a deliberately
  broken GT, which silently penalized the model whenever it correctly fixed
  the seeded error.  The fixed best_of_n_selection.py expands the set to all
  9 non-`none` types.  This script applies the same correction to existing
  results so SFT and GRPO numbers are directly comparable on the new
  convention without re-running the full eval.

Inputs:
  --input PATH         Existing results JSON (the full per-result output, not
                       the *.stats.json summary).  Must contain `runs.<name>`
                       lists where each result has at least `query_id`,
                       `error_type`, `ground_truth`, and `best_plan`.

Outputs:
  By default writes:
    <input_basename>.recomputed.json         (full per-result, updated)
    <input_basename>.recomputed.stats.json   (aggregated stats)

Usage:
  # Single file
  python recompute_structural_metrics.py \
      --input results/Qwen-7b-GRPO-120/results_full.json

  # Multiple files in one invocation
  python recompute_structural_metrics.py \
      --input results/Qwen-7b-GRPO-120/results_full.json \
              results/Qwen-7b-SFT-Only-New/results_full.json

  # Overwrite in place (use with care; original is backed up to .pre_recompute.json)
  python recompute_structural_metrics.py \
      --input results/Qwen-7b-GRPO-120/results_full.json \
      --in-place

What this script does NOT change:
  - judge_success, best_judge_score, best_success, best_confidence,
    best_candidate_idx/temperature, all_candidate_scores/temperatures,
    bon1_*, mean_candidate_score, candidate_score_std, error_type_handled,
    judge_agrees_with_ref, success_prediction_dist, temperature_diversity.
  Judge-derived metrics depend only on the model output + judge state, both
  of which are already in the JSON; recomputing them would just reproduce
  what's there.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import the canonical functions / constants from the eval module so this
# script and the live eval can never disagree about how a metric is computed.
try:
    from best_of_n_selection import (
        ARTIFACT_ERROR_TYPES,
        parse_plan_steps,
        evaluate_plan_vs_gt,
        compute_stats,
    )
except ImportError as e:
    sys.stderr.write(
        "ERROR: could not import from best_of_n_selection.py.\n"
        "Place this script in the same directory as best_of_n_selection.py, "
        "or set PYTHONPATH accordingly.\n"
        f"  Underlying ImportError: {e}\n"
    )
    sys.exit(2)


# ── Per-result recomputation ─────────────────────────────────────────────────

# Fields that the structural recomputation overwrites.  Listed explicitly so
# we never silently drop any other field on the result dict.
_STRUCTURAL_FIELDS = (
    "exact_match",
    "functional_match",
    "param_only_match",
    "step_count_match",
    "generated_n_steps",
    "gt_n_steps",
    "exact_tool_accuracy",
    "functional_tool_accuracy",
    "param_accuracy",
    "dependency_accuracy",
    "gt_uses_nl_tool_names",
)


def build_perfect_gt_by_qid(results: List[Dict]) -> Dict[Any, str]:
    """
    Recover the perfect-plan lookup from the results themselves.

    For the full test set, every query_id has one entry with error_type='none'
    and ref_is_perfect=True; that entry's `ground_truth` IS the perfect plan
    we want to use as the structural reference for this query_id's other
    error-type variants.

    Falls back to ref_quality_score==100 if `ref_is_perfect` is missing
    (covers older result files).
    """
    perfect: Dict[Any, str] = {}
    for r in results:
        if r.get("error_type") != "none":
            continue
        is_perfect = r.get("ref_is_perfect")
        if is_perfect is None:
            is_perfect = (r.get("ref_quality_score", 0) >= 100)
        if not is_perfect:
            continue
        qid = r.get("query_id")
        gt  = r.get("ground_truth", "")
        if qid is not None and gt and qid not in perfect:
            perfect[qid] = gt
    return perfect


def recompute_one_result(
    result: Dict,
    perfect_gt_by_qid: Dict[Any, str],
) -> Tuple[Dict, bool]:
    """
    Recompute structural fields on a single result dict using the FIXED
    ARTIFACT_ERROR_TYPES set (now all 9 non-`none` error types).

    Returns:
      (updated_result, was_remapped)
        was_remapped: True iff the structural GT used here differs from the
                      raw `ground_truth` field on the result.

    Notes:
      - We pass tools=None because the result JSON does not store the tools
        dict.  The tools-dependent code path inside evaluate_plan_vs_gt only
        fires when GT uses NL tool names AND the generated plan does not.
        For the new fixed ToolHop both sides use API names, so this path is
        inert and tools=None is safe.  If the user later evaluates a model
        that produces NL names against an NL-name GT, comparison still works
        directly (NL-vs-NL).
    """
    error_type = result.get("error_type", "none")
    query_id   = result.get("query_id")

    raw_gt = result.get("ground_truth", "")
    if (error_type in ARTIFACT_ERROR_TYPES
            and query_id in perfect_gt_by_qid):
        struct_gt_text = perfect_gt_by_qid[query_id]
    else:
        struct_gt_text = raw_gt

    was_remapped = (struct_gt_text != raw_gt)

    struct_gt_steps = parse_plan_steps(struct_gt_text)
    best_steps      = parse_plan_steps(result.get("best_plan", ""))

    eval_out = evaluate_plan_vs_gt(best_steps, struct_gt_steps, tools=None)

    updated = dict(result)
    updated["exact_match"]              = eval_out["exact_match"]
    updated["functional_match"]         = eval_out["functional_match"]
    updated["param_only_match"]         = eval_out["param_only_match"]
    updated["step_count_match"]         = eval_out["step_count_match"]
    updated["generated_n_steps"]        = eval_out.get("generated_steps", 0)
    updated["gt_n_steps"]               = eval_out.get(
        "ground_truth_steps", len(struct_gt_steps)
    )
    updated["exact_tool_accuracy"]      = eval_out["exact_tool_accuracy"]
    updated["functional_tool_accuracy"] = eval_out["functional_tool_accuracy"]
    updated["param_accuracy"]           = eval_out["param_accuracy"]
    updated["dependency_accuracy"]      = eval_out["dependency_accuracy"]
    updated["gt_uses_nl_tool_names"]    = eval_out["gt_uses_nl_tool_names"]
    updated["_structural_recomputed"]   = True
    updated["_structural_gt_remapped"]  = was_remapped

    return updated, was_remapped


# ── Diff reporting ───────────────────────────────────────────────────────────

def _get(stats: Dict, section: str, key: str, default: float = 0.0) -> float:
    try:
        return float(stats.get(section, {}).get(key, default))
    except (TypeError, ValueError):
        return default


def print_overall_diff(label: str, old_stats: Dict, new_stats: Dict) -> None:
    rows = [
        ("structural", "exact_match_rate"),
        ("structural", "functional_match_rate"),
        ("structural", "param_only_match_rate"),
        ("structural", "step_count_match_rate"),
        ("structural", "mean_functional_tool_acc"),
        ("structural", "mean_param_accuracy"),
        ("structural", "mean_dependency_accuracy"),
    ]
    print(f"\n  -- Overall Structural Diff for run '{label}' "
          f"(Before -> After) ----")
    print(f"  {'Metric':<28} {'Before':>10} {'After':>10} {'Δ':>11}")
    print(f"  {'-' * 62}")
    for section, metric in rows:
        old_v = _get(old_stats, section, metric)
        new_v = _get(new_stats, section, metric)
        delta = new_v - old_v
        print(f"  {metric:<28} {old_v:>10.3f} {new_v:>10.3f} "
              f"{delta:>+11.3f}")


def print_per_error_diff(label: str, old_stats: Dict, new_stats: Dict,
                         metric: str = "functional_match_rate") -> None:
    old_pet = old_stats.get("per_error_type", {}) or {}
    new_pet = new_stats.get("per_error_type", {}) or {}
    types = sorted(set(old_pet.keys()) | set(new_pet.keys()))
    if not types:
        return
    print(f"\n  -- Per Error-Type '{metric}' for run '{label}' "
          f"(Before -> After) ----")
    print(f"  {'Error Type':<25} {'Before':>10} {'After':>10} {'Δ':>11}")
    print(f"  {'-' * 59}")
    for et in types:
        old_v = float((old_pet.get(et) or {}).get(metric, 0.0))
        new_v = float((new_pet.get(et) or {}).get(metric, 0.0))
        delta = new_v - old_v
        marker = "  *" if et in ARTIFACT_ERROR_TYPES else "   "
        print(f"  {et:<25} {old_v:>10.3f} {new_v:>10.3f} "
              f"{delta:>+11.3f}{marker}")
    print(f"  (* = error type subject to GT remap under FIXED set)")


# ── Main per-file handler ────────────────────────────────────────────────────

def process_file(in_path: Path, out_path: Path, stats_out_path: Path,
                 in_place: bool) -> None:
    print(f"\n{'='*72}")
    print(f"  INPUT : {in_path}")
    print(f"  OUTPUT: {out_path}")
    print(f"  STATS : {stats_out_path}")
    print(f"{'='*72}")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if in_place:
        backup = in_path.with_suffix(in_path.suffix + ".pre_recompute")
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"  Backed up original to: {backup}")
        else:
            print(f"  Backup already exists, leaving as-is: {backup}")

    data = json.loads(in_path.read_text())
    if "runs" not in data:
        raise ValueError(
            f"{in_path}: expected top-level 'runs' key (this is for the full "
            "per-result JSON, not the *.stats.json summary)."
        )

    print(f"\n  Fixed ARTIFACT_ERROR_TYPES ({len(ARTIFACT_ERROR_TYPES)} types):")
    for et in sorted(ARTIFACT_ERROR_TYPES):
        print(f"    - {et}")

    new_runs:        Dict[str, List[Dict]] = {}
    new_runs_stats:  Dict[str, Dict]       = {}

    for run_name, results in data["runs"].items():
        if not isinstance(results, list):
            print(f"\n  Skipping '{run_name}': not a list of results.")
            new_runs[run_name] = results
            continue

        print(f"\n  --- Run '{run_name}' ({len(results)} results) ---")

        perfect_gt = build_perfect_gt_by_qid(results)
        n_unique_qids = len({r.get("query_id") for r in results
                             if r.get("query_id") is not None})
        print(f"  Built perfect-GT lookup for {len(perfect_gt)}/"
              f"{n_unique_qids} unique query_ids")

        # Compute "before" stats from the existing per-result fields
        old_stats = compute_stats(results, f"BEFORE recompute -- {run_name}")

        recomputed: List[Dict] = []
        n_remapped = 0
        n_changed  = 0
        for r in results:
            new_r, remapped = recompute_one_result(r, perfect_gt)
            if remapped:
                n_remapped += 1
            for f in _STRUCTURAL_FIELDS:
                if r.get(f) != new_r.get(f):
                    n_changed += 1
                    break
            recomputed.append(new_r)

        print(f"\n  Structural GT was remapped on {n_remapped}/{len(results)} "
              f"results ({100*n_remapped/max(1,len(results)):.1f}%)")
        print(f"  Structural fields changed on   {n_changed}/{len(results)} "
              f"results ({100*n_changed/max(1,len(results)):.1f}%)")

        new_stats = compute_stats(recomputed, f"AFTER  recompute -- {run_name}")

        print_overall_diff(run_name, old_stats, new_stats)
        print_per_error_diff(run_name, old_stats, new_stats,
                             metric="functional_match_rate")
        print_per_error_diff(run_name, old_stats, new_stats,
                             metric="exact_match_rate")
        print_per_error_diff(run_name, old_stats, new_stats,
                             metric="mean_param_accuracy")

        new_runs[run_name]       = recomputed
        new_runs_stats[run_name] = new_stats

    config = dict(data.get("config") or {})
    config["_structural_recomputed_with_fixed_artifact_set"] = True
    config["_structural_recomputation_artifact_error_types"] = sorted(
        ARTIFACT_ERROR_TYPES
    )
    config["_structural_recomputation_source"] = str(in_path)

    out_data       = {"config": config, "runs": new_runs}
    out_data_stats = {"config": config, "runs": new_runs_stats}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Wrote: {out_path}")

    stats_out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_out_path.write_text(json.dumps(out_data_stats, indent=2))
    print(f"  Wrote: {stats_out_path}")


def _resolve_output_paths(in_path: Path,
                          out_arg: str,
                          stats_out_arg: str,
                          in_place: bool) -> Tuple[Path, Path]:
    if in_place:
        out_path = in_path
    elif out_arg:
        out_path = Path(out_arg)
    else:
        # foo.json -> foo.recomputed.json
        out_path = in_path.with_suffix("").with_suffix(".recomputed.json") \
                   if in_path.suffix == ".json" \
                   else in_path.with_name(in_path.name + ".recomputed.json")

    if stats_out_arg:
        stats_out_path = Path(stats_out_arg)
    else:
        # foo.recomputed.json -> foo.recomputed.stats.json
        if out_path.suffix == ".json":
            stats_out_path = out_path.with_suffix(".stats.json")
        else:
            stats_out_path = out_path.with_name(out_path.name + ".stats.json")

    return out_path, stats_out_path


def main():
    parser = argparse.ArgumentParser(
        description="Recompute structural metrics on already-generated "
                    "Best-of-N results JSON files using the fixed "
                    "ARTIFACT_ERROR_TYPES set."
    )
    parser.add_argument(
        "--input", required=True, nargs="+",
        help="One or more results JSON files (the full per-result output, "
             "NOT the *.stats.json summary). Multiple files are processed "
             "sequentially with the same settings."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for the recomputed JSON. Only valid when exactly "
             "one --input file is given. Defaults to <input>.recomputed.json."
    )
    parser.add_argument(
        "--stats-output", default=None,
        help="Output path for the recomputed stats JSON. Only valid when "
             "exactly one --input file is given. Defaults to "
             "<output>.stats.json."
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite each input file. Original is backed up to "
             "<input>.pre_recompute on first run."
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]

    if (args.output or args.stats_output) and len(inputs) > 1:
        parser.error(
            "--output / --stats-output can only be used with a single --input."
        )
    if args.in_place and (args.output or args.stats_output):
        parser.error(
            "--in-place is incompatible with --output / --stats-output."
        )

    print(f"Processing {len(inputs)} file(s).")
    print(f"Fixed ARTIFACT_ERROR_TYPES: {sorted(ARTIFACT_ERROR_TYPES)}")

    failures = []
    for in_path in inputs:
        try:
            out_path, stats_out_path = _resolve_output_paths(
                in_path, args.output, args.stats_output, args.in_place
            )
            process_file(in_path, out_path, stats_out_path, args.in_place)
        except Exception as e:
            print(f"\n  FAIL  {in_path}: {type(e).__name__}: {e}")
            failures.append((in_path, e))

    if failures:
        print(f"\n{len(failures)}/{len(inputs)} file(s) failed:")
        for p, e in failures:
            print(f"  - {p}: {e}")
        sys.exit(1)

    print(f"\nDone. {len(inputs)} file(s) processed successfully.")


if __name__ == "__main__":
    main()