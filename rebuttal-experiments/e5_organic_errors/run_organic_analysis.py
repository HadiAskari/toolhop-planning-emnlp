#!/usr/bin/env python3
"""
E5 — Judge behavior on ORGANIC (non-injected) failures (answers fiFx W2,
second half: real planning failures are messier than deterministic edits).

Organic failures = real plans emitted by baseline models (ReAct, α-UMi,
GNN4TaskPlan, ToolPlanner) that fail a judge-independent criterion. Two modes:

AGGREGATE (default, NO GPU): baseline result files already store, per record,
  the trained judge's score AND judge-independent structural metrics. We
  define organic failures as records with functional_match == False (or, with
  --use-execution, exec_answer_correct == False from an E1 .execution.json)
  and report how the frozen judge scored them vs how it scored gold plans.

RESCORE (--rescore, needs judge server): re-scores the failed plans with a
  rich judge pass (300 tokens) to also collect the judge's PREDICTED ERROR
  TYPES on organic failures — evidence the taxonomy transfers to real errors.

Usage:
  python run_organic_analysis.py --results $ROOT/baselines/react_*.json \\
      $ROOT/baselines/tool_planner_*.json --gold-scores ../e2_compound_errors/e2_compound_toolhop.json
"""

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import (  # noqa: E402
    set_judge_url, check_judge_server, batch_score_plans, mean, save_json,
)


def load_runs(path):
    with open(path) as f:
        data = json.load(f)
    for run_name in ("full", "perfect_only"):
        recs = data.get("runs", {}).get(run_name)
        if recs:
            return run_name, recs, data.get("config", {})
    return None, [], {}


def method_name(config, records, path):
    if records and records[0].get("method"):
        return records[0]["method"]
    stem = Path(path).stem
    return stem.replace("_results", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True,
                    help="baseline results JSONs (globs ok)")
    ap.add_argument("--use-execution", action="store_true",
                    help="define failure by E1 execution (needs <results>.execution.json "
                         "next to each results file) instead of functional_match")
    ap.add_argument("--gold-scores", default=None,
                    help="optional E2 output JSON; its gold judge scores are used "
                         "as the gold reference line")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score failures with the judge server (rich pass) to "
                         "collect predicted error types")
    ap.add_argument("--judge-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    paths = []
    for p in args.results:
        hits = sorted(glob.glob(p))
        paths.extend(hits if hits else [p])

    per_method = {}
    all_failed_plans = []

    for path in paths:
        run_name, records, config = load_runs(path)
        if not records:
            print(f"[skip] no records in {path}")
            continue
        m = method_name(config, records, path)
        if m in per_method:  # same method run on several backbones
            m = f"{m} [{Path(path).stem}]"

        exec_by_qid = {}
        if args.use_execution:
            exec_path = path.replace(".json", ".execution.json")
            if not Path(exec_path).exists():
                sys.exit(f"--use-execution: missing {exec_path} (run E1 first)")
            ex = json.load(open(exec_path))
            for i, r in enumerate(ex["records"]):
                exec_by_qid[(r["query_id"], i)] = r["exec_answer_correct"]

        failures, non_failures = [], []
        for i, rec in enumerate(records):
            plan = rec.get("generated_plan") or rec.get("best_plan") or ""
            if not plan.strip():
                continue  # empty plans are parse failures, not organic plans
            if args.use_execution:
                failed = not exec_by_qid.get((rec.get("query_id"), i), False)
            else:
                failed = rec.get("functional_match") is False
            (failures if failed else non_failures).append(rec)

        js = [r.get("judge_score", r.get("best_judge_score")) for r in failures]
        js = [s for s in js if s is not None]
        jn = [r.get("judge_score", r.get("best_judge_score")) for r in non_failures]
        jn = [s for s in jn if s is not None]

        per_method[m] = {
            "source": path,
            "run": run_name,
            "n_records": len(records),
            "n_organic_failures": len(failures),
            "failure_criterion": ("execution" if args.use_execution
                                  else "functional_match"),
            "judge_mean_on_failures": mean(js),
            "judge_flag_rate_on_failures": mean([1.0 if s < 80 else 0.0 for s in js]),
            "judge_mean_on_non_failures": mean(jn),
        }
        for rec in failures:
            all_failed_plans.append({"method": m, "record": rec})
        print(f"[{m}] {len(failures)} organic failures / {len(records)} records | "
              f"judge mean on failures: {mean(js):.1f} | flagged(<80): "
              f"{mean([1.0 if s < 80 else 0.0 for s in js]) * 100:.1f}%")

    gold_ref = None
    if args.gold_scores and Path(args.gold_scores).exists():
        e2 = json.load(open(args.gold_scores))
        gold_recs = [r for r in e2.get("records", []) if r.get("kind") == "gold"]
        if gold_recs:
            gold_ref = {
                "source": args.gold_scores,
                "n": len(gold_recs),
                "judge_mean_on_gold": mean([r["judge_score"] for r in gold_recs]),
            }

    # combined
    combined_js = [x for p in all_failed_plans
                   for x in [p["record"].get("judge_score",
                                             p["record"].get("best_judge_score"))]
                   if x is not None]
    combined = {
        "n_failures_total": len(combined_js),
        "judge_mean_on_failures": mean(combined_js),
        "judge_flag_rate_on_failures": mean([1.0 if s < 80 else 0.0
                                             for s in combined_js]),
        "gold_reference": gold_ref,
    }

    rescored = None
    if args.rescore:
        set_judge_url(args.judge_url)
        if not check_judge_server(args.judge_url):
            sys.exit(f"Judge server not reachable at {args.judge_url}.")
        # NOTE: baseline records don't store the tools dict; the rich pass
        # sends an empty tool list, which still yields issue types. For exact
        # tool context re-run the baselines or extend here with parquet tools.
        items = [{"query": p["record"].get("question", ""),
                  "plan_str": (p["record"].get("generated_plan")
                               or p["record"].get("best_plan") or ""),
                  "tools": {}}
                 for p in all_failed_plans]
        anns = batch_score_plans(items, max_tokens=300, workers=args.workers,
                                 desc="E5 rescore")
        issue_types = Counter()
        for ann in anns:
            for iss in (ann.get("issues") or []):
                if isinstance(iss, dict) and iss.get("type"):
                    issue_types[iss["type"]] += 1
        rescored = {
            "n": len(anns),
            "mean_score": mean([a["quality_score"] for a in anns]),
            "predicted_error_type_distribution": dict(issue_types.most_common()),
        }
        print("[rescore] predicted error types on organic failures:",
              dict(issue_types.most_common(10)))

    out = {"config": vars(args), "per_method": per_method,
           "combined": combined, "rescored": rescored}
    default_out = str(Path(__file__).parent / "e5_organic_errors.json")
    save_json(out, args.output or default_out)

    print("\n════════ E5 ORGANIC-ERROR SUMMARY ════════")
    print(f"  organic failures pooled: {combined['n_failures_total']}")
    print(f"  judge mean score on organic failures: "
          f"{combined['judge_mean_on_failures']:.1f}")
    print(f"  judge flags (<80): "
          f"{combined['judge_flag_rate_on_failures'] * 100:.1f}%")
    if gold_ref:
        print(f"  vs judge mean on gold plans: "
              f"{gold_ref['judge_mean_on_gold']:.1f}  ({args.gold_scores})")


if __name__ == "__main__":
    main()
