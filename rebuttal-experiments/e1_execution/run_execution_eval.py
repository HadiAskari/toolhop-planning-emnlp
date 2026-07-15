#!/usr/bin/env python3
"""
E1 — Execution-based end-task accuracy (the rebuttal's highest-priority number).

Consumes ANY existing results JSON produced by the paper's pipelines and
executes each record's plan against real tool implementations, comparing the
final output to the benchmark's gold answer. No judge involvement anywhere.

Supported inputs (auto-detected):
  - Best-of-N results        (plan key: "best_plan")        e.g. FORTE, E3 GPT-BoN
  - baseline results          (plan key: "generated_plan")   ReAct / ToolPlanner / α-UMi / GNN4TaskPlan / GPT greedy
  Both use the {"config":..., "runs": {"full": [...], "perfect_only": [...]}} envelope.

Gold-answer sources:
  toolhop : ToolHop.json entry (by query_id)  -> entry["answer"], executable
            code in entry["functions"]
  nestful : annotated combined corpus (by query_id) -> record["gold_answer"];
            executed against vendored IBM basic_functions.py (math subset).
            Records whose GOLD plan uses tools outside the executable set are
            excluded from the denominator and reported as coverage.

Self-validation (run this FIRST on the lab server):
  --gold-check executes the gold reference plans themselves. High gold
  executability + answer-match validates the harness; the gold numbers are
  also the natural ceiling to report alongside method numbers.

Examples:
  # harness validation
  python run_execution_eval.py --dataset toolhop --gold-check
  python run_execution_eval.py --dataset nestful --gold-check

  # score a method's results file
  python run_execution_eval.py --dataset toolhop \\
      --results $ROOT/verl-integration.../best_of_n_results_qwen7b.json --run full
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import (  # noqa: E402
    find_data_file, load_annotated_corpus, load_canonical_splits,
    corpus_records_by_qid, gold_record, parse_plan_steps, mean, save_json,
)
from execution_harness import (  # noqa: E402
    run_plan_toolhop, run_plan_toolhop_grounded, run_plan_nestful,
    NestfulExtendedRegistry, nestful_gold_tools_covered,
)

PLAN_KEYS = ["best_plan", "generated_plan"]


def detect_plan_key(records):
    for key in PLAN_KEYS:
        if records and key in records[0]:
            return key
    raise KeyError(
        f"Could not find a plan field among {PLAN_KEYS} in result records "
        f"(keys present: {list(records[0].keys())[:20]})"
    )


def load_results_records(path, run):
    with open(path) as f:
        data = json.load(f)
    runs = data.get("runs", {})
    if run == "auto":
        for name in ("full", "perfect_only"):
            if name in runs and runs[name]:
                run = name
                break
    if run not in runs:
        raise KeyError(f"run '{run}' not in results file (has: {list(runs.keys())})")
    return data, runs[run], run


def aggregate(records_out, label):
    """Aggregate execution stats over per-record outputs."""
    covered = [r for r in records_out if r.get("exec_covered", True)]
    n = len(covered)
    if n == 0:
        return {"label": label, "n_records": len(records_out), "n_covered": 0}

    status_counts = Counter(r["exec_status"] for r in covered)
    ok = [r for r in covered if r["exec_status"] == "ok"]
    strict = [r for r in covered if r["exec_answer_correct"]]
    lenient = [r for r in covered if r["exec_answer_correct_lenient"]]

    agg = {
        "label": label,
        "n_records": len(records_out),
        "n_covered": n,
        "coverage": n / len(records_out),
        "execution_completion_rate": len(ok) / n,          # plan ran to the end
        "end_task_accuracy_strict": len(strict) / n,       # HEADLINE number
        "end_task_accuracy_lenient": len(lenient) / n,
        "failure_breakdown": dict(status_counts),
    }

    # cross-tabs against judge / structural fields when present
    jp = [r for r in covered if r.get("judge_success") is True]
    if jp:
        agg["judge_pass_n"] = len(jp)
        agg["exec_accuracy_given_judge_pass"] = (
            sum(1 for r in jp if r["exec_answer_correct"]) / len(jp))
    jf = [r for r in covered if r.get("judge_success") is False]
    if jf:
        agg["exec_accuracy_given_judge_fail"] = (
            sum(1 for r in jf if r["exec_answer_correct"]) / len(jf))
    fm = [r for r in covered if r.get("functional_match") is True]
    if fm:
        agg["exec_accuracy_given_fm"] = (
            sum(1 for r in fm if r["exec_answer_correct"]) / len(fm))

    # unique-query view (first record per qid)
    seen, uniq = set(), []
    for r in covered:
        if r["query_id"] not in seen:
            seen.add(r["query_id"])
            uniq.append(r)
    agg["n_unique_queries"] = len(uniq)
    agg["end_task_accuracy_strict_unique_queries"] = (
        sum(1 for r in uniq if r["exec_answer_correct"]) / len(uniq))
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["toolhop", "nestful"], required=True)
    ap.add_argument("--results", default=None,
                    help="Existing results JSON (BoN/baseline/GPT). Omit with --gold-check.")
    ap.add_argument("--run", default="auto", choices=["auto", "full", "perfect_only"])
    ap.add_argument("--gold-check", action="store_true",
                    help="Execute the gold reference plans (harness validation + ceiling)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"],
                    help="Split for --gold-check (default: test)")
    ap.add_argument("--toolhop-mode", default="grounded",
                    choices=["grounded", "real"],
                    help="ToolHop protocol: 'grounded' (default; hop-consistent "
                         "oracle sub-answers — see execution_harness docstring) or "
                         "'real' (run the shipped per-query tool code; brittle — "
                         "gold ceiling ~8%%, report the gold-check alongside)")
    ap.add_argument("--toolhop-json", default=None)
    ap.add_argument("--annotated-path", default=None,
                    help="Override path to the annotated corpus")
    ap.add_argument("--nestful-exec-dir", default=None,
                    help="Optional: IBM/NESTFUL data_v2/executable_functions clone "
                         "for coding-subset coverage")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--plan-timeout", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default=None,
                    help="Default: <results>.execution.json or e1_gold_check_<dataset>.json")
    args = ap.parse_args()

    if not args.gold_check and not args.results:
        ap.error("Provide --results, or use --gold-check.")

    extended = (NestfulExtendedRegistry(args.nestful_exec_dir)
                if args.nestful_exec_dir else None)

    # ── gold-answer sources ──────────────────────────────────────────────────
    if args.dataset == "toolhop":
        th_path = find_data_file("ToolHop.json", args.toolhop_json, args.repo_root)
        toolhop = json.load(open(th_path))
        print(f"[data] ToolHop.json: {th_path} ({len(toolhop)} entries)")

    corpus = load_annotated_corpus(args.dataset, args.annotated_path, args.repo_root)
    by_qid = corpus_records_by_qid(corpus)

    records_out = []

    if args.gold_check:
        splits = load_canonical_splits(args.dataset, repo_root=args.repo_root)
        qids = splits[f"{args.split}_qids"]
        if args.limit:
            qids = qids[: args.limit]
        print(f"[gold-check] executing gold plans for {len(qids)} {args.split} queries")

        for qid in qids:
            g = gold_record(by_qid.get(qid, []))
            if g is None:
                continue
            steps = g["plan"]["steps"]
            if args.dataset == "toolhop":
                entry = toolhop[qid]
                if args.toolhop_mode == "grounded":
                    res = run_plan_toolhop_grounded(steps, entry, args.plan_timeout)
                    covered = res.pop("covered", True)
                else:
                    res = run_plan_toolhop(steps, entry, args.plan_timeout)
                    covered = True
            else:
                covered = nestful_gold_tools_covered(steps, extended)
                res = run_plan_nestful(steps, g["gold_answer"], extended,
                                       args.plan_timeout)
            records_out.append({
                "query_id": qid,
                "exec_covered": covered,
                "exec_status": res["status"],
                "exec_answer_correct": res["answer_check"]["strict"],
                "exec_answer_correct_lenient": res["answer_check"]["lenient"],
                "exec_final_output": res["final_output_repr"],
                "exec_error": res["error"],
                "exec_failed_step": res["failed_step"],
                "gold_answer": res["gold_answer"],
                "n_steps": res["n_steps"],
            })
        mode_tag = f"_{args.toolhop_mode}" if args.dataset == "toolhop" else ""
        label = f"GOLD-{args.dataset}{mode_tag}-{args.split}"
        default_out = str(Path(__file__).parent / f"e1_gold_check_{args.dataset}{mode_tag}.json")
        config = {"mode": "gold_check", **vars(args)}

    else:
        data, records, run = load_results_records(args.results, args.run)
        plan_key = detect_plan_key(records)
        if args.limit:
            records = records[: args.limit]
        print(f"[results] {args.results} run={run} plan_key={plan_key} "
              f"({len(records)} records)")

        for i, rec in enumerate(records):
            qid = rec.get("query_id", -1)
            plan_str = rec.get(plan_key, "") or ""
            if args.dataset == "toolhop":
                if not (0 <= qid < len(toolhop)):
                    continue
                entry = toolhop[qid]
                if args.toolhop_mode == "grounded":
                    res = run_plan_toolhop_grounded(plan_str, entry, args.plan_timeout)
                    covered = res.pop("covered", True)
                else:
                    res = run_plan_toolhop(plan_str, entry, args.plan_timeout)
                    covered = True
            else:
                g = gold_record(by_qid.get(qid, []))
                if g is None:
                    continue
                covered = nestful_gold_tools_covered(g["plan"]["steps"], extended)
                res = run_plan_nestful(plan_str, g["gold_answer"], extended,
                                       args.plan_timeout)
            records_out.append({
                "query_id": qid,
                "error_type": rec.get("error_type"),
                "judge_success": rec.get("judge_success"),
                "functional_match": rec.get("functional_match"),
                "exec_covered": covered,
                "exec_status": res["status"],
                "exec_answer_correct": res["answer_check"]["strict"] and covered,
                "exec_answer_correct_lenient": res["answer_check"]["lenient"] and covered,
                "exec_final_output": res["final_output_repr"],
                "exec_error": res["error"],
                "exec_failed_step": res["failed_step"],
                "gold_answer": res["gold_answer"],
                "n_steps": res["n_steps"],
            })
            if (i + 1) % 100 == 0:
                print(f"  executed {i + 1}/{len(records)}", flush=True)
        rp = Path(args.results)
        label = f"{rp.parent.name}/{rp.stem}:{run}"
        default_out = args.results.replace(".json", "") + ".execution.json"
        config = {"mode": "results", "plan_key": plan_key, "run": run, **vars(args)}

    stats = aggregate(records_out, label)
    out_path = args.output or default_out
    save_json({"config": config, "stats": stats, "records": records_out}, out_path)

    print("\n════════ E1 EXECUTION SUMMARY ════════")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:45s} {v * 100:6.1f}%" if 0 <= v <= 1 else f"  {k}: {v}")
        else:
            print(f"  {k:45s} {v}")


if __name__ == "__main__":
    main()
