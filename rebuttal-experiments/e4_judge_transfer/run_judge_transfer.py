#!/usr/bin/env python3
"""
E4 — Cross-dataset judge transfer (answers VJ8B W3: "the method is
dataset-specific; transfer to new tool domains stays unclear").

Scores the held-out TEST plans (gold + 9 stored negatives per query) of a
target dataset with whichever judge the server is currently serving, and
reports gold-vs-negative discrimination (Mann-Whitney AUC over judge scores),
detection rate, and mean scores.

Run it FOUR times (2 judges x 2 datasets); the judge is identified only by
your --judge-label, so start the right server before each run:

  # ToolHop judge scoring both datasets
  bash scripts/planner_rl/start_judge_server.sh $ROOT/judge_finetuning/models/judge/merged 7
  python run_judge_transfer.py --dataset toolhop --judge-label judge-toolhop
  python run_judge_transfer.py --dataset nestful --judge-label judge-toolhop
  # swap server to the NESTFUL judge
  bash scripts/planner_rl/start_judge_server.sh $ROOT/judge_finetuning/models/judge-nestful/merged 7
  python run_judge_transfer.py --dataset nestful --judge-label judge-nestful
  python run_judge_transfer.py --dataset toolhop --judge-label judge-nestful

aggregate_results.py then pairs in-domain vs cross-domain AUCs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import (  # noqa: E402
    find_data_file, import_module_from_file, load_annotated_corpus,
    load_canonical_splits, corpus_records_by_qid, gold_record,
    format_steps_as_plan_string, set_judge_url, check_judge_server,
    batch_score_plans, mann_whitney_auc, mean, save_json,
)


def get_tools(dataset, qid, toolhop_data, nestful_by_sample, rec, mod):
    if dataset == "toolhop":
        return mod.reindex_tools_by_api_name(toolhop_data[qid]["tools"])
    sample = nestful_by_sample.get(rec.get("sample_id"))
    return (mod.NestfulSchemaAdapter.normalize_tools(sample["tools"])
            if sample else {})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["toolhop", "nestful"], required=True,
                    help="dataset whose test plans get scored")
    ap.add_argument("--judge-label", required=True,
                    help="which judge the server is running, e.g. judge-toolhop "
                         "or judge-nestful (recorded in the output)")
    ap.add_argument("--judge-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--judge-max-tokens", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    set_judge_url(args.judge_url)
    if not check_judge_server(args.judge_url):
        sys.exit(f"Judge server not reachable at {args.judge_url}.")

    # generator module only for its tool-normalization helpers
    if args.dataset == "toolhop":
        mod = import_module_from_file(
            find_data_file("toolhop_plan_generator", repo_root=args.repo_root),
            "forte_toolhop_generator")
        toolhop_data = json.load(open(find_data_file("ToolHop.json",
                                                     repo_root=args.repo_root)))
        nestful_by_sample = {}
    else:
        mod = import_module_from_file(
            find_data_file("nestful_annotator", repo_root=args.repo_root),
            "forte_nestful_annotator")
        toolhop_data = None
        nestful_by_sample = {}
        with open(find_data_file("nestful_data", repo_root=args.repo_root)) as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    nestful_by_sample[s["sample_id"]] = s

    corpus = load_annotated_corpus(args.dataset, repo_root=args.repo_root)
    by_qid = corpus_records_by_qid(corpus)
    splits = load_canonical_splits(args.dataset, repo_root=args.repo_root)
    qids = splits["test_qids"]
    if args.limit:
        qids = qids[: args.limit]

    items, meta = [], []
    for qid in qids:
        recs = by_qid.get(qid, [])
        g = gold_record(recs)
        if g is None:
            continue
        tools = get_tools(args.dataset, qid, toolhop_data, nestful_by_sample, g, mod)
        gold_str = format_steps_as_plan_string(g["plan"]["steps"])
        for rec in recs:
            et = rec["plan"]["error_type"]
            plan_str = format_steps_as_plan_string(rec["plan"]["steps"])
            if et != "none" and plan_str == gold_str:
                continue  # corpus no-op negative; skip
            items.append({"query": rec["query"], "plan_str": plan_str, "tools": tools})
            meta.append((qid, et, rec["annotation"]["quality_score"]))

    n_gold = sum(1 for m in meta if m[1] == "none")
    print(f"[pool] {len(items)} plans over {len(qids)} test queries "
          f"({n_gold} gold, {len(items) - n_gold} negatives)")

    annotations = batch_score_plans(items, max_tokens=args.judge_max_tokens,
                                    workers=args.workers,
                                    desc=f"E4 {args.judge_label}->{args.dataset}")

    records = []
    for (qid, et, gpt_score), it, ann in zip(meta, items, annotations):
        records.append({"query_id": qid, "error_type": et,
                        "gpt_annotation_score": gpt_score,
                        "judge_score": ann["quality_score"]})

    gold_scores = [r["judge_score"] for r in records if r["error_type"] == "none"]
    neg_scores = [r["judge_score"] for r in records if r["error_type"] != "none"]

    per_error = {}
    for et in sorted({r["error_type"] for r in records} - {"none"}):
        sub = [r["judge_score"] for r in records if r["error_type"] == et]
        per_error[et] = {
            "n": len(sub), "mean_score": mean(sub),
            "detection_rate": mean([1.0 if s < 80 else 0.0 for s in sub]),
            "auc_vs_gold": mann_whitney_auc(gold_scores, sub),
        }

    stats = {
        "judge_label": args.judge_label,
        "dataset": args.dataset,
        "n_gold": len(gold_scores),
        "n_negatives": len(neg_scores),
        "gold_mean_score": mean(gold_scores),
        "negative_mean_score": mean(neg_scores),
        "separation": mean(gold_scores) - mean(neg_scores),
        "auc_gold_vs_negative": mann_whitney_auc(gold_scores, neg_scores),
        "negative_detection_rate": mean([1.0 if s < 80 else 0.0 for s in neg_scores]),
        "gold_pass_rate": mean([1.0 if s >= 80 else 0.0 for s in gold_scores]),
        "per_error_type": per_error,
    }

    default_out = str(Path(__file__).parent /
                      f"e4_transfer_{args.judge_label}_on_{args.dataset}.json")
    save_json({"config": vars(args), "stats": stats, "records": records},
              args.output or default_out)

    print("\n════════ E4 JUDGE TRANSFER SUMMARY ════════")
    print(f"  judge={args.judge_label}  scored dataset={args.dataset}")
    print(f"  AUC (gold vs negative):  {stats['auc_gold_vs_negative']:.3f}")
    print(f"  gold mean {stats['gold_mean_score']:.1f} | negative mean "
          f"{stats['negative_mean_score']:.1f} | separation {stats['separation']:.1f}")
    print(f"  negative detection (<80): {stats['negative_detection_rate'] * 100:.1f}% | "
          f"gold pass (>=80): {stats['gold_pass_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
