#!/usr/bin/env python3
"""
E6 — Human agreement study (answers fiFx W3b: "no human annotations exist").

Samples N plans (stratified: gold + all nine error types, drawn from the
held-out test/val splits) and writes BLINDED annotation sheets — the GPT-5.4
score/issues are withheld — plus a separate answer key.

Each of two authors independently fills their copy of the CSV (columns
`human_acceptable` yes/no and `human_error_type`), then compute_agreement.py
reports human-human and human-GPT agreement with Cohen's kappa.

NO GPU / NO API needed. Runs anywhere.

Usage:
  python sample_for_annotation.py --dataset toolhop --n 50
  python sample_for_annotation.py --dataset nestful --n 50
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import (  # noqa: E402
    load_annotated_corpus, load_canonical_splits, corpus_records_by_qid,
    format_steps_as_plan_string, gold_record, save_json,
    ANNOTATED_ERROR_ORDER,
)

INSTRUCTIONS = """\
FORTE human agreement study — annotator instructions
====================================================
You are shown a user query and a multi-step tool plan. Judge SYNTACTIC plan
correctness only (the same rubric the paper's judge uses):
 - Are all required parameters provided, tools spelled correctly, steps present?
 - Are {{N}} references valid (no forward references, no circular deps,
   no hardcoded literals where a {{N}} reference is clearly needed)?
 - Do NOT judge whether the plan semantically answers the query, and do NOT
   type-check any value containing {{N}}.

Fill in two columns:
 - human_acceptable: yes  (no errors, or only minor issues — would score >= 80/100)
                     no   (a substantive structural error — would score < 80/100)
 - human_error_type: one of
     none | type_mismatch | missing_dependency | wrong_tool | parameter_typo |
     circular_dependency | inefficient_order | incomplete_plan |
     unnecessary_steps | forward_reference | other
   (the FIRST/most severe error you see; 'none' iff acceptable with no issues)

Work independently. Do not discuss items with the other annotator until both
sheets are complete.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["toolhop", "nestful"], required=True)
    ap.add_argument("--n", type=int, default=50, help="items per dataset (default 50)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--splits", nargs="+", default=["test", "val"],
                    help="which canonical splits to draw from")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.outdir or Path(__file__).parent / "annotation_sheets")
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = load_annotated_corpus(args.dataset, repo_root=args.repo_root)
    by_qid = corpus_records_by_qid(corpus)
    splits = load_canonical_splits(args.dataset, repo_root=args.repo_root)
    qids = [q for s in args.splits for q in splits[f"{s}_qids"]]
    rng.shuffle(qids)

    # stratified: equal share per candidate slot (gold + 9 error types)
    per_type = max(1, args.n // len(ANNOTATED_ERROR_ORDER))
    chosen = []
    for et in ANNOTATED_ERROR_ORDER:
        picked = 0
        for qid in qids:
            if picked >= per_type:
                break
            recs = by_qid.get(qid, [])
            g = gold_record(recs)
            if g is None:
                continue
            gold_str = format_steps_as_plan_string(g["plan"]["steps"])
            for rec in recs:
                if rec["plan"]["error_type"] != et:
                    continue
                plan_str = format_steps_as_plan_string(rec["plan"]["steps"])
                if et != "none" and plan_str == gold_str:
                    continue  # corpus no-op negative
                if any(c["qid"] == qid and c["error_type"] == et for c in chosen):
                    continue
                chosen.append({
                    "qid": qid, "error_type": et, "query": rec["query"],
                    "plan_str": plan_str,
                    "gpt_score": rec["annotation"]["quality_score"],
                    "gpt_acceptable": "yes" if rec["annotation"]["quality_score"] >= 80 else "no",
                    "gpt_issues": [i.get("type") for i in rec["annotation"].get("issues", [])
                                   if isinstance(i, dict)],
                })
                picked += 1
                break
    rng.shuffle(chosen)
    chosen = chosen[: args.n]
    for i, c in enumerate(chosen):
        c["item_id"] = f"{args.dataset}_{i:03d}"

    # blinded sheets (identical for both annotators)
    for annot in ("A", "B"):
        sheet = outdir / f"annotation_sheet_{args.dataset}_annotator{annot}.csv"
        with open(sheet, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "query", "plan", "human_acceptable",
                        "human_error_type"])
            for c in chosen:
                w.writerow([c["item_id"], c["query"], c["plan_str"], "", ""])
        print(f"[out] {sheet}")

    key_path = outdir / f"answer_key_{args.dataset}.json"
    save_json({"config": vars(args),
               "items": [{k: c[k] for k in
                          ("item_id", "qid", "error_type", "gpt_score",
                           "gpt_acceptable", "gpt_issues")}
                         for c in chosen]}, str(key_path))

    instr = outdir / "INSTRUCTIONS.txt"
    instr.write_text(INSTRUCTIONS)
    print(f"[out] {instr}")
    counts = ", ".join(
        f"{et}:{sum(1 for c in chosen if c['error_type'] == et)}"
        for et in ANNOTATED_ERROR_ORDER)
    print(f"\nSampled {len(chosen)} items ({counts})")


if __name__ == "__main__":
    main()
