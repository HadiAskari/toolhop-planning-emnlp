#!/usr/bin/env python3
"""
E6 — Compute human-human and human-GPT agreement (Cohen's kappa).

Run after both annotators have filled their sheets:
  python compute_agreement.py --dataset toolhop
  python compute_agreement.py --dataset nestful
  python compute_agreement.py --dataset toolhop nestful   # pooled

Reports, on the binary acceptable/unacceptable decision (the rubric's >= 80
threshold) and on the error-type label:
  - annotator A vs annotator B (human-human)
  - each annotator vs the GPT-5.4 rubric annotation (human-model)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import cohens_kappa, mean, save_json  # noqa: E402


def load_sheet(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            acc = (row.get("human_acceptable") or "").strip().lower()
            et = (row.get("human_error_type") or "").strip().lower()
            if acc in ("yes", "no"):
                rows[row["item_id"]] = {"acceptable": acc, "error_type": et or "none"}
    return rows


def agreement(a, b, key):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    la = [a[i][key] for i in ids]
    lb = [b[i][key] for i in ids]
    return {
        "n": len(ids),
        "percent_agreement": mean([1.0 if x == y else 0.0 for x, y in zip(la, lb)]),
        "cohens_kappa": cohens_kappa(la, lb),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True,
                    choices=["toolhop", "nestful"])
    ap.add_argument("--dir", default=None,
                    help="annotation_sheets directory (default: alongside this script)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    d = Path(args.dir or Path(__file__).parent / "annotation_sheets")
    A, B, G = {}, {}, {}
    for ds in args.dataset:
        A.update(load_sheet(d / f"annotation_sheet_{ds}_annotatorA.csv"))
        B.update(load_sheet(d / f"annotation_sheet_{ds}_annotatorB.csv"))
        key = json.load(open(d / f"answer_key_{ds}.json"))
        for item in key["items"]:
            gt_types = item.get("gpt_issues") or []
            G[item["item_id"]] = {
                "acceptable": item["gpt_acceptable"],
                "error_type": (gt_types[0] if gt_types else "none"),
            }

    if not A or not B:
        sys.exit("Sheets are empty or missing filled rows — complete the "
                 "human_acceptable column first (values: yes/no).")

    stats = {
        "n_items_A": len(A), "n_items_B": len(B),
        "binary_acceptable": {
            "human_vs_human": agreement(A, B, "acceptable"),
            "humanA_vs_gpt": agreement(A, G, "acceptable"),
            "humanB_vs_gpt": agreement(B, G, "acceptable"),
        },
        "error_type": {
            "human_vs_human": agreement(A, B, "error_type"),
            "humanA_vs_gpt": agreement(A, G, "error_type"),
            "humanB_vs_gpt": agreement(B, G, "error_type"),
        },
    }
    out = args.output or str(Path(__file__).parent /
                             f"e6_agreement_{'_'.join(args.dataset)}.json")
    save_json({"config": vars(args), "stats": stats}, out)

    print("\n════════ E6 AGREEMENT SUMMARY ════════")
    for section, pairs in (("binary acceptable (>=80)", stats["binary_acceptable"]),
                           ("error type", stats["error_type"])):
        print(f"  {section}:")
        for name, s in pairs.items():
            if s:
                print(f"    {name:18s} n={s['n']:4d}  agreement="
                      f"{s['percent_agreement'] * 100:5.1f}%  kappa={s['cohens_kappa']:.3f}")


if __name__ == "__main__":
    main()
