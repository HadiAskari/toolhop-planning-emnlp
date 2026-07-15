#!/usr/bin/env python3
"""
make_canonical_split.py

Generate a single canonical query-level train/val/test split for use by
ALL three pipelines (SFT, judge, RL).  This is the single source of
truth for which qids belong to which split.

Decoupling the split from each pipeline's prep script eliminates the
class of bugs where three scripts each shuffle the same qids with the
same seed but produce different orders due to subtle numpy/Python
version differences or filter-then-shuffle ordering.

Output: canonical_splits.json containing:
  {
    "split_seed":   42,
    "split_ratios": [0.8, 0.1, 0.1],
    "n_total_qids": 1000,
    "train_qids":   [0, 1, 3, 4, ...],   (sorted)
    "val_qids":     [2, 7, ...],         (sorted)
    "test_qids":    [5, 8, ...]          (sorted)
  }

Usage:
    # Generate from the annotated dataset (uses every unique qid present)
    python make_canonical_split.py \
        --annotated-plans /path/to/toolhop_all_japinder.json \
        --output canonical_splits.json

    # Or generate from the original ToolHop dataset (every qid 0..999)
    python make_canonical_split.py \
        --toolhop /path/to/ToolHop.json \
        --output canonical_splits.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def load_qids_from_annotated(path: str) -> list:
    """Pull the unique set of query_ids from the annotated dataset."""
    with open(path, "r") as f:
        data = json.load(f)
    items = data.get("data", data)
    if isinstance(items, dict):
        items = list(items.values())
    qids = sorted({item["query_id"] for item in items
                   if item.get("query_id") is not None})
    return qids


def load_qids_from_toolhop(path: str) -> list:
    """Pull the unique set of qids from the original ToolHop dataset."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    qids = sorted({item.get("query_id", item.get("id")) for item in data
                   if item.get("query_id", item.get("id")) is not None})
    return qids


def make_split(qids: list, ratios: tuple, seed: int) -> dict:
    """Shuffle qids with a fixed seed, partition by ratios, sort each
    side for deterministic display and easier downstream auditing.

    Note: we sort the input qids first so the pre-shuffle order is
    canonical — this insulates against any input-file ordering changes.
    The shuffle then produces a deterministic permutation given the
    seed."""
    qids = sorted(qids)
    rng = np.random.default_rng(seed)
    shuffled = list(qids)  # copy so we don't mutate input
    rng.shuffle(shuffled)

    n = len(shuffled)
    t_end = int(n * ratios[0])
    v_end = t_end + int(n * ratios[1])

    train = sorted(shuffled[:t_end])
    val   = sorted(shuffled[t_end:v_end])
    test  = sorted(shuffled[v_end:])

    return {
        "split_seed":    seed,
        "split_ratios":  list(ratios),
        "n_total_qids":  n,
        "n_train":       len(train),
        "n_val":         len(val),
        "n_test":        len(test),
        "train_qids":    train,
        "val_qids":      val,
        "test_qids":     test,
    }


def verify_split(split: dict):
    """Sanity checks before saving."""
    train = set(split["train_qids"])
    val   = set(split["val_qids"])
    test  = set(split["test_qids"])

    # 1. No overlap between any pair
    assert not (train & val),  f"train ∩ val = {train & val}"
    assert not (train & test), f"train ∩ test = {train & test}"
    assert not (val & test),   f"val ∩ test = {val & test}"

    # 2. Counts add up
    n_total = split["n_total_qids"]
    n_sum   = len(train) + len(val) + len(test)
    assert n_sum == n_total, f"counts don't sum: {n_sum} != {n_total}"

    # 3. No duplicates within any split
    assert len(split["train_qids"]) == len(train), "duplicates in train"
    assert len(split["val_qids"])   == len(val),   "duplicates in val"
    assert len(split["test_qids"])  == len(test),  "duplicates in test"

    print("✓ All split sanity checks pass:")
    print(f"  - No overlap between train/val/test")
    print(f"  - Counts sum correctly ({n_total})")
    print(f"  - No duplicate qids within any split")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotated-plans", default=None,
                   help="Path to annotated dataset (toolhop_all_japinder.json) — "
                        "use this if you only want qids that survived annotation")
    p.add_argument("--toolhop", default=None,
                   help="Path to original ToolHop dataset — use this if you want "
                        "to include all 1000 qids regardless of annotation status")
    p.add_argument("--output", default="canonical_splits.json")
    p.add_argument("--seed",   type=int,   default=SPLIT_SEED)
    p.add_argument("--train-ratio", type=float, default=SPLIT_RATIOS[0])
    p.add_argument("--val-ratio",   type=float, default=SPLIT_RATIOS[1])
    args = p.parse_args()

    if not args.annotated_plans and not args.toolhop:
        p.error("Must provide --annotated-plans or --toolhop (or both — they should match)")

    # Load qids
    if args.annotated_plans:
        print(f"Loading qids from annotated dataset: {args.annotated_plans}")
        qids_a = load_qids_from_annotated(args.annotated_plans)
        print(f"  {len(qids_a)} unique qids")
    else:
        qids_a = None

    if args.toolhop:
        print(f"Loading qids from original ToolHop: {args.toolhop}")
        qids_t = load_qids_from_toolhop(args.toolhop)
        print(f"  {len(qids_t)} unique qids")
    else:
        qids_t = None

    # If both provided, sanity-check overlap
    if qids_a is not None and qids_t is not None:
        only_a = set(qids_a) - set(qids_t)
        only_t = set(qids_t) - set(qids_a)
        if only_a or only_t:
            print(f"\n⚠  Disagreement between sources:")
            if only_a:
                print(f"    {len(only_a)} qids in annotated but not in ToolHop: "
                      f"{sorted(only_a)[:10]}{'...' if len(only_a) > 10 else ''}")
            if only_t:
                print(f"    {len(only_t)} qids in ToolHop but not in annotated: "
                      f"{sorted(only_t)[:10]}{'...' if len(only_t) > 10 else ''}")
            print(f"  Using annotated dataset's qid set "
                  f"(reflects what's actually trainable).")
        qids = qids_a
    else:
        qids = qids_a or qids_t

    # Generate split
    val_ratio = args.val_ratio
    train_ratio = args.train_ratio
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    if abs(test_ratio - 0.0) < 1e-6:
        p.error(f"train ({train_ratio}) + val ({val_ratio}) = 1.0, no room for test")
    if test_ratio < 0:
        p.error(f"train ({train_ratio}) + val ({val_ratio}) > 1.0")

    print(f"\nGenerating split with:")
    print(f"  seed         = {args.seed}")
    print(f"  ratios       = train {train_ratio} / val {val_ratio} / test {test_ratio:.4f}")
    print(f"  shuffle algo = numpy default_rng (independent of global state)")

    split = make_split(qids, (train_ratio, val_ratio, test_ratio), args.seed)

    print(f"\nSplit sizes:")
    print(f"  train: {split['n_train']:>4} qids")
    print(f"  val:   {split['n_val']:>4} qids")
    print(f"  test:  {split['n_test']:>4} qids")
    print(f"  total: {split['n_total_qids']:>4} qids")

    verify_split(split)

    # First/last qids of each split for quick visual sanity check
    print(f"\nFirst 10 qids per split (for visual sanity check):")
    print(f"  train: {split['train_qids'][:10]}")
    print(f"  val:   {split['val_qids'][:10]}")
    print(f"  test:  {split['test_qids'][:10]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\n✓ Wrote canonical splits to {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Pass --canonical-split {args.output} to prepare_sft_data.py")
    print(f"  2. Pass --canonical-split {args.output} to prepare_verl_rl_data.py")
    print(f"  3. Pass --canonical-split {args.output} to finetune_judge.py")
    print(f"  4. After running, verify all three pipelines produced the same qid")
    print(f"     lists by running audit_train_test_leak_v2.py — every leak% should be 0.")


if __name__ == "__main__":
    main()