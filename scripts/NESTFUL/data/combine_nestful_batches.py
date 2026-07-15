#!/usr/bin/env python3
"""
Combine annotated NESTFUL batch JSON files into a single file.

Usage:
    python combine_nestful_batches.py \
        --inputs nestful_0_200.json nestful_200_400.json nestful_400_600.json \
        --output nestful_annotated_combined.json
"""
import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="nestful_annotated_combined.json")
    args = parser.parse_args()

    all_data = []
    total_queries = 0
    total_skipped = 0
    skipped_ids = []

    for path in args.inputs:
        with open(path) as f:
            batch = json.load(f)
        m = batch["metadata"]
        print(f"{path}: {m['n_queries']} queries, {m['total_plans']} plans")
        
        # Reindex query_ids to be globally unique (offset by cumulative count)
        offset = total_queries
        for item in batch["data"]:
            item["query_id"] = item["query_id"] + offset
            all_data.append(item)
        
        total_queries += m["n_queries"]
        total_skipped += m.get("skipped_bad_spec", 0)
        skipped_ids.extend(m.get("skipped_sample_ids", []))

    combined = {
        "metadata": {
            "dataset": "NESTFUL",
            "n_queries": total_queries,
            "n_candidates_per_query": 10,
            "total_plans": len(all_data),
            "skipped_bad_spec": total_skipped,
            "skipped_sample_ids": skipped_ids,
            "source_files": args.inputs,
        },
        "data": all_data,
    }

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)
    
    print(f"\nCombined: {total_queries} queries, {len(all_data)} plans → {args.output}")

if __name__ == "__main__":
    main()