#!/usr/bin/env python3
"""
Prepare NESTFUL data for OOD Best-of-N evaluation using the ToolHop BoN pipeline.

Merges:
  1. Combined annotated NESTFUL (plans + annotations, no tools)
  2. Raw NESTFUL dataset JSONL (has tools per sample_id)

Into the parquet schema that best_of_n_selection.py expects.

Usage:
    python prepare_nestful_for_bon.py \
        --annotated nestful_annotated_combined.json \
        --raw-nestful data_v2/nestful_data.jsonl \
        --output-dir data/nestful_bon \
        --train-ratio 0.8 --val-ratio 0.1 --seed 42

    # Then run BoN eval (reuse ToolHop script as-is):
    python best_of_n_selection.py \
        --planner-model <model_path> \
        --test-parquet data/nestful_bon/test.parquet \
        --n 5 --full \
        --judge-eval-max-tokens 300 \
        --output results/nestful_ood/results.json
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

import datasets


SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)


# ── NestfulSchemaAdapter (from nestful_annotator.py) ──────────────────────────

_TYPE_MAP = {
    "int or float": "number", "int": "integer", "float": "number",
    "integer": "integer", "string": "string", "str": "string",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "dict": "object", "object": "object",
}

def _resolve_type(raw_type: Any) -> str:
    if raw_type is None:
        return "string"
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        raw_type = non_null[0] if non_null else "string"
    if not isinstance(raw_type, str):
        return "string"
    return _TYPE_MAP.get(raw_type.lower().strip(), "string")


def normalize_tools(tools_list: List[Dict]) -> Dict[str, Dict]:
    """Convert NESTFUL tools list → ToolHop-compatible dict.
    Handles both MathQA flat params and StarCoder2 JSON-Schema params."""
    tools_dict = {}
    for tool in tools_list:
        name = tool["name"]
        raw_params = tool.get("parameters", {})

        if "properties" in raw_params:
            # StarCoder2 JSON-Schema style
            sc2_props = raw_params.get("properties", {})
            sc2_required = raw_params.get("required", list(sc2_props.keys()))
            properties = {}
            for pn, pi in sc2_props.items():
                if isinstance(pi, dict):
                    properties[pn] = {"type": _resolve_type(pi.get("type")), "description": pi.get("description", "")}
                else:
                    # pi is a bare string like "int or float"
                    properties[pn] = {"type": _resolve_type(pi), "description": ""}
            required_list = sc2_required
        else:
            # MathQA flat style — pi can be a dict or a bare string
            properties = {}
            for pn, pi in raw_params.items():
                if isinstance(pi, dict):
                    properties[pn] = {"type": _resolve_type(pi.get("type")), "description": pi.get("description", "")}
                else:
                    properties[pn] = {"type": _resolve_type(pi), "description": ""}
            required_list = list(properties.keys())

        tools_dict[name] = {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": {"properties": properties, "required": required_list},
            "output_parameters": tool.get("output_parameters", {}),
        }
    return tools_dict


# ── Plan string formatting ────────────────────────────────────────────────────

def plan_dict_to_string(plan: Dict[str, Any]) -> str:
    lines = []
    for step in plan.get("steps", []):
        sid = step["step_id"]
        params = step.get("parameters", {})
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str) and "{{" in value:
                param_strs.append(f"{key}={value}")
            elif isinstance(value, str):
                param_strs.append(f"{key}={repr(value)}")
            else:
                param_strs.append(f"{key}={value}")
        lines.append(f"Step {sid + 1}: {step['output_variable']} = {step['tool_name']}({', '.join(param_strs)})")
    return "\n".join(lines)


# ── Prompt (must match ToolHop SFT/RL format exactly) ─────────────────────────

def build_user_message(question: str, tools: Dict[str, Any]) -> str:
    tools_str = "Available Tools:\n"
    for tool_name, tool_info in tools.items():
        props = tool_info.get("parameters", {}).get("properties", {})
        req = tool_info.get("parameters", {}).get("required", [])
        parts = [f"{pn}: {pi.get('type','any')}{' (required)' if pn in req else ''}"
                 for pn, pi in props.items()]
        tools_str += f"- {tool_name}({', '.join(parts)})\n"
    return (
        f"Generate a tool execution plan to answer this query.\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n"
        f"Generate a step-by-step plan using the available tools. Each step should:\n"
        f"1. Call exactly one tool\n"
        f"2. Use output variables {{{{0}}}}, {{{{1}}}}, {{{{2}}}}, etc. for results\n"
        f"3. Reference previous step outputs using {{{{N}}}} — never substitute a hardcoded value for an output that comes from a prior step\n"
        f"4. Use the exact parameter names shown in the tool signatures above\n"
        f"5. Provide all required parameters\n\n"
        f"Generate only the steps the query requires — no redundant steps, no missing steps.\n\n"
        f"Format each step as: Step N: {{{{N}}}} = tool_name(param1=value1, param2=value2, ...)"
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw_nestful(path: str) -> Dict[str, Dict]:
    """Load raw NESTFUL and index by sample_id. Supports JSON array, wrapped JSON, or JSONL."""
    with open(path) as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        samples = data if isinstance(data, list) else list(data.values())
    except json.JSONDecodeError:
        samples = [json.loads(line) for line in content.split("\n") if line.strip()]
    lookup = {}
    for s in samples:
        sid = s.get("sample_id", s.get("id", ""))
        lookup[sid] = s
    print(f"Loaded {len(lookup)} raw NESTFUL samples from {path}")
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", required=True, help="Combined annotated NESTFUL JSON")
    parser.add_argument("--raw-nestful", required=True, help="Raw NESTFUL with tools (JSONL or JSON)")
    parser.add_argument("--output-dir", default="data/nestful_bon")
    parser.add_argument("--min-quality", type=int, default=0)
    parser.add_argument("--perfect-only", action="store_true",
                        help="Keep only error_type=none AND quality>=100")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotated
    with open(args.annotated) as f:
        ann_data = json.load(f)
    items = ann_data.get("data", ann_data)
    if isinstance(items, dict):
        items = list(items.values())
    meta = ann_data.get("metadata", {})
    print(f"Loaded {len(items)} annotated plans "
          f"({meta.get('n_queries', '?')} queries, {meta.get('total_plans', '?')} plans)")

    # Load raw for tools
    raw_lookup = load_raw_nestful(args.raw_nestful)

    records = {"prompt": [], "reward_model": [], "extra_info": [],
               "data_source": [], "ability": []}
    skipped_no_tools = 0
    skipped_quality = 0

    for i, item in enumerate(items):
        sample_id = item.get("sample_id", "")
        plan = item["plan"]
        annotation = item["annotation"]
        error_type = plan.get("error_type", "none")
        quality_score = annotation.get("quality_score", 0)

        if args.perfect_only and not (error_type == "none" and quality_score >= 100):
            skipped_quality += 1
            continue
        if quality_score < args.min_quality:
            skipped_quality += 1
            continue

        raw = raw_lookup.get(sample_id)
        if not raw:
            skipped_no_tools += 1
            continue

        question = item.get("query", raw.get("input", ""))
        tools_dict = normalize_tools(raw.get("tools", []))
        gt_plan_str = plan_dict_to_string(plan)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, tools_dict)},
        ]

        records["prompt"].append(prompt)
        records["data_source"].append("nestful_ood")
        records["ability"].append("tool_plan_generation")
        records["reward_model"].append({
            "style": "rule",
            "ground_truth": gt_plan_str,
            "ground_truth_plan": json.dumps(plan),
        })
        records["extra_info"].append({
            "index": i, "split": "test",
            "query_id": int(item.get("query_id", i)),
            "quality_score": int(quality_score),
            "num_steps": len(plan.get("steps", [])),
            "error_type": str(error_type),
            "sample_id": sample_id,
            "gold_answer": json.dumps(item.get("gold_answer")),
            "data_json": json.dumps({"question": question, "tools": tools_dict}),
        })

    n = len(records["prompt"])
    if skipped_no_tools:
        print(f"Skipped {skipped_no_tools} items (sample_id not in raw NESTFUL)")
    if skipped_quality:
        print(f"Skipped {skipped_quality} items (quality/error_type filter)")
    print(f"Converted {n} examples")

    et_counts = {}
    for ei in records["extra_info"]:
        et = ei["error_type"]
        et_counts[et] = et_counts.get(et, 0) + 1
    print("Error type distribution:")
    for et, c in sorted(et_counts.items()):
        print(f"  {et:30s}: {c}")

    # ── Split by query_id to prevent leakage ──────────────────────────────
    # Group indices by query_id
    query_groups: Dict[int, List[int]] = {}
    for idx, ei in enumerate(records["extra_info"]):
        qid = ei["query_id"]
        query_groups.setdefault(qid, []).append(idx)

    qids = list(query_groups.keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(qids)

    nq = len(qids)
    t = int(nq * args.train_ratio)
    v = t + int(nq * args.val_ratio)

    split_qids = {
        "train": qids[:t],
        "val":   qids[t:v],
        "test":  qids[v:],
    }

    for split_name, sq in split_qids.items():
        indices = [i for qid in sq for i in query_groups[qid]]
        print(f"  {split_name:5s}: {len(sq):4d} queries, {len(indices):5d} examples")

    # ── Save per-split parquets ───────────────────────────────────────────
    for split_name, sq in split_qids.items():
        indices = sorted(i for qid in sq for i in query_groups[qid])
        if not indices:
            print(f"Skipping empty {split_name} split")
            continue

        split_records = {
            key: [vals[i] for i in indices]
            for key, vals in records.items()
        }
        # Update split field in extra_info
        for ei in split_records["extra_info"]:
            ei["split"] = split_name

        ds = datasets.Dataset.from_dict(split_records)
        out_path = output_dir / f"{split_name}.parquet"
        ds.to_parquet(str(out_path))
        print(f"Saved {split_name} → {out_path}  ({len(indices)} examples)")

    if n > 0:
        d = json.loads(records["extra_info"][0]["data_json"])
        print(f"\nSample 0: {d['question'][:120]}...")
        print(f"GT plan:\n{records['reward_model'][0]['ground_truth']}")
        print(f"N tools: {len(d['tools'])}")


if __name__ == "__main__":
    main()