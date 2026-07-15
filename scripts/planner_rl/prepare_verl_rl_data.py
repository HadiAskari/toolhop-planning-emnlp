#!/usr/bin/env python3
"""
Prepare data for verl RL Training (dataset-agnostic, canonical-split)

Supports both ToolHop and NESTFUL annotated datasets via --dataset.

CHANGES FROM PREVIOUS VERSION:
  1. --dataset {toolhop, nestful}: controls how raw data is loaded and joined.
  2. NESTFUL tools (list) are normalized into ToolHop's dict-keyed form so
     the prompt formatting is unchanged.
  3. Join key differs: toolhop joins on query_id; nestful joins on sample_id.
  4. Question field differs: toolhop uses 'question'; nestful uses 'input'.
  5. step_id is 0-indexed in both datasets — no per-dataset offset applied.

Usage (ToolHop, perfect plans only):
    python prepare_verl_rl_data.py \
        --dataset toolhop \
        --annotated-plans /path/to/toolhop_annotated_v1_remapped.json \
        --raw-data /path/to/ToolHop.json \
        --canonical-split /path/to/canonical_splits.json \
        --output-dir data/verl_rl_toolhop \
        --perfect-only

Usage (NESTFUL):
    python prepare_verl_rl_data.py \
        --dataset nestful \
        --annotated-plans ${FORTE_ROOT}/NESTFUL/data/nestful_annotated_combined.json \
        --raw-data ${FORTE_ROOT}/NESTFUL/data/nestful_data.jsonl \
        --canonical-split ${FORTE_ROOT}/canonical_splits_nestful.json \
        --output-dir data/verl_rl_nestful_full \
        --min-quality 0
        
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import datasets


SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)


# ─────────────────────────────────────────────────────────────────────────────
# NESTFUL tools normalizer
# ─────────────────────────────────────────────────────────────────────────────

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
    """Convert NESTFUL tools list → ToolHop-compatible dict keyed by tool name."""
    tools_dict: Dict[str, Dict] = {}
    for tool in tools_list:
        name = tool["name"]
        raw_params = tool.get("parameters", {})

        if "properties" in raw_params:
            sc2_props = raw_params.get("properties", {})
            sc2_required = raw_params.get("required", list(sc2_props.keys()))
            properties = {}
            for pn, pi in sc2_props.items():
                if isinstance(pi, dict):
                    properties[pn] = {
                        "type": _resolve_type(pi.get("type")),
                        "description": pi.get("description", ""),
                    }
                else:
                    properties[pn] = {"type": _resolve_type(pi), "description": ""}
            required_list = sc2_required
        else:
            properties = {}
            for pn, pi in raw_params.items():
                if isinstance(pi, dict):
                    properties[pn] = {
                        "type": _resolve_type(pi.get("type")),
                        "description": pi.get("description", ""),
                    }
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


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_tools_for_prompt(tools: Dict[str, Any]) -> str:
    """Format tools (dict keyed by name or sub-question) into prompt text."""
    tools_str = "Available Tools:\n"
    unique_tools: Dict[str, Any] = {}
    for key, tool_info in tools.items():
        tool_name = tool_info.get("name", key)
        if tool_name not in unique_tools:
            unique_tools[tool_name] = tool_info
    for tool_name, tool_info in unique_tools.items():
        params = tool_info.get("parameters", {})
        properties = params.get("properties", {})
        required_params = params.get("required", [])
        param_parts = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            req_marker = " (required)" if param_name in required_params else ""
            param_parts.append(f"{param_name}: {param_type}{req_marker}")
        params_str = ", ".join(param_parts)
        tools_str += f"- {tool_name}({params_str})\n"
    return tools_str


def build_user_message(question: str, tools: Dict[str, Any]) -> str:
    tools_str = format_tools_for_prompt(tools)
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


def build_chat_prompt(question: str, tools: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(question, tools)},
    ]


def format_plan_as_string(plan: Dict[str, Any]) -> str:
    """Format plan steps. step_id is 0-indexed for both datasets."""
    plan_lines = []
    for step in plan.get("steps", []):
        step_id    = step["step_id"]
        tool_name  = step["tool_name"]
        output_var = step["output_variable"]
        params     = step.get("parameters", {})
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                param_strs.append(f"{key}={value}")
            elif isinstance(value, str):
                param_strs.append(f"{key}={repr(value)}")
            else:
                param_strs.append(f"{key}={value}")
        plan_lines.append(
            f"Step {step_id}: {output_var} = {tool_name}({', '.join(param_strs)})"
        )
    return "\n".join(plan_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-aware loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_annotated_plans(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    items = data.get("data", data)
    if isinstance(items, dict):
        items = list(items.values())
    print(f"Loaded {len(items)} annotated plans from {path}")
    return items


def load_raw_data(path: str, dataset: str) -> Dict[Any, Dict[str, Any]]:
    """Load raw data → {join_key: {question, tools}} lookup."""
    if dataset == "toolhop":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        lookup = {}
        for item in data:
            qid = item.get("query_id", item.get("id"))
            if qid is None:
                continue
            lookup[qid] = {
                "question": item.get("question", item.get("query", "")),
                "tools": item.get("tools", {}),
            }
        return lookup

    elif dataset == "nestful":
        with open(path, "r") as f:
            content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            samples = data if isinstance(data, list) else list(data.values())
        except json.JSONDecodeError:
            samples = [json.loads(line) for line in content.split("\n") if line.strip()]
        lookup = {}
        for item in samples:
            sid = item.get("sample_id", item.get("id"))
            if not sid:
                continue
            lookup[sid] = {
                "question": item.get("input", ""),
                "tools": normalize_tools(item.get("tools", [])),
            }
        return lookup

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def get_annotated_join_key(item: Dict, dataset: str):
    if dataset == "toolhop":
        return item["query_id"]
    elif dataset == "nestful":
        return item.get("sample_id")
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def merge_data(annotated_items: List[Dict], raw_lookup: Dict,
               dataset: str) -> List[Dict]:
    merged, missing = [], []
    for item in annotated_items:
        qid = item["query_id"]
        join_key = get_annotated_join_key(item, dataset)
        if join_key not in raw_lookup:
            missing.append(qid)
            continue
        raw = raw_lookup[join_key]
        merged.append({
            "query_id":   qid,
            "question":   raw["question"],
            "tools":      raw["tools"],
            "plan":       item["plan"],
            "annotation": item["annotation"],
        })
    if missing:
        print(f"⚠  {len(missing)} annotated items couldn't be joined with raw data "
              f"(missing {('query_id' if dataset == 'toolhop' else 'sample_id')})")
    print(f"Merged {len(merged)} complete examples")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Canonical split + quality filter
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical_split(path: str) -> Dict[str, set]:
    with open(path, "r") as f:
        splits = json.load(f)
    print(f"Loaded canonical split from {path}")
    print(f"  seed:  {splits['split_seed']}")
    print(f"  train: {splits['n_train']} qids")
    print(f"  val:   {splits['n_val']} qids")
    print(f"  test:  {splits['n_test']} qids")
    return {
        "train": set(splits["train_qids"]),
        "val":   set(splits["val_qids"]),
        "test":  set(splits["test_qids"]),
    }


def split_by_canonical(examples: List[Dict],
                       canonical: Dict[str, set]) -> tuple:
    train, val, test, orphan = [], [], [], []
    for ex in examples:
        qid = ex["query_id"]
        if qid in canonical["train"]:
            train.append(ex)
        elif qid in canonical["val"]:
            val.append(ex)
        elif qid in canonical["test"]:
            test.append(ex)
        else:
            orphan.append(ex)

    print(f"\nApplied canonical split:")
    print(f"  train: {len(train)} examples ({len(set(e['query_id'] for e in train))} qids)")
    print(f"  val:   {len(val)} examples ({len(set(e['query_id'] for e in val))} qids)")
    print(f"  test:  {len(test)} examples ({len(set(e['query_id'] for e in test))} qids)")
    if orphan:
        orphan_qids = sorted(set(e["query_id"] for e in orphan))
        print(f"  ⚠  {len(orphan)} examples with qids NOT in canonical split: "
              f"{orphan_qids[:10]}{'...' if len(orphan_qids) > 10 else ''}")
        print(f"  These will be DROPPED.")

    return train, val, test


def filter_by_quality(examples: List[Dict], min_quality: int,
                      perfect_only: bool, label: str) -> List[Dict]:
    if perfect_only:
        filtered = [
            ex for ex in examples
            if ex["annotation"]["quality_score"] >= 100
            and ex["plan"].get("error_type", "none") == "none"
        ]
    else:
        filtered = [
            ex for ex in examples
            if ex["annotation"]["quality_score"] >= min_quality
        ]

    error_type_counts = {}
    for ex in filtered:
        et = ex["plan"].get("error_type", "none")
        error_type_counts[et] = error_type_counts.get(et, 0) + 1

    print(f"  {label}: {len(filtered)}/{len(examples)} survived "
          f"(min_quality={min_quality}, perfect_only={perfect_only})")
    if len(error_type_counts) > 1:
        print(f"    error type distribution: {error_type_counts}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# verl record construction
# ─────────────────────────────────────────────────────────────────────────────

DATA_SOURCE_TOOLHOP = "toolhop_planner"
DATA_SOURCE_NESTFUL = "nestful_planner"


def examples_to_flat_dataset(ex_list: List[Dict]) -> datasets.Dataset:
    return datasets.Dataset.from_dict({
        "query_id":   [ex["query_id"]                for ex in ex_list],
        "question":   [ex["question"]                for ex in ex_list],
        "tools":      [json.dumps(ex["tools"])       for ex in ex_list],
        "plan":       [json.dumps(ex["plan"])        for ex in ex_list],
        "annotation": [json.dumps(ex["annotation"])  for ex in ex_list],
    })


def make_map_fn(split: str, dataset: str):
    data_source = DATA_SOURCE_NESTFUL if dataset == "nestful" else DATA_SOURCE_TOOLHOP

    def process_fn(example, idx):
        question   = example["question"]
        tools      = json.loads(example["tools"])
        plan       = json.loads(example["plan"])
        annotation = json.loads(example["annotation"])

        prompt           = build_chat_prompt(question, tools)
        ground_truth_str = format_plan_as_string(plan)

        return {
            "data_source": data_source,
            "prompt":      prompt,
            "ability":     "tool_plan_generation",
            "reward_model": {
                "style":             "rule",
                "ground_truth":      ground_truth_str,
                "ground_truth_plan": json.dumps(plan),
            },
            "extra_info": {
                "index":         idx,
                "split":         split,
                "query_id":      int(example["query_id"]),
                "quality_score": int(annotation.get("quality_score", 100)),
                "num_steps":     int(len(plan.get("steps", []))),
                "error_type":    str(plan.get("error_type", "none")),
                "data_json":     json.dumps({
                    "question": question,
                    "tools":    tools,
                }),
            },
        }
    return process_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare annotated data for verl RL planner training (dataset-agnostic)"
    )
    parser.add_argument("--dataset", type=str, choices=["toolhop", "nestful"],
                        default="toolhop",
                        help="Which dataset's raw-data format to use.")
    parser.add_argument("--annotated-plans", required=True)
    parser.add_argument("--raw-data", default=None,
                        help="Raw data file. ToolHop: ToolHop.json. NESTFUL: nestful_data.jsonl.")
    parser.add_argument("--toolhop", default=None,
                        help="DEPRECATED alias for --raw-data.")
    parser.add_argument("--canonical-split", required=True,
                        help="Path to canonical_splits.json from make_canonical_split.py")
    parser.add_argument("--output-dir", default="data/verl_rl")
    parser.add_argument("--min-quality", type=int, default=0)
    parser.add_argument("--perfect-only", action="store_true",
                        help="Keep only quality_score=100 AND error_type=none plans")
    parser.add_argument("--hdfs-dir", default=None)
    args = parser.parse_args()

    raw_data_path = args.raw_data or args.toolhop
    if raw_data_path is None:
        parser.error("Must provide --raw-data (or legacy --toolhop)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PREPARING VERL RL DATA  (dataset={args.dataset})")
    print("=" * 70)

    canonical = load_canonical_split(args.canonical_split)
    annotated = load_annotated_plans(args.annotated_plans)
    raw_lookup = load_raw_data(raw_data_path, args.dataset)
    print(f"Loaded {len(raw_lookup)} raw {args.dataset} items from {raw_data_path}")
    examples = merge_data(annotated, raw_lookup, args.dataset)
    train_ex, val_ex, test_ex = split_by_canonical(examples, canonical)

    print(f"\nApplying quality filter (perfect_only={args.perfect_only}, "
          f"min_quality={args.min_quality}):")
    train_ex = filter_by_quality(train_ex, args.min_quality, args.perfect_only, "train")
    val_ex   = filter_by_quality(val_ex,   args.min_quality, args.perfect_only, "val")
    test_ex  = filter_by_quality(test_ex,  args.min_quality, args.perfect_only, "test")

    if not train_ex:
        print("\n❌ Train split is empty after filtering. Exiting.")
        return

    saved_paths = {}
    for split_name, ex_list in [("train", train_ex), ("val", val_ex), ("test", test_ex)]:
        if not ex_list:
            print(f"\nSkipping empty {split_name} split.")
            continue
        print(f"\nProcessing {split_name} ({len(ex_list)} examples)...")
        ds = examples_to_flat_dataset(ex_list)
        ds = ds.map(
            function=make_map_fn(split_name, args.dataset),
            with_indices=True,
            remove_columns=ds.column_names,
        )
        out_path = output_dir / f"{split_name}.parquet"
        ds.to_parquet(str(out_path))
        saved_paths[split_name] = out_path
        print(f"  ✓ Saved {split_name} → {out_path}  ({len(ds)} rows)")

    train_qids = sorted({ex["query_id"] for ex in train_ex})
    val_qids   = sorted({ex["query_id"] for ex in val_ex})
    test_qids  = sorted({ex["query_id"] for ex in test_ex})

    for label, qids in [("train", train_qids), ("val", val_qids), ("test", test_qids)]:
        path = output_dir / f"{label}_qids.json"
        with open(path, "w") as f:
            json.dump(qids, f, indent=2)
        print(f"  ✓ Saved {label} qid list ({len(qids)} qids) to {path}")

    extra_train = set(train_qids) - canonical["train"]
    extra_val   = set(val_qids)   - canonical["val"]
    extra_test  = set(test_qids)  - canonical["test"]
    if extra_train or extra_val or extra_test:
        print("\n❌ CONSISTENCY ERROR: RL qids do not match canonical split.")
        print(f"   Extra train: {extra_train}")
        print(f"   Extra val:   {extra_val}")
        print(f"   Extra test:  {extra_test}")
    else:
        print("\n✓ CONSISTENCY CHECK: every RL qid is in the corresponding canonical split.")

    if args.hdfs_dir:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=str(output_dir), dst=args.hdfs_dir)
            print(f"\nCopied to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("Warning: verl hdfs utils not available; skipping HDFS copy.")

    print("\n" + "=" * 70)
    print(f"DONE  ({args.dataset})")
    print("=" * 70)
    print("\nSaved files:")
    for name, path in saved_paths.items():
        print(f"  {path}")


if __name__ == "__main__":
    main()