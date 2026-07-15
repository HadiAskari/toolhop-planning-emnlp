#!/usr/bin/env python3
"""
Create SFT Dataset for Planner Training (dataset-agnostic, canonical-split)

Supports both ToolHop and NESTFUL annotated datasets via --dataset.

CHANGES FROM PREVIOUS VERSION:
  1. --dataset {toolhop, nestful}: controls how raw data is loaded and joined.
  2. NESTFUL tools (list) are normalized into ToolHop's dict form so the prompt
     formatting code is unchanged.
  3. Join key differs: toolhop joins annotated→raw on query_id; nestful joins
     on sample_id.
  4. Question field differs: toolhop uses 'question'; nestful uses 'input'.
  5. step_id is 0-indexed in both datasets — no per-dataset offset applied.

Usage (ToolHop):
    python prepare_sft_data.py \
        --dataset toolhop \
        --annotated-plans /path/to/toolhop_annotated_v1_remapped.json \
        --raw-data /path/to/ToolHop.json \
        --canonical-split /path/to/canonical_splits.json \
        --output-dir data/planner_sft_toolhop \
        --perfect-only

Usage (NESTFUL):
    python prepare_sft_data.py \
        --dataset nestful \
        --annotated-plans ${FORTE_ROOT}/NESTFUL/data/nestful_annotated_combined.json \
        --raw-data ${FORTE_ROOT}/NESTFUL/data/nestful_data.jsonl \
        --canonical-split ${FORTE_ROOT}/canonical_splits_nestful.json \
        --output-dir data/planner_sft_nestful \
        --perfect-only
"""

import json
import argparse
import os
from typing import Dict, List, Any

from datasets import Dataset


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
# Prompt formatting (works for both datasets after tool normalization)
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


def format_plan_as_response(plan: Dict[str, Any]) -> str:
    """Format plan steps. step_id is 0-indexed for both datasets."""
    plan_lines = []
    for step in plan["steps"]:
        step_id = step["step_id"]
        tool_name = step["tool_name"]
        output_var = step["output_variable"]
        params = step.get("parameters", {})
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str):
                if value.startswith("{{") and value.endswith("}}"):
                    param_strs.append(f"{key}={value}")
                else:
                    param_strs.append(f"{key}={repr(value)}")
            elif isinstance(value, list):
                param_strs.append(f"{key}={value}")
            else:
                param_strs.append(f"{key}={value}")
        params_str = ", ".join(param_strs)
        plan_lines.append(f"Step {step_id}: {output_var} = {tool_name}({params_str})")
    return "\n".join(plan_lines)


def create_sft_example(query: str, tools: Dict, plan: Dict,
                       use_chat_template: bool = True) -> Dict[str, Any]:
    tools_str = format_tools_for_prompt(tools)
    user_message = f"""Generate a tool execution plan to answer this query.

Query: {query}

{tools_str}

Generate a step-by-step plan using the available tools. Each step should:
1. Call exactly one tool
2. Use output variables {{{{0}}}}, {{{{1}}}}, {{{{2}}}}, etc. for results
3. Reference previous step outputs using {{{{N}}}} — never substitute a hardcoded value for an output that comes from a prior step
4. Use the exact parameter names shown in the tool signatures above
5. Provide all required parameters

Generate only the steps the query requires — no redundant steps, no missing steps.

Format each step as: Step N: {{{{N}}}} = tool_name(param1=value1, param2=value2, ...)"""

    response = format_plan_as_response(plan)

    if use_chat_template:
        prompt = [
            {"role": "system",
             "content": "You are an expert at creating multi-step tool execution plans. "
                        "Given a query and available tools, generate a correct sequence of "
                        "tool calls to answer the query."},
            {"role": "user", "content": user_message},
        ]
    else:
        prompt = (f"System: You are an expert at creating multi-step tool execution plans. "
                  f"Given a query and available tools, generate a correct sequence of "
                  f"tool calls to answer the query.\n\nUser: {user_message}")

    return {"prompt": prompt, "response": response}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-aware loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str, dataset: str) -> Dict[Any, Dict[str, Any]]:
    """Load raw data → {join_key: {question, tools}} lookup.

    Join key: query_id (toolhop) or sample_id (nestful).
    Tools are normalized to a dict keyed by tool name."""
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


def load_and_merge_data(annotated_path: str, raw_data_path: str,
                        dataset: str) -> List[Dict[str, Any]]:
    print(f"Loading annotated plans ({dataset}) from {annotated_path}...")
    with open(annotated_path, "r") as f:
        annotated_data = json.load(f)
    annotated_items = annotated_data.get("data", annotated_data)
    if isinstance(annotated_items, dict):
        annotated_items = list(annotated_items.values())
    print(f"  ✓ {len(annotated_items)} annotated plans")

    print(f"Loading raw {dataset} data from {raw_data_path}...")
    raw_lookup = load_raw_data(raw_data_path, dataset)
    print(f"  ✓ {len(raw_lookup)} raw items")

    merged = []
    missing_queries = set()
    for item in annotated_items:
        qid = item["query_id"]
        join_key = get_annotated_join_key(item, dataset)
        if join_key not in raw_lookup:
            missing_queries.add(qid)
            continue
        raw = raw_lookup[join_key]
        merged.append({
            "query_id":   qid,
            "question":   raw["question"],
            "tools":      raw["tools"],
            "plan":       item["plan"],
            "annotation": item["annotation"],
        })
    if missing_queries:
        print(f"  ⚠  {len(missing_queries)} annotated items couldn't be joined "
              f"with raw data (missing {('query_id' if dataset == 'toolhop' else 'sample_id')})")
    print(f"  ✓ Merged {len(merged)} complete examples")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Canonical split
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


# ─────────────────────────────────────────────────────────────────────────────
# Quality filter
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"  {label}: {len(filtered)}/{len(examples)} survived "
          f"(min_quality={min_quality}, perfect_only={perfect_only})")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# HF Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def create_sft_dataset(examples: List[Dict],
                        use_chat_template: bool = True) -> Dataset:
    sft_examples = []
    for ex in examples:
        sft_ex = create_sft_example(
            query=ex["question"], tools=ex["tools"], plan=ex["plan"],
            use_chat_template=use_chat_template,
        )
        sft_ex["query_id"]      = ex["query_id"]
        sft_ex["quality_score"] = ex["annotation"]["quality_score"]
        sft_ex["error_type"]    = ex["plan"].get("error_type", "none")
        sft_examples.append(sft_ex)
    return Dataset.from_list(sft_examples)


def flatten_chat_prompts(dataset: Dataset) -> Dataset:
    def flatten(example):
        prompt = example["prompt"]
        if not isinstance(prompt, list):
            example["prompt_flat"] = prompt
            return example
        parts = []
        for msg in prompt:
            role = msg["role"]; content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        example["prompt_flat"] = "\n\n".join(parts)
        return example
    return dataset.map(flatten)


def save_dataset(dataset: Dataset, output_path: str):
    dataset.to_parquet(output_path)
    print(f"  ✓ Saved to {output_path}  ({len(dataset)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create SFT dataset for planner training")
    parser.add_argument("--dataset", type=str, choices=["toolhop", "nestful"],
                        default="toolhop",
                        help="Which dataset's raw-data format to use.")
    parser.add_argument("--annotated-plans", required=True,
                        help="Annotated plans JSON (toolhop_annotated_v1_remapped.json or "
                             "nestful_annotated_combined.json)")
    # Backwards-compat: --toolhop was the old name
    parser.add_argument("--raw-data", default=None,
                        help="Raw data file. ToolHop: ToolHop.json. NESTFUL: nestful_data.jsonl.")
    parser.add_argument("--toolhop", default=None,
                        help="DEPRECATED alias for --raw-data.")
    parser.add_argument("--canonical-split", required=True,
                        help="Path to canonical_splits.json from make_canonical_split.py")
    parser.add_argument("--output-dir", default="data/planner_sft")
    parser.add_argument("--min-quality", type=int, default=80)
    parser.add_argument("--perfect-only", action="store_true",
                        help="Keep only quality_score=100 AND error_type=none plans")
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument("--add-flat-prompts", action="store_true", default=True)
    args = parser.parse_args()

    raw_data_path = args.raw_data or args.toolhop
    if raw_data_path is None:
        parser.error("Must provide --raw-data (or legacy --toolhop)")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"PREPARING SFT DATA  (dataset={args.dataset})")
    print("=" * 80)

    canonical = load_canonical_split(args.canonical_split)
    examples = load_and_merge_data(args.annotated_plans, raw_data_path, args.dataset)
    train_ex, val_ex, test_ex = split_by_canonical(examples, canonical)

    print(f"\nApplying quality filter (perfect_only={args.perfect_only}, "
          f"min_quality={args.min_quality}):")
    train_ex = filter_by_quality(train_ex, args.min_quality, args.perfect_only, "train")
    val_ex   = filter_by_quality(val_ex,   args.min_quality, args.perfect_only, "val")
    test_ex  = filter_by_quality(test_ex,  args.min_quality, args.perfect_only, "test")

    if not train_ex:
        print("\n❌ Train split is empty after filtering. Exiting.")
        return

    print("\nCreating SFT datasets...")
    train_ds = create_sft_dataset(train_ex, args.use_chat_template)
    val_ds   = create_sft_dataset(val_ex,   args.use_chat_template)
    test_ds  = create_sft_dataset(test_ex,  args.use_chat_template)

    if args.add_flat_prompts:
        print("Adding flattened prompts...")
        train_ds = flatten_chat_prompts(train_ds)
        val_ds   = flatten_chat_prompts(val_ds)
        test_ds  = flatten_chat_prompts(test_ds)

    print(f"\nSaving datasets to {args.output_dir}...")
    save_dataset(train_ds, os.path.join(args.output_dir, "train_sft.parquet"))
    save_dataset(val_ds,   os.path.join(args.output_dir, "val_sft.parquet"))
    save_dataset(test_ds,  os.path.join(args.output_dir, "test_sft.parquet"))

    train_qids = sorted({ex["query_id"] for ex in train_ex})
    val_qids   = sorted({ex["query_id"] for ex in val_ex})
    test_qids  = sorted({ex["query_id"] for ex in test_ex})

    for label, qids in [("train", train_qids), ("val", val_qids), ("test", test_qids)]:
        path = os.path.join(args.output_dir, f"{label}_qids.json")
        with open(path, "w") as f:
            json.dump(qids, f, indent=2)
        print(f"  ✓ Saved {label} qid list ({len(qids)} qids) to {path}")

    extra_train = set(train_qids) - canonical["train"]
    extra_val   = set(val_qids)   - canonical["val"]
    extra_test  = set(test_qids)  - canonical["test"]
    if extra_train or extra_val or extra_test:
        print("\n❌ CONSISTENCY ERROR: SFT qids do not match canonical split.")
        print(f"   Extra train: {extra_train}")
        print(f"   Extra val:   {extra_val}")
        print(f"   Extra test:  {extra_test}")
    else:
        print("\n✓ CONSISTENCY CHECK: every SFT qid is in the corresponding canonical split.")

    print("\n" + "=" * 80)
    print(f"DATASET STATISTICS  ({args.dataset})")
    print("=" * 80)
    for ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
        if len(ds) == 0:
            continue
        scores = ds["quality_score"]
        ets    = ds["error_type"]
        print(f"\n{name}:")
        print(f"  examples:      {len(ds)}")
        print(f"  unique qids:   {len(set(ds['query_id']))}")
        print(f"  mean quality:  {sum(scores)/len(scores):.1f}")
        print(f"  error types:   {dict((e, ets.count(e)) for e in set(ets))}")


if __name__ == "__main__":
    main()