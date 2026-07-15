"""
Fine-tune Judge Model for Plan Evaluation (dataset-agnostic)

This script fine-tunes Qwen2.5-7B-Instruct to evaluate tool execution plans.
It supports BOTH ToolHop and NESTFUL annotated datasets via --dataset.

Usage (ToolHop):
    python finetune_judge.py \
        --dataset toolhop \
        --data-path /path/to/toolhop_annotated_v1_remapped.json \
        --raw-data-path /path/to/ToolHop.json \
        --canonical-split /path/to/canonical_splits.json \
        --output-dir models/judge-toolhop

Usage (NESTFUL):
    python finetune_judge.py \
        --dataset nestful \
        --data-path /path/to/nestful_annotated_combined.json \
        --raw-data-path /path/to/nestful_data.jsonl \
        --canonical-split /path/to/canonical_splits_nestful.json \
        --output-dir models/judge-nestful

DIFFERENCES BETWEEN DATASETS:
  - ToolHop: annotated and raw both keyed by `query_id`; raw question is in `question`;
    raw tools is a dict keyed by sub-question with `name`+`parameters`.
  - NESTFUL: annotated has `query_id` AND `sample_id`; raw is JSONL keyed by `sample_id`;
    raw question is in `input`; raw tools is a LIST of {name, parameters, ...}.
    Parameters can be either MathQA-flat or StarCoder2 JSON-Schema style.

The dataset-aware loader normalizes NESTFUL tools to ToolHop's dict-keyed format
so the rest of the pipeline is dataset-agnostic.

Original three patches retained:
  1. Deterministic split via seed=42 (canonical_splits.json preferred).
  2. Prompt-token masking in labels — loss flows only on the JSON annotation tokens.
  3. train/val/test qid JSON lists saved next to test_split.json.
"""

import json
import torch
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


SPLIT_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# NESTFUL tools normalizer (lifted from prepare_nestful_for_bon.py)
# Converts NESTFUL's tools-as-list into ToolHop's tools-as-dict-keyed-by-name.
# Handles both MathQA flat-params and StarCoder2 JSON-Schema params.
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
            # StarCoder2 JSON-Schema style
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
            # MathQA flat style
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
# Dataset-aware raw-data loader
# Returns a lookup keyed appropriately for joining with annotated items.
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str, dataset: str) -> Dict[Any, Dict[str, Any]]:
    """Load raw data and return {join_key: {question, tools}} lookup.

    For ToolHop, join_key is query_id; for NESTFUL it's sample_id.
    Tools are always returned as a dict keyed by tool name."""
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
                # ToolHop tools are already dict-keyed
                "tools": item.get("tools", {}),
            }
        return lookup

    elif dataset == "nestful":
        # Support JSON array, wrapped JSON ({"data": [...]}), and JSONL
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
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'toolhop' or 'nestful'.")


def get_annotated_join_key(item: Dict, dataset: str):
    """Return the key used to join an annotated item to raw data."""
    if dataset == "toolhop":
        return item["query_id"]
    elif dataset == "nestful":
        return item.get("sample_id")
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_length: int = 2048

    use_8bit: bool = True
    gradient_checkpointing: bool = True

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatter
# ─────────────────────────────────────────────────────────────────────────────

class JudgeDataFormatter:
    """Format judge training data for instruction fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.system_prompt = """You are an expert judge for evaluating tool execution plans. Your task is to:
1. Analyze the plan's correctness and efficiency
2. Assign a quality score (0-100)
3. Predict success likelihood (yes/likely_yes/uncertain/likely_no/no)
4. Identify specific issues with severity levels
5. Provide detailed reasoning

Scoring guidelines:
- 100: Perfect execution, no errors
- 80-99: Minor issues, likely to succeed
- 60-79: Moderate issues, uncertain outcome
- 40-59: Major issues, likely to fail
- 0-39: Critical errors, will fail

Error types to check for:
- wrong_tool: Incorrect tool selection
- missing_parameter: Required parameter not provided
- type_mismatch: Parameter type doesn't match expected
- missing_dependency: Missing reference to prior step
- wrong_dependency: References wrong step output
- circular_dependency: Step depends on itself
- forward_reference: References future step
- invalid_output_variable: Invalid variable format
- hallucinated_parameter: Parameter doesn't exist in tool"""

    def format_plan_for_evaluation(
        self,
        query: str,
        tools: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> str:
        """Format plan into evaluation prompt.

        `tools` is expected to be a dict keyed by tool name (or by sub-question
        with a 'name' field, as in ToolHop). NESTFUL tools should be normalized
        to dict form via normalize_tools() before reaching this method."""
        tools_str = "Available Tools:\n"

        # Extract unique tools (ToolHop can have duplicate tool definitions)
        unique_tools: Dict[str, Any] = {}
        for key, tool_info in tools.items():
            tool_name = tool_info.get("name", key)
            if tool_name not in unique_tools:
                unique_tools[tool_name] = tool_info

        for tool_name, tool_info in unique_tools.items():
            params = tool_info.get("parameters", {})
            properties = params.get("properties", {})
            params_str = ", ".join([f"{name}: {info['type']}" for name, info in properties.items()])
            tools_str += f"- {tool_name}({params_str})\n"

        # Plan steps — step_id is 0-indexed across both datasets
        plan_str = "Plan to Evaluate:\n"
        for step in plan["steps"]:
            params = ", ".join([f"{k}={repr(v)}" for k, v in step["parameters"].items()])
            plan_str += (
                f"Step {step['step_id']}: {step['output_variable']} = {step['tool_name']}({params})\n"
            )

        prompt = f"""Query: {query}

{tools_str}

{plan_str}

Please evaluate this plan and provide:
1. Quality score (0-100)
2. Success prediction (yes/likely_yes/uncertain/likely_no/no)
3. Detailed reasoning
4. List of issues (if any)
5. Confidence (0.0-1.0)

Format your response as JSON:
{{
  "quality_score": <int>,
  "success_prediction": "<string>",
  "reasoning": "<string>",
  "issues": [
    {{
      "type": "<error_type>",
      "severity": "<critical/high/medium/low>",
      "step": <int>,
      "description": "<string>",
      "suggestion": "<string>",
      "points_deducted": <int>
    }}
  ],
  "confidence": <float>
}}"""
        return prompt

    def format_annotation(self, annotation: Dict[str, Any]) -> str:
        return json.dumps(annotation, indent=2)

    def build_prompt_and_full_text(
        self,
        query: str,
        tools: Dict[str, Any],
        plan: Dict[str, Any],
        annotation: Dict[str, Any],
    ) -> tuple:
        """Build (prompt_text, full_text) for prompt-token masking."""
        user_prompt = self.format_plan_for_evaluation(query, tools, plan)
        assistant_response = self.format_annotation(annotation)

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": assistant_response},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class JudgeDataset:
    """Dataset preparation for judge training (dataset-agnostic)."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        dataset: str = "toolhop",
        max_length: int = 2048,
        split_ratios: tuple = (0.8, 0.1, 0.1),
        canonical_split_path: str = None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.formatter = JudgeDataFormatter(tokenizer)
        self.canonical_split_path = canonical_split_path

        self.train_data, self.val_data, self.test_data = self.load_and_split(split_ratios)

    def load_and_split(self, split_ratios):
        print(f"Loading annotated data ({self.dataset}) from {self.data_path}...")
        with open(self.data_path, "r") as f:
            full_data = json.load(f)

        query_groups: Dict[int, List[Dict[str, Any]]] = {}
        for item in full_data["data"]:
            qid = item["query_id"]
            query_groups.setdefault(qid, []).append(item)

        if self.canonical_split_path:
            with open(self.canonical_split_path, "r") as f:
                splits = json.load(f)
            train_set = set(splits["train_qids"])
            val_set   = set(splits["val_qids"])
            test_set  = set(splits["test_qids"])
            print(f"  Loaded canonical split from {self.canonical_split_path}")
            print(f"  canonical train/val/test qids: "
                  f"{len(train_set)}/{len(val_set)}/{len(test_set)}")

            train_queries = [q for q in query_groups if q in train_set]
            val_queries   = [q for q in query_groups if q in val_set]
            test_queries  = [q for q in query_groups if q in test_set]
            orphan_qids   = [q for q in query_groups
                             if q not in train_set
                             and q not in val_set
                             and q not in test_set]
            if orphan_qids:
                print(f"  ⚠  {len(orphan_qids)} qids in this dataset are NOT in "
                      f"the canonical split and will be DROPPED:")
                print(f"     {sorted(orphan_qids)[:10]}"
                      f"{'...' if len(orphan_qids) > 10 else ''}")
        else:
            query_ids = sorted(query_groups.keys())
            rng = np.random.RandomState(SPLIT_SEED)
            rng.shuffle(query_ids)
            print(f"  Legacy split: SPLIT_SEED={SPLIT_SEED}, ratios={split_ratios}")
            print(f"  ⚠  Reproducible across re-runs but NOT guaranteed to match")
            print(f"     SFT/RL splits.  Pass --canonical-split for consistency.")

            n_queries = len(query_ids)
            train_end = int(n_queries * split_ratios[0])
            val_end = train_end + int(n_queries * split_ratios[1])

            train_queries = query_ids[:train_end]
            val_queries = query_ids[train_end:val_end]
            test_queries = query_ids[val_end:]

        self.train_qids = sorted(train_queries)
        self.val_qids   = sorted(val_queries)
        self.test_qids  = sorted(test_queries)

        train_items = [item for qid in train_queries for item in query_groups[qid]]
        val_items = [item for qid in val_queries for item in query_groups[qid]]
        test_items = [item for qid in test_queries for item in query_groups[qid]]

        print(f"Split: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test "
              f"({len(train_queries)}/{len(val_queries)}/{len(test_queries)} unique qids)")
        return train_items, val_items, test_items

    def prepare_dataset(self, items: List[Dict], raw_data_path: str) -> Dataset:
        """Tokenize examples. raw_data_path points to ToolHop.json or
        nestful_data.jsonl depending on self.dataset."""
        raw_lookup = load_raw_data(raw_data_path, self.dataset)
        print(f"Loaded {len(raw_lookup)} raw {self.dataset} items from {raw_data_path}")

        examples_with_boundary: List[Dict[str, str]] = []
        missing_queries = set()

        for item in tqdm(items, desc="Formatting examples"):
            qid = item["query_id"]
            join_key = get_annotated_join_key(item, self.dataset)
            if join_key not in raw_lookup:
                missing_queries.add(qid)
                continue

            raw = raw_lookup[join_key]
            prompt_text, full_text = self.formatter.build_prompt_and_full_text(
                query=raw["question"],
                tools=raw["tools"],
                plan=item["plan"],
                annotation=item["annotation"],
            )
            examples_with_boundary.append({
                "prompt_text": prompt_text,
                "full_text":   full_text,
            })

        if missing_queries:
            print(f"\nWarning: {len(missing_queries)} annotated items couldn't be joined "
                  f"with raw data (missing join keys: {self.dataset}):")
            print(f"  Missing query_ids: {sorted(list(missing_queries))[:10]}...")
            print(f"  Skipped {len(missing_queries)} examples")

        tokenizer = self.tokenizer
        max_length = self.max_length

        def tokenize_function(examples):
            n = len(examples["full_text"])

            full_enc = tokenizer(
                examples["full_text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )
            prompt_enc = tokenizer(
                examples["prompt_text"],
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )

            input_ids_list = []
            attn_list = []
            labels_list = []
            n_truncated_response = 0
            n_dropped = 0

            for i in range(n):
                ids  = full_enc["input_ids"][i]
                attn = full_enc["attention_mask"][i]
                p_len = len(prompt_enc["input_ids"][i])

                if p_len >= len(ids):
                    n_dropped += 1
                    continue
                if p_len >= max_length:
                    n_dropped += 1
                    continue
                if len(ids) == max_length and p_len < max_length:
                    n_truncated_response += 1

                labels = [-100] * p_len + list(ids[p_len:])
                assert len(labels) == len(ids)

                input_ids_list.append(ids)
                attn_list.append(attn)
                labels_list.append(labels)

            if n_dropped > 0:
                print(f"  ⚠  {n_dropped}/{n} examples dropped: prompt ≥ max_length")
            if n_truncated_response > 0:
                print(f"  ⚠  {n_truncated_response}/{n} examples had response truncated mid-way")

            return {
                "input_ids":      input_ids_list,
                "attention_mask": attn_list,
                "labels":         labels_list,
            }

        dataset = Dataset.from_list(examples_with_boundary)
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names,
            desc="Tokenizing (with prompt-token masking)",
        )
        return tokenized


# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_model_and_tokenizer(config: JudgeTrainingConfig):
    print(f"Loading model: {config.base_model}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if config.use_8bit:
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)

    if config.use_8bit:
        model = prepare_model_for_kbit_training(model)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if config.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def make_sft_data_collator(tokenizer):
    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        seq_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) < seq_len:
                lab = lab + [-100] * (seq_len - len(lab))
            else:
                lab = lab[:seq_len]
            padded_labels.append(lab)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
    return collate


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train_judge(
    config: JudgeTrainingConfig,
    data_path: str,
    raw_data_path: str,
    output_dir: str,
    dataset: str,
    canonical_split_path: str = None,
):
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    torch.manual_seed(SPLIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SPLIT_SEED)

    model, tokenizer = setup_model_and_tokenizer(config)

    dataset_manager = JudgeDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        dataset=dataset,
        max_length=config.max_length,
        canonical_split_path=canonical_split_path,
    )

    train_dataset = dataset_manager.prepare_dataset(dataset_manager.train_data, raw_data_path)
    eval_dataset  = dataset_manager.prepare_dataset(dataset_manager.val_data, raw_data_path)

    optim_choice = "adamw_torch"
    try:
        import bitsandbytes as bnb  # noqa: F401
        optim_choice = "paged_adamw_8bit"
    except Exception:
        pass

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
        group_by_length=True,
        dataloader_pin_memory=True,
        bf16_full_eval=True,
        optim=optim_choice,
        seed=SPLIT_SEED,
    )

    data_collator = make_sft_data_collator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 80)
    print(f"STARTING TRAINING  (dataset={dataset})")
    print("=" * 80)
    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    split_info = {
        "dataset":       dataset,
        "split_seed":    SPLIT_SEED,
        "n_train_qids":  len(dataset_manager.train_qids),
        "n_val_qids":    len(dataset_manager.val_qids),
        "n_test_qids":   len(dataset_manager.test_qids),
        "n_train_items": len(dataset_manager.train_data),
        "n_val_items":   len(dataset_manager.val_data),
        "n_test_items":  len(dataset_manager.test_data),
    }

    with open(f"{output_dir}/train_qids.json", "w") as f:
        json.dump(dataset_manager.train_qids, f, indent=2)
    with open(f"{output_dir}/val_qids.json", "w") as f:
        json.dump(dataset_manager.val_qids, f, indent=2)
    with open(f"{output_dir}/test_qids.json", "w") as f:
        json.dump(dataset_manager.test_qids, f, indent=2)
    with open(f"{output_dir}/test_split.json", "w") as f:
        json.dump(
            {"metadata": {"split": "test", "split_seed": SPLIT_SEED, "dataset": dataset},
             "data": dataset_manager.test_data},
            f,
            indent=2,
        )
    with open(f"{output_dir}/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    with open(f"{output_dir}/test_info.json", "w") as f:
        json.dump({
            "dataset":          dataset,
            "n_test_queries":   split_info["n_test_qids"],
            "n_test_examples":  split_info["n_test_items"],
            "test_split_saved": f"{output_dir}/test_split.json",
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE  ({dataset})")
    print("=" * 80)
    print(f"Model saved to:   {output_dir}/final")
    print(f"Test items saved: {output_dir}/test_split.json")
    print(f"Qid lists saved:  {output_dir}/train_qids.json, val_qids.json, test_qids.json")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune judge model")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["toolhop", "nestful"],
        default="toolhop",
        help="Which dataset's raw-data format to use for joining annotations with tools/questions.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to annotated plans JSON (toolhop_annotated_v1_remapped.json "
             "or nestful_annotated_combined.json)",
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Path to raw data file. For toolhop: ToolHop.json. For nestful: nestful_data.jsonl.",
    )
    # Backwards-compat alias
    parser.add_argument(
        "--toolhop-path",
        type=str,
        default=None,
        help="DEPRECATED alias for --raw-data-path (kept for backward compat).",
    )

    parser.add_argument("--output-dir", type=str, default="models/judge")

    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=2048)

    parser.add_argument("--use-8bit", action="store_true", default=False)
    parser.add_argument("--no-8bit", dest="use_8bit", action="store_false")

    parser.add_argument("--canonical-split", type=str, default=None,
                        help="Path to canonical_splits.json. REQUIRED for cross-pipeline consistency.")

    args = parser.parse_args()

    raw_data_path = args.raw_data_path or args.toolhop_path
    if raw_data_path is None:
        parser.error("Must provide --raw-data-path (or legacy --toolhop-path)")

    config = JudgeTrainingConfig(
        base_model=args.base_model,
        use_lora=args.use_lora,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_8bit=args.use_8bit,
    )

    train_judge(
        config=config,
        data_path=args.data_path,
        raw_data_path=raw_data_path,
        output_dir=args.output_dir,
        dataset=args.dataset,
        canonical_split_path=args.canonical_split,
    )


if __name__ == "__main__":
    main()