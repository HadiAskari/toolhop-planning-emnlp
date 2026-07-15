"""
vLLM-accelerated Judge Inference (drop-in replacement for judge_inference.py)

Why this exists:
    Original judge_inference.py uses HuggingFace `model.generate()` + 8-bit
    quantization at batch_size=1. That's ~2-3 generated tokens/sec on a 7B
    model, so a few hundred test plans take an hour+.

What changed:
    HF generate + PEFT + 8bit + bs=1
        ↓
    vLLM + LoRARequest + continuous batching (process *all* prompts in one call)

    Expected speedup: 20-50× depending on hardware.

What did NOT change:
    - System prompt
    - format_plan_for_evaluation()
    - JSON extraction / validation / fallback parsing
    - Output schema (predictions JSON is byte-identical for parseable rows)
    - Integrated evaluation step
    - CLI flags

So this is a drop-in replacement — if `judge_inference.py results.json` worked
in your pipeline, `judge_inference_vllm.py results.json` works the same way.

Requirements:
    pip install vllm>=0.6.0

Usage:
    # Inference + eval (default):
    python judge_inference_vllm.py \
        --model models/judge/final \
        --test-data test_split.json \
        --toolhop-path ToolHop.json \
        --output predictions.json

    # If your judge is a LoRA adapter (adapter_config.json exists in --model),
    # vLLM will detect it and apply the adapter automatically. You may need to
    # bump --max-lora-rank if your adapter rank is > 16.
"""

import json
import re
import sys
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm

# vLLM imports — fail fast with a clear message if not installed
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    print("ERROR: vllm not installed. `pip install vllm>=0.6.0`", file=sys.stderr)
    raise


SYSTEM_PROMPT = """You are an expert judge for evaluating tool execution plans. Your task is to:
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


class JudgeInferenceVLLM:
    """vLLM-backed judge inference. Same I/O contract as original JudgeInference."""

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        max_lora_rank: int = 64,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path

        # Detect LoRA adapter vs full model
        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            peft_cfg = json.loads(adapter_cfg.read_text())
            base_model_name = peft_cfg.get("base_model_name_or_path")
            if base_model_name is None:
                raise ValueError(
                    "adapter_config.json found but base_model_name_or_path is missing."
                )
            adapter_rank = peft_cfg.get("r", 16)
            effective_rank = max(adapter_rank, max_lora_rank)
            print(f"Detected LoRA adapter (r={adapter_rank}). Base: {base_model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.llm = LLM(
                model=base_model_name,
                enable_lora=True,
                max_lora_rank=effective_rank,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
            )
            # LoRARequest: (name, int_id, local_path). int_id must be unique per adapter.
            self.lora_request = LoRARequest("judge_adapter", 1, model_path)
        else:
            print(f"Loading full model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = LLM(
                model=model_path,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
            )
            self.lora_request = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✓ vLLM engine ready")

    # ─────────────────────────────────────────────────────────────────
    # Prompt formatting (identical to original)
    # ─────────────────────────────────────────────────────────────────
    def format_plan_for_evaluation(self, query, tools, plan):
        tools_str = "Available Tools:\n"
        unique_tools: Dict[str, Any] = {}
        for sub_question, tool_info in tools.items():
            tool_name = tool_info.get('name', sub_question)
            if tool_name not in unique_tools:
                unique_tools[tool_name] = tool_info
        for tool_name, tool_info in unique_tools.items():
            params = tool_info.get('parameters', {})
            properties = params.get('properties', {})
            params_str = ", ".join([f"{name}: {info['type']}" for name, info in properties.items()])
            tools_str += f"- {tool_name}({params_str})\n"

        plan_str = "Plan to Evaluate:\n"
        for step in plan['steps']:
            params = ", ".join([f"{k}={repr(v)}" for k, v in step['parameters'].items()])
            plan_str += f"Step {step['step_id']}: {step['output_variable']} = {step['tool_name']}({params})\n"

        return f"""Query: {query}

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

    # ─────────────────────────────────────────────────────────────────
    # JSON parsing — copied verbatim from original (battle-tested)
    # ─────────────────────────────────────────────────────────────────
    def _extract_json_object(self, text: str) -> str:
        s = text.strip()
        if "```json" in s:
            start = s.find("```json") + 7
            end = s.find("```", start)
            if end != -1:
                s = s[start:end].strip()
        elif "```" in s:
            start = s.find("```") + 3
            end = s.find("```", start)
            if end != -1:
                s = s[start:end].strip()
        if s.startswith("{") and s.endswith("}"):
            return s
        start_idx = s.find("{")
        if start_idx == -1:
            return s
        depth = 0
        for i in range(start_idx, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start_idx:i + 1]
        return s[start_idx:]

    def _validate_and_normalize_annotation(self, annotation):
        required_fields = ['quality_score', 'success_prediction', 'reasoning', 'issues', 'confidence']
        for field in required_fields:
            if field not in annotation:
                annotation[field] = self.get_default_value(field)
        try:
            annotation['quality_score'] = int(annotation['quality_score'])
        except Exception:
            annotation['quality_score'] = 50
        annotation['quality_score'] = max(0, min(100, annotation['quality_score']))
        try:
            annotation['confidence'] = float(annotation['confidence'])
        except Exception:
            annotation['confidence'] = 0.5
        annotation['confidence'] = max(0.0, min(1.0, annotation['confidence']))
        valid_predictions = {'yes', 'likely_yes', 'uncertain', 'likely_no', 'no'}
        if annotation['success_prediction'] not in valid_predictions:
            annotation['success_prediction'] = 'uncertain'
        if not isinstance(annotation.get("issues", []), list):
            annotation["issues"] = []
        if not isinstance(annotation.get("reasoning", ""), str):
            annotation["reasoning"] = str(annotation.get("reasoning", ""))
        return annotation

    def _parse_annotation_json(self, generated):
        text = generated.strip()
        try:
            candidate = self._extract_json_object(text)
            annotation = json.loads(candidate)
            return self._validate_and_normalize_annotation(annotation)
        except Exception:
            pass
        try:
            candidate = self._extract_json_object(text)
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            annotation = json.loads(candidate)
            return self._validate_and_normalize_annotation(annotation)
        except Exception:
            pass
        annotation = self.get_fallback_annotation()
        m = re.search(r'"quality_score"\s*:\s*([0-9]{1,3})', text)
        if m:
            try:
                annotation["quality_score"] = max(0, min(100, int(m.group(1))))
                annotation["confidence"] = 0.2
                annotation["reasoning"] += " (fallback: extracted quality_score)"
            except Exception:
                pass
        else:
            m2 = re.search(r'\bquality_score\b\s*:\s*([0-9]{1,3})', text)
            if m2:
                try:
                    annotation["quality_score"] = max(0, min(100, int(m2.group(1))))
                    annotation["confidence"] = 0.2
                    annotation["reasoning"] += " (fallback: extracted quality_score)"
                except Exception:
                    pass
        sp = re.search(r'"success_prediction"\s*:\s*"([^"]+)"', text)
        if sp:
            pred = sp.group(1).strip()
            if pred in {'yes', 'likely_yes', 'uncertain', 'likely_no', 'no'}:
                annotation["success_prediction"] = pred
        return annotation

    def get_default_value(self, field):
        return {'quality_score': 50, 'success_prediction': 'uncertain',
                'reasoning': 'No reasoning provided', 'issues': [], 'confidence': 0.5}.get(field)

    def get_fallback_annotation(self):
        return {'quality_score': 50, 'success_prediction': 'uncertain',
                'reasoning': 'Model failed to generate valid annotation',
                'issues': [], 'confidence': 0.0}

    # ─────────────────────────────────────────────────────────────────
    # Inference (vLLM-batched — the actual speedup)
    # ─────────────────────────────────────────────────────────────────
    def predict_batch(
        self, test_data_path, toolhop_path, output_path,
        temperature=0.1, max_new_tokens=1024,
        print_every=0, print_chars=600,
        # batch_size, max_length kept for CLI compat but unused — vLLM handles batching
        batch_size=None, max_length=None,
    ):
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        with open(toolhop_path, 'r') as f:
            toolhop = json.load(f)

        toolhop_lookup: Dict[int, Dict[str, Any]] = {}
        for item in toolhop:
            query_id = item.get('query_id', item.get('id'))
            if query_id is not None:
                toolhop_lookup[query_id] = item

        # Build all prompts upfront
        packed: List[Tuple[int, Dict[str, Any], str]] = []
        skipped = 0
        for item in test_data['data']:
            query_id = item['query_id']
            if query_id not in toolhop_lookup:
                skipped += 1
                continue
            toolhop_item = toolhop_lookup[query_id]
            user_prompt = self.format_plan_for_evaluation(
                query=toolhop_item['question'],
                tools=toolhop_item.get('tools', {}),
                plan=item['plan'],
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            packed.append((query_id, item['plan'], prompt))

        if skipped:
            print(f"Skipped {skipped} items (query_id not in ToolHop)")

        print(f"Generating {len(packed)} predictions via vLLM continuous batching...")

        # Single vLLM call for everything — this is the performance win.
        # vLLM internally schedules them with continuous batching.
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_new_tokens,
            # Stop on </s> or <|im_end|> if the tokenizer has them; vLLM handles eos automatically
        )

        prompts = [t[2] for t in packed]
        gen_kwargs = {"sampling_params": sampling_params}
        if self.lora_request is not None:
            gen_kwargs["lora_request"] = self.lora_request
        outputs = self.llm.generate(prompts, **gen_kwargs)

        # Parse results
        predictions = {'metadata': test_data.get('metadata', {}), 'data': []}
        failed_parses = 0

        for n_printed, ((qid, plan, _), output) in enumerate(zip(packed, outputs), start=1):
            generated = output.outputs[0].text
            annotation = self._parse_annotation_json(generated)
            if annotation.get('confidence', 0.0) == 0.0:
                failed_parses += 1

            if print_every > 0 and (n_printed <= 5 or (n_printed % print_every == 0)):
                print("\n" + "=" * 100)
                print(f"[DEBUG] Example #{n_printed} | query_id={qid} | "
                      f"quality_score={annotation.get('quality_score')} | "
                      f"success_prediction={annotation.get('success_prediction')} | "
                      f"confidence={annotation.get('confidence')}")
                print("[DEBUG] Annotation:")
                print(json.dumps(annotation, indent=2, ensure_ascii=False))
                print("[DEBUG] Raw generated (head):")
                print(generated[:print_chars])
                print("=" * 100 + "\n")

            predictions['data'].append({
                'query_id': qid,
                'plan': plan,
                'annotation': annotation,
            })

        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"\n✓ Predictions saved to {output_path}")
        print(f"  Total predictions: {len(predictions['data'])}")
        print(f"  Failed parses:     {failed_parses}")
        if failed_parses > 0:
            pct = 100 * failed_parses / len(predictions['data'])
            print(f"  ⚠ {failed_parses}/{len(predictions['data'])} ({pct:.1f}%) failed to parse")

        return output_path


# ─────────────────────────────────────────────────────────────────────
# Integrated evaluation (copied from original — same logic)
# ─────────────────────────────────────────────────────────────────────
def run_evaluation(predictions_path: str, ground_truth_path: str, eval_output_path: str) -> bool:
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    print(f"  predictions:   {predictions_path}")
    print(f"  ground truth:  {ground_truth_path}")
    print(f"  eval output:   {eval_output_path}\n")

    try:
        from dataset_splitter import JudgeEvaluator
    except ImportError as e:
        print(f"❌ Could not import JudgeEvaluator: {e}")
        print(f"   Manual: python dataset_splitter.py --action evaluate \\")
        print(f"           --dataset-path {ground_truth_path} \\")
        print(f"           --predictions-path {predictions_path}")
        return False

    try:
        evaluator = JudgeEvaluator()
        evaluator.evaluate(
            predictions_path=predictions_path,
            ground_truth_path=ground_truth_path,
            output_path=eval_output_path,
        )
        return True
    except Exception as e:
        print(f"\n❌ Evaluation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="vLLM-accelerated judge inference + eval")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--toolhop-path', type=str, required=True)
    parser.add_argument('--output', type=str, default='predictions.json')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max-new-tokens', type=int, default=1024)

    # vLLM-specific
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--max-lora-rank', type=int, default=64,
                        help='Set ≥ your adapter rank (check adapter_config.json "r")')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)

    # Kept for CLI back-compat (unused by vLLM — it handles batching natively)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='[unused] vLLM handles batching via continuous batching')
    parser.add_argument('--max-length', type=int, default=None,
                        help='[unused] use --max-model-len instead')

    parser.add_argument('--print-every', type=int, default=0)
    parser.add_argument('--print-chars', type=int, default=600)

    # Evaluation
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=True)
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false')
    parser.add_argument('--eval-ground-truth', type=str, default=None)
    parser.add_argument('--eval-output', type=str, default=None)

    args = parser.parse_args()

    inference = JudgeInferenceVLLM(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        max_lora_rank=args.max_lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    predictions_path = inference.predict_batch(
        test_data_path=args.test_data,
        toolhop_path=args.toolhop_path,
        output_path=args.output,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        print_every=args.print_every,
        print_chars=args.print_chars,
    )

    if args.evaluate:
        gt_path = args.eval_ground_truth or args.test_data
        eval_out = args.eval_output or str(Path(predictions_path).with_suffix(".eval.json"))
        ok = run_evaluation(predictions_path, gt_path, eval_out)
        if ok:
            print("\n✅ INFERENCE + EVALUATION COMPLETE")
            print(f"   Predictions: {predictions_path}")
            print(f"   Eval JSON:   {eval_out}")
        else:
            print(f"\n⚠ Inference saved at {predictions_path}; eval failed (see above)")
    else:
        print(f"\n✅ INFERENCE COMPLETE (eval skipped)")
        print(f"   Predictions: {predictions_path}")


if __name__ == '__main__':
    main()