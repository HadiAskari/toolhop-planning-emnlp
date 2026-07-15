"""
Judge Model Inference for Fine-tuned Qwen2.5  (with integrated evaluation)

This script runs inference with your fine-tuned judge model and outputs
predictions in the format expected by dataset_splitter.py evaluation.

CHANGES vs. previous version:
  - After predictions are written, automatically runs JudgeEvaluator on them
    against the same --test-data file (which already contains the reference
    annotations). Pass --no-evaluate to skip.
  - --eval-output flag for where the eval results JSON goes
    (default: <output>.eval.json next to the predictions file).
  - --eval-ground-truth flag if you want to evaluate against a different
    reference file than the one used for inference. Defaults to --test-data.
  - Eval runs in the same process (no extra model load), so it's free —
    just JudgeEvaluator parsing the predictions JSON we just wrote.

Usage:
    # Inference + eval in one command (default):
    python judge_inference.py \\
        --model models/judge/final \\
        --test-data test_split.json \\
        --toolhop-path ToolHop.json \\
        --output predictions.json

    # Inference only (skip eval):
    python judge_inference.py ... --no-evaluate

    # Evaluate against a different ground truth file:
    python judge_inference.py ... --eval-ground-truth other_split.json
"""

import json
import re
import sys
import torch
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Optional: new HF quantization API (we keep a fallback if unavailable)
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


class JudgeInference:
    """Inference wrapper for fine-tuned judge model"""

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_8bit: bool = True
    ):
        self.device = device
        self.model_path = model_path
        self.use_8bit = use_8bit

        print(f"Loading judge model from {model_path}...")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decoder-only + batching => left padding for correct generation
        self.tokenizer.padding_side = "left"

        compute_dtype = torch.float16 if use_8bit else torch.bfloat16

        model_kwargs: Dict[str, Any] = {"device_map": "auto"}
        model_kwargs_dtype_new = dict(model_kwargs); model_kwargs_dtype_new["dtype"] = compute_dtype
        model_kwargs_dtype_old = dict(model_kwargs); model_kwargs_dtype_old["torch_dtype"] = compute_dtype

        quantization_config = None
        if use_8bit and BitsAndBytesConfig is not None:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        adapter_cfg_path = Path(model_path) / "adapter_config.json"
        if adapter_cfg_path.exists():
            peft_cfg = json.loads(adapter_cfg_path.read_text())
            base_model_name = peft_cfg.get("base_model_name_or_path")
            if base_model_name is None:
                raise ValueError(
                    "adapter_config.json found but base_model_name_or_path is missing. "
                    "Cannot load PEFT adapter safely."
                )

            base_model = self._load_model_compatible(
                base_model_name, model_kwargs_dtype_new, model_kwargs_dtype_old,
                quantization_config=quantization_config, use_8bit=use_8bit,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = self._load_model_compatible(
                model_path, model_kwargs_dtype_new, model_kwargs_dtype_old,
                quantization_config=quantization_config, use_8bit=use_8bit,
            )

        self.model.eval()
        try:
            self.model.config.use_cache = True
        except Exception:
            pass

        print("✓ Model loaded successfully")

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

    def _load_model_compatible(
        self, path, kwargs_new_dtype, kwargs_old_dtype,
        quantization_config=None, use_8bit=True,
    ):
        if quantization_config is not None:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    path, **kwargs_new_dtype, quantization_config=quantization_config,
                )
            except TypeError:
                pass
            try:
                return AutoModelForCausalLM.from_pretrained(
                    path, **kwargs_old_dtype, quantization_config=quantization_config,
                )
            except TypeError:
                pass
        if use_8bit:
            try:
                return AutoModelForCausalLM.from_pretrained(path, **kwargs_new_dtype, load_in_8bit=True)
            except TypeError:
                return AutoModelForCausalLM.from_pretrained(path, **kwargs_old_dtype, load_in_8bit=True)
        try:
            return AutoModelForCausalLM.from_pretrained(path, **kwargs_new_dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(path, **kwargs_old_dtype)

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

    @staticmethod
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def predict_batch(
        self, test_data_path, toolhop_path, output_path,
        temperature=0.1, batch_size=1, max_new_tokens=1024, max_length=2048,
        print_every=1, print_chars=600,
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

        print(f"Running inference on {len(test_data['data'])} plans...")

        predictions = {'metadata': test_data.get('metadata', {}), 'data': []}
        failed_parses = 0
        n_printed = 0

        packed: List[Tuple[int, Dict[str, Any], str]] = []
        for item in test_data['data']:
            query_id = item['query_id']
            if query_id not in toolhop_lookup:
                print(f"Warning: query_id {query_id} not found in ToolHop")
                continue
            toolhop_item = toolhop_lookup[query_id]
            user_prompt = self.format_plan_for_evaluation(
                query=toolhop_item['question'],
                tools=toolhop_item.get('tools', {}),
                plan=item['plan'],
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            packed.append((query_id, item['plan'], prompt))

        for chunk in tqdm(list(self._chunks(packed, max(1, batch_size))), desc="Generating predictions"):
            qids = [x[0] for x in chunk]
            plans = [x[1] for x in chunk]
            prompts = [x[2] for x in chunk]

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            for i in range(len(chunk)):
                gen_ids = outputs[i][prompt_len:]
                generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                annotation = self._parse_annotation_json(generated)
                if annotation.get('confidence', 0.0) == 0.0:
                    failed_parses += 1

                n_printed += 1
                if print_every > 0 and (n_printed <= 5 or (n_printed % print_every == 0)):
                    print("\n" + "=" * 100)
                    print(f"[DEBUG] Example #{n_printed} | query_id={qids[i]} | "
                          f"quality_score={annotation.get('quality_score')} | "
                          f"success_prediction={annotation.get('success_prediction')} | "
                          f"confidence={annotation.get('confidence')}")
                    print("[DEBUG] Annotation:")
                    print(json.dumps(annotation, indent=2, ensure_ascii=False))
                    print("[DEBUG] Raw generated (head):")
                    print(generated[:print_chars])
                    print("=" * 100 + "\n")

                predictions['data'].append({
                    'query_id': qids[i],
                    'plan': plans[i],
                    'annotation': annotation,
                })

        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"\n✓ Predictions saved to {output_path}")
        print(f"  Total predictions: {len(predictions['data'])}")
        print(f"  Failed parses: {failed_parses}")
        if failed_parses > 0:
            print(f"  Warning: {failed_parses}/{len(predictions['data'])} predictions failed to parse")

        return output_path


# ─────────────────────────────────────────────────────────────────────
# Integrated evaluation
# ─────────────────────────────────────────────────────────────────────


def run_evaluation(
    predictions_path: str,
    ground_truth_path: str,
    eval_output_path: str,
) -> bool:
    """
    Run JudgeEvaluator on the predictions we just wrote.
    Returns True on success, False on failure (caller decides what to do).

    Imported lazily so the inference path doesn't pull in dataset_splitter
    until we actually need it — keeps inference startup fast and lets
    --no-evaluate work even if dataset_splitter has an import error.
    """
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    print(f"  predictions:   {predictions_path}")
    print(f"  ground truth:  {ground_truth_path}")
    print(f"  eval output:   {eval_output_path}")
    print()

    try:
        from dataset_splitter import JudgeEvaluator
    except ImportError as e:
        print(f"❌ Could not import JudgeEvaluator from dataset_splitter: {e}")
        print("   Make sure dataset_splitter.py is on PYTHONPATH or in CWD.")
        print("   Predictions are still saved; you can run the eval manually:")
        print(f"     python dataset_splitter.py --action evaluate \\")
        print(f"         --dataset-path {ground_truth_path} \\")
        print(f"         --predictions-path {predictions_path}")
        return False

    try:
        evaluator = JudgeEvaluator()
        _ = evaluator.evaluate(
            predictions_path=predictions_path,
            ground_truth_path=ground_truth_path,
            output_path=eval_output_path,
        )
        return True
    except Exception as e:
        print(f"\n❌ Evaluation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n   Predictions are still saved at:")
        print(f"     {predictions_path}")
        print("   You can re-run eval manually once the issue is fixed:")
        print(f"     python dataset_splitter.py --action evaluate \\")
        print(f"         --dataset-path {ground_truth_path} \\")
        print(f"         --predictions-path {predictions_path}")
        return False


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run judge model inference + eval")

    parser.add_argument('--model', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test split JSON (used for inference and, by default, for eval)')
    parser.add_argument('--toolhop-path', type=str, required=True,
                        help='Path to original ToolHop.json')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output path for predictions JSON')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-8bit', action='store_true', default=True)
    parser.add_argument('--no-8bit', dest='use_8bit', action='store_false')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--print-every', type=int, default=0)
    parser.add_argument('--print-chars', type=int, default=600)

    # ── NEW: integrated evaluation ──────────────────────────────────────
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=True,
                        help='Run JudgeEvaluator on predictions after inference (default: ON)')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false',
                        help='Skip the post-inference eval step')
    parser.add_argument('--eval-ground-truth', type=str, default=None,
                        help='Ground truth file for evaluation (default: same as --test-data, '
                             'which already has the reference annotations)')
    parser.add_argument('--eval-output', type=str, default=None,
                        help='Where to save evaluation results JSON '
                             '(default: <output>.eval.json next to the predictions file)')
    # ────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    # Inference
    inference = JudgeInference(
        model_path=args.model, device=args.device, use_8bit=args.use_8bit,
    )
    predictions_path = inference.predict_batch(
        test_data_path=args.test_data,
        toolhop_path=args.toolhop_path,
        output_path=args.output,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        print_every=args.print_every,
        print_chars=args.print_chars,
    )

    # Evaluation (optional, default ON)
    if args.evaluate:
        gt_path = args.eval_ground_truth or args.test_data
        if args.eval_output:
            eval_out = args.eval_output
        else:
            # Default: <output>.eval.json next to predictions file
            p = Path(predictions_path)
            eval_out = str(p.with_suffix(".eval.json"))

        ok = run_evaluation(
            predictions_path=predictions_path,
            ground_truth_path=gt_path,
            eval_output_path=eval_out,
        )

        if ok:
            print("\n" + "=" * 80)
            print("✅ INFERENCE + EVALUATION COMPLETE")
            print("=" * 80)
            print(f"  Predictions: {predictions_path}")
            print(f"  Eval JSON:   {eval_out}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠  Inference completed, evaluation did not")
            print("=" * 80)
            print(f"  Predictions saved at: {predictions_path}")
            print("  See error message above for details.")
            print("=" * 80)
            # Exit 0 — predictions are valuable, don't lose them to a CI failure
    else:
        print("\n" + "=" * 80)
        print("✅ INFERENCE COMPLETE  (eval skipped via --no-evaluate)")
        print("=" * 80)
        print(f"  Predictions: {predictions_path}")
        print()
        print("  To run eval later:")
        print(f"    python dataset_splitter.py --action evaluate \\")
        print(f"        --dataset-path {args.test_data} \\")
        print(f"        --predictions-path {predictions_path}")
        print("=" * 80)


if __name__ == '__main__':
    main()