#!/usr/bin/env python3
"""
Complete Test Script for SFT-Trained Planner Model

This script fully tests your fine-tuned planner model by:
1. Loading ToolHop data (queries + tools)
2. Generating plans with your model
3. Parsing and comparing with ground truth
4. Computing comprehensive metrics

Usage:
    python test_planner_sft.py \
        --model checkpoints_planner_sft-7b-hf \
        --test-data data/planner_sft/test_sft.parquet \
        --toolhop ToolHop.json \
        --output results.json \
        --n-samples 50
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import re
import numpy as np
from collections import defaultdict


def format_tools_for_prompt(tools: dict) -> str:
    """Format tool definitions (same as training)"""
    tools_str = "Available Tools:\n"
    
    # Extract unique tools (ToolHop has duplicate tool definitions)
    unique_tools = {}
    for sub_question, tool_info in tools.items():
        tool_name = tool_info.get('name', sub_question)
        if tool_name not in unique_tools:
            unique_tools[tool_name] = tool_info
    
    for tool_name, tool_info in unique_tools.items():
        params = tool_info.get('parameters', {})
        properties = params.get('properties', {})
        required_params = params.get('required', [])
        
        param_parts = []
        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'any')
            is_required = param_name in required_params
            req_marker = " (required)" if is_required else ""
            param_parts.append(f"{param_name}: {param_type}{req_marker}")
        
        params_str = ", ".join(param_parts)
        tools_str += f"- {tool_name}({params_str})\n"
    
    return tools_str


def create_inference_prompt(query: str, tools: dict, tokenizer) -> str:
    """Create prompt for inference (same as training)"""
    
    tools_str = format_tools_for_prompt(tools)
    
    user_message = f"""Generate a tool execution plan to answer this query.

Query: {query}

{tools_str}

Generate a step-by-step plan using the available tools. Each step should:
1. Call exactly one tool
2. Use output variables {{{{0}}}}, {{{{1}}}}, {{{{2}}}}, etc. for results
3. Reference previous step outputs using {{{{N}}}} syntax
4. Provide all required parameters

Format each step as: Step N: {{{{N}}}} = tool_name(param1=value1, param2=value2, ...)"""
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert at creating multi-step tool execution plans. Given a query and available tools, generate a correct sequence of tool calls to answer the query."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def generate_plan(
    model,
    tokenizer,
    query: str,
    tools: dict,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    device: str = 'cuda'
) -> str:
    """Generate a plan using the model"""
    
    # Create prompt
    prompt = create_inference_prompt(query, tools, tokenizer)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated.strip()


def parse_plan_from_text(plan_text: str) -> list:
    """
    Parse plan text into structured steps.
    
    Expected format:
    Step 0: {{0}} = tool_name(param1=value1, param2=value2)
    Step 1: {{1}} = tool_name2(...)
    
    Returns list of:
    {
        'step_id': int,
        'output_variable': str,
        'tool_name': str,
        'parameters': dict
    }
    """
    steps = []
    
    # Split by lines and process each
    lines = plan_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('Step '):
            continue
        
        # Try to parse: Step N: {{N}} = tool_name(params)
        try:
            # Extract step number
            step_match = re.match(r'Step (\d+):', line)
            if not step_match:
                continue
            
            step_id = int(step_match.group(1))
            
            # Extract output variable
            var_match = re.search(r'(\{\{\d+\}\})\s*=', line)
            if not var_match:
                continue
            
            output_var = var_match.group(1)
            
            # Extract tool name and parameters
            tool_match = re.search(r'=\s*([^\(]+)\((.*)\)\s*$', line)
            if not tool_match:
                # Maybe no parameters
                tool_match = re.search(r'=\s*([^\(]+)\(\)\s*$', line)
                if tool_match:
                    tool_name = tool_match.group(1).strip()
                    params = {}
                else:
                    continue
            else:
                tool_name = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                
                # Parse parameters - this is simplified
                params = {}
                if params_str:
                    # Handle nested structures by counting parentheses
                    param_parts = []
                    current = ""
                    depth = 0
                    in_string = False
                    string_char = None
                    
                    for char in params_str:
                        if char in ('"', "'") and (not in_string or char == string_char):
                            in_string = not in_string
                            if in_string:
                                string_char = char
                            else:
                                string_char = None
                        
                        if not in_string:
                            if char in '([{':
                                depth += 1
                            elif char in ')]}':
                                depth -= 1
                            elif char == ',' and depth == 0:
                                param_parts.append(current.strip())
                                current = ""
                                continue
                        
                        current += char
                    
                    if current.strip():
                        param_parts.append(current.strip())
                    
                    # Parse each parameter
                    for part in param_parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            params[key.strip()] = value.strip()
            
            steps.append({
                'step_id': step_id,
                'output_variable': output_var,
                'tool_name': tool_name,
                'parameters': params
            })
            
        except Exception as e:
            # Skip malformed lines
            print(f"Warning: Could not parse line: {line[:100]}")
            continue
    
    return steps


def normalize_value(value: str) -> str:
    """Normalize parameter value for comparison"""
    # Remove quotes
    value = value.strip().strip('"').strip("'")
    # Lowercase
    value = value.lower()
    # Remove extra whitespace
    value = ' '.join(value.split())
    return value


def compare_parameters(gen_params: dict, gt_params: dict) -> dict:
    """
    Compare generated and ground truth parameters.
    
    Returns:
    {
        'total_params': int,
        'correct_params': int,
        'missing_params': list,
        'extra_params': list,
        'incorrect_params': list
    }
    """
    gt_keys = set(gt_params.keys())
    gen_keys = set(gen_params.keys())
    
    missing = list(gt_keys - gen_keys)
    extra = list(gen_keys - gt_keys)
    common = gt_keys & gen_keys
    
    incorrect = []
    correct_count = 0
    
    for key in common:
        gt_val = normalize_value(str(gt_params[key]))
        gen_val = normalize_value(str(gen_params[key]))
        
        if gt_val == gen_val or gt_val in gen_val or gen_val in gt_val:
            correct_count += 1
        else:
            incorrect.append({
                'param': key,
                'generated': gen_params[key],
                'ground_truth': gt_params[key]
            })
    
    return {
        'total_params': len(gt_params),
        'correct_params': correct_count,
        'missing_params': missing,
        'extra_params': extra,
        'incorrect_params': incorrect
    }


def evaluate_plan(gen_steps: list, gt_steps: list) -> dict:
    """
    Comprehensive evaluation of generated plan vs ground truth.
    
    Returns detailed metrics:
    - Step count match
    - Tool selection accuracy
    - Parameter accuracy
    - Dependency correctness
    - Overall quality
    """
    # Basic checks
    if not gen_steps:
        return {
            'valid': False,
            'error': 'No steps generated',
            'step_count_match': False,
            'tool_accuracy': 0.0,
            'param_accuracy': 0.0,
            'exact_match': False,
            'step_details': []
        }
    
    if not gt_steps:
        return {
            'valid': False,
            'error': 'No ground truth steps',
            'step_count_match': False,
            'tool_accuracy': 0.0,
            'param_accuracy': 0.0,
            'exact_match': False,
            'step_details': []
        }
    
    # Check step count
    step_count_match = len(gen_steps) == len(gt_steps)
    
    # Evaluate each step
    step_details = []
    correct_tools = 0
    total_params_correct = 0
    total_params = 0
    
    # Pair up steps
    max_steps = max(len(gen_steps), len(gt_steps))
    
    for i in range(max_steps):
        gen_step = gen_steps[i] if i < len(gen_steps) else None
        gt_step = gt_steps[i] if i < len(gt_steps) else None
        
        step_eval = {
            'step_id': i,
            'generated_tool': gen_step['tool_name'] if gen_step else None,
            'ground_truth_tool': gt_step['tool_name'] if gt_step else None,
            'tool_correct': False,
            'param_comparison': None
        }
        
        if gen_step and gt_step:
            # Check tool name
            gen_tool = gen_step['tool_name'].strip().lower()
            gt_tool = gt_step['tool_name'].strip().lower()
            
            step_eval['tool_correct'] = (gen_tool == gt_tool)
            if step_eval['tool_correct']:
                correct_tools += 1
            
            # Compare parameters
            param_comp = compare_parameters(
                gen_step['parameters'],
                gt_step['parameters']
            )
            step_eval['param_comparison'] = param_comp
            
            total_params += param_comp['total_params']
            total_params_correct += param_comp['correct_params']
        
        step_details.append(step_eval)
    
    # Calculate metrics
    tool_accuracy = correct_tools / len(gt_steps) if gt_steps else 0.0
    param_accuracy = total_params_correct / total_params if total_params > 0 else 0.0
    
    # Exact match: all tools correct AND all params correct
    exact_match = (
        step_count_match and
        tool_accuracy == 1.0 and
        param_accuracy == 1.0
    )
    
    return {
        'valid': True,
        'step_count_match': step_count_match,
        'generated_steps': len(gen_steps),
        'ground_truth_steps': len(gt_steps),
        'tool_accuracy': tool_accuracy,
        'correct_tools': correct_tools,
        'param_accuracy': param_accuracy,
        'correct_params': total_params_correct,
        'total_params': total_params,
        'exact_match': exact_match,
        'step_details': step_details
    }


def compute_aggregate_metrics(all_results: list) -> dict:
    """Compute aggregate metrics across all test examples"""
    
    # Filter valid results
    valid_results = [r for r in all_results if r['evaluation']['valid']]
    
    if not valid_results:
        return {
            'error': 'No valid results',
            'total_examples': len(all_results)
        }
    
    # Compute averages
    metrics = {
        'total_examples': len(all_results),
        'valid_examples': len(valid_results),
        'invalid_examples': len(all_results) - len(valid_results),
        
        # Accuracy metrics
        'mean_tool_accuracy': np.mean([r['evaluation']['tool_accuracy'] for r in valid_results]),
        'mean_param_accuracy': np.mean([r['evaluation']['param_accuracy'] for r in valid_results]),
        
        # Exact matches
        'exact_matches': sum(r['evaluation']['exact_match'] for r in valid_results),
        'exact_match_rate': np.mean([r['evaluation']['exact_match'] for r in valid_results]),
        
        # Step count
        'step_count_matches': sum(r['evaluation']['step_count_match'] for r in valid_results),
        'step_count_match_rate': np.mean([r['evaluation']['step_count_match'] for r in valid_results]),
        
        # Perfect tool selection
        'perfect_tool_selection': sum(r['evaluation']['tool_accuracy'] == 1.0 for r in valid_results),
        'perfect_tool_rate': np.mean([r['evaluation']['tool_accuracy'] == 1.0 for r in valid_results]),
        
        # Distribution
        'tool_accuracy_distribution': {
            '0.0-0.2': sum(r['evaluation']['tool_accuracy'] < 0.2 for r in valid_results),
            '0.2-0.4': sum(0.2 <= r['evaluation']['tool_accuracy'] < 0.4 for r in valid_results),
            '0.4-0.6': sum(0.4 <= r['evaluation']['tool_accuracy'] < 0.6 for r in valid_results),
            '0.6-0.8': sum(0.6 <= r['evaluation']['tool_accuracy'] < 0.8 for r in valid_results),
            '0.8-1.0': sum(0.8 <= r['evaluation']['tool_accuracy'] < 1.0 for r in valid_results),
            '1.0': sum(r['evaluation']['tool_accuracy'] == 1.0 for r in valid_results),
        }
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Complete test for SFT-trained planner")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test parquet file')
    parser.add_argument('--toolhop', type=str, required=True,
                       help='Path to ToolHop.json')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of test samples (None = all)')
    parser.add_argument('--output', type=str, default='planner_test_results.json',
                       help='Output file for results')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature (0 for greedy)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each example')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TESTING SFT-TRAINED PLANNER - COMPLETE EVALUATION")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    print("✓ Model loaded")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    ds = load_dataset('parquet', data_files=args.test_data, split='train')
    print(f"✓ Loaded {len(ds)} examples")
    
    # Load ToolHop
    print(f"\nLoading ToolHop from {args.toolhop}...")
    with open(args.toolhop, 'r') as f:
        toolhop = json.load(f)
    
    # Create lookup (handle both 'id' and 'query_id')
    toolhop_lookup = {}
    for item in toolhop:
        query_id = item.get('query_id', item.get('id'))
        if query_id is not None:
            toolhop_lookup[query_id] = item
    
    print(f"✓ Loaded {len(toolhop_lookup)} ToolHop items")
    
    # Determine test samples
    n_test = len(ds) if args.n_samples is None else min(args.n_samples, len(ds))
    
    print(f"\n" + "="*80)
    print(f"RUNNING INFERENCE ON {n_test} EXAMPLES")
    print("="*80)
    
    # Run tests
    all_results = []
    
    for idx in tqdm(range(n_test), desc="Testing"):
        example = ds[idx]
        query_id = example['query_id']
        
        # Get ToolHop item
        if query_id not in toolhop_lookup:
            print(f"\nWarning: query_id {query_id} not found in ToolHop, skipping")
            continue
        
        toolhop_item = toolhop_lookup[query_id]
        query = toolhop_item.get('question', toolhop_item.get('query', ''))
        tools = toolhop_item.get('tools', {})
        
        # Generate plan
        try:
            generated_text = generate_plan(
                model=model,
                tokenizer=tokenizer,
                query=query,
                tools=tools,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=args.device
            )
        except Exception as e:
            print(f"\nError generating for query_id {query_id}: {e}")
            generated_text = ""
        
        # Parse plans
        gen_steps = parse_plan_from_text(generated_text)
        gt_steps = parse_plan_from_text(example['response'])
        
        # Evaluate
        evaluation = evaluate_plan(gen_steps, gt_steps)
        
        # Store result
        result = {
            'query_id': query_id,
            'query': query,
            'quality_score': example.get('quality_score', None),
            'generated_text': generated_text,
            'generated_steps': gen_steps,
            'ground_truth_text': example['response'],
            'ground_truth_steps': gt_steps,
            'evaluation': evaluation
        }
        
        all_results.append(result)
        
        # Print if verbose
        if args.verbose and idx < 5:  # Show first 5
            print(f"\n{'='*80}")
            print(f"Example {idx} (query_id={query_id}):")
            print(f"Query: {query[:100]}...")
            print(f"\nGenerated ({len(gen_steps)} steps):")
            for step in gen_steps[:3]:
                print(f"  Step {step['step_id']}: {step['tool_name']}")
            print(f"\nGround Truth ({len(gt_steps)} steps):")
            for step in gt_steps[:3]:
                print(f"  Step {step['step_id']}: {step['tool_name']}")
            print(f"\nEvaluation:")
            print(f"  Tool Accuracy: {evaluation['tool_accuracy']:.2%}")
            print(f"  Param Accuracy: {evaluation['param_accuracy']:.2%}")
            print(f"  Exact Match: {evaluation['exact_match']}")
    
    # Compute aggregate metrics
    print(f"\n" + "="*80)
    print("COMPUTING AGGREGATE METRICS")
    print("="*80)
    
    metrics = compute_aggregate_metrics(all_results)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print("="*80)
    print(f"\nTotal Examples: {metrics['total_examples']}")
    print(f"Valid Results: {metrics['valid_examples']}")
    print(f"Invalid Results: {metrics['invalid_examples']}")
    
    print(f"\n{'─'*80}")
    print("ACCURACY METRICS")
    print("─"*80)
    print(f"Mean Tool Accuracy: {metrics['mean_tool_accuracy']:.2%}")
    print(f"Mean Parameter Accuracy: {metrics['mean_param_accuracy']:.2%}")
    
    print(f"\n{'─'*80}")
    print("EXACT MATCHES")
    print("─"*80)
    print(f"Exact Matches: {metrics['exact_matches']}/{metrics['valid_examples']}")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.2%}")
    
    print(f"\n{'─'*80}")
    print("STEP COUNT")
    print("─"*80)
    print(f"Step Count Matches: {metrics['step_count_matches']}/{metrics['valid_examples']}")
    print(f"Step Count Match Rate: {metrics['step_count_match_rate']:.2%}")
    
    print(f"\n{'─'*80}")
    print("PERFECT TOOL SELECTION")
    print("─"*80)
    print(f"Perfect Tool Selection: {metrics['perfect_tool_selection']}/{metrics['valid_examples']}")
    print(f"Perfect Tool Rate: {metrics['perfect_tool_rate']:.2%}")
    
    print(f"\n{'─'*80}")
    print("TOOL ACCURACY DISTRIBUTION")
    print("─"*80)
    for range_str, count in metrics['tool_accuracy_distribution'].items():
        print(f"  {range_str}: {count} examples")
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    output_data = {
        'config': {
            'model': args.model,
            'test_data': args.test_data,
            'n_samples': n_test,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens
        },
        'metrics': metrics,
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved to {args.output}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    if metrics['mean_tool_accuracy'] >= 0.9:
        verdict = "✅ EXCELLENT - Model performs very well!"
    elif metrics['mean_tool_accuracy'] >= 0.75:
        verdict = "✅ GOOD - Model performs well, ready for RL"
    elif metrics['mean_tool_accuracy'] >= 0.5:
        verdict = "⚠️  ACCEPTABLE - Model learned basics, needs improvement"
    else:
        verdict = "❌ POOR - Model needs more training or better data"
    
    print(f"\nVerdict: {verdict}")
    print(f"\nKey Metrics:")
    print(f"  • Tool Selection: {metrics['mean_tool_accuracy']:.1%}")
    print(f"  • Parameter Accuracy: {metrics['mean_param_accuracy']:.1%}")
    print(f"  • Exact Matches: {metrics['exact_match_rate']:.1%}")
    print(f"  • Perfect Tools: {metrics['perfect_tool_rate']:.1%}")
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()