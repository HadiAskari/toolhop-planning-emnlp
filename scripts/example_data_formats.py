"""
Example Usage and Data Format Reference

This script shows:
1. Expected input format (ToolHop)
2. Generated output format
3. How to load and use the data
"""

import json


# ============================================================================
# EXAMPLE 1: ToolHop Input Format
# ============================================================================

toolhop_example = {
    "id": 0,
    "question": "How many letters (exclude first and last) are in the first name of the designer of Stanley Park?",
    "answer": "4",
    "sub_task": {
        "Salisbury Woodland Gardens links a zoo with which park?": "Stanley Park, Blackpool",
        "Stanley Park is designed and built by who?": "Thomas Mawson",
        "What is the first name of Thomas Mawson?": "Thomas",
        "How many letters (exclude first and last) in Thomas?": "4"
    },
    "tools": {
        "Salisbury Woodland Gardens links a zoo with which park?": {
            "name": "geo_relationship_finder",
            "description": "Find relationships between locations",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {"type": "string"},
                    "entity_types": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["location_name"]
            }
        },
        "Stanley Park is designed and built by who?": {
            "name": "historical_figure_identifier",
            "description": "Identify historical figures",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_name": {"type": "string"}
                },
                "required": ["event_name"]
            }
        }
        # ... more tools
    }
}

print("="*80)
print("EXAMPLE 1: ToolHop Input Format")
print("="*80)
print(json.dumps(toolhop_example, indent=2)[:500] + "...")


# ============================================================================
# EXAMPLE 2: Generated Output Format
# ============================================================================

generated_example = {
    "metadata": {
        "n_queries": 10,
        "n_candidates_per_query": 10,
        "total_plans": 100,
        "model": "gpt-4o"
    },
    "data": [
        {
            "query_id": 0,
            "plan": {
                "steps": [
                    {
                        "step_id": 0,
                        "tool_name": "geo_relationship_finder",
                        "parameters": {
                            "location_name": "Salisbury Woodland Gardens",
                            "entity_types": ["zoo", "park"]
                        },
                        "output_variable": "{{0}}",
                        "expected_output": "Stanley Park, Blackpool"
                    },
                    {
                        "step_id": 1,
                        "tool_name": "historical_figure_identifier",
                        "parameters": {
                            "event_name": "{{0}}"
                        },
                        "output_variable": "{{1}}",
                        "expected_output": "Thomas Mawson"
                    },
                    {
                        "step_id": 2,
                        "tool_name": "extract_first_name",
                        "parameters": {
                            "full_name": "{{1}}"
                        },
                        "output_variable": "{{2}}",
                        "expected_output": "Thomas"
                    },
                    {
                        "step_id": 3,
                        "tool_name": "count_letters",
                        "parameters": {
                            "input": "{{2}}",
                            "ignore_position": ["first", "last"]
                        },
                        "output_variable": "{{3}}",
                        "expected_output": "4"
                    }
                ],
                "error_type": "none"
            },
            "annotation": {
                "quality_score": 95,
                "success_prediction": "yes",
                "confidence": 0.95,
                "reasoning": "This plan correctly breaks down the query into logical steps. Step 0 finds the park linked to the gardens, Step 1 identifies the designer, Step 2 extracts the first name, and Step 3 counts the letters excluding first and last. All dependencies are properly chained using {{N}} references. The tool selections are appropriate for each sub-task.",
                "issues": []
            }
        },
        {
            "query_id": 0,
            "plan": {
                "steps": [
                    {
                        "step_id": 0,
                        "tool_name": "geo_relationship_finder",
                        "parameters": {
                            "location_name": "Salisbury Woodland Gardens",
                            "entity_types": 123  # TYPE MISMATCH: should be array
                        },
                        "output_variable": "{{0}}",
                        "expected_output": None
                    },
                    # ... rest of steps
                ],
                "error_type": "type_mismatch"
            },
            "annotation": {
                "quality_score": 25,
                "success_prediction": "no",
                "confidence": 0.9,
                "reasoning": "This plan has a critical type mismatch error in Step 0. The entity_types parameter expects an array of strings, but an integer (123) is provided. This will cause the tool call to fail immediately, preventing the entire plan from executing successfully.",
                "issues": [
                    {
                        "type": "type_mismatch",
                        "severity": "critical",
                        "description": "entity_types parameter expects array but got integer",
                        "suggestion": "Change entity_types to [\"zoo\", \"park\"]"
                    }
                ]
            }
        }
    ]
}

print("\n" + "="*80)
print("EXAMPLE 2: Generated Output Format (Truncated)")
print("="*80)
print(json.dumps(generated_example, indent=2)[:1000] + "...")


# ============================================================================
# EXAMPLE 3: Loading and Using the Generated Data
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: How to Load and Use the Data")
print("="*80)

def load_dataset(path: str):
    """Load the generated dataset"""
    with open(path, 'r') as f:
        return json.load(f)

def filter_by_error_type(dataset, error_type: str):
    """Filter plans by error type"""
    return [
        item for item in dataset['data']
        if item['plan']['error_type'] == error_type
    ]

def filter_by_quality_range(dataset, min_score: int, max_score: int):
    """Filter plans by quality score range"""
    return [
        item for item in dataset['data']
        if min_score <= item['annotation']['quality_score'] <= max_score
    ]

def create_preference_pairs(dataset, min_diff: int = 15):
    """Create preference pairs for DPO training"""
    pairs = []
    
    # Group by query_id
    by_query = {}
    for item in dataset['data']:
        qid = item['query_id']
        if qid not in by_query:
            by_query[qid] = []
        by_query[qid].append(item)
    
    # Create pairs within each query
    for qid, items in by_query.items():
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                score1 = item1['annotation']['quality_score']
                score2 = item2['annotation']['quality_score']
                
                if abs(score1 - score2) >= min_diff:
                    if score1 > score2:
                        pairs.append({
                            'query_id': qid,
                            'chosen': item1,
                            'rejected': item2,
                            'score_diff': score1 - score2
                        })
                    else:
                        pairs.append({
                            'query_id': qid,
                            'chosen': item2,
                            'rejected': item1,
                            'score_diff': score2 - score1
                        })
    
    return pairs

def format_for_judge_training(item):
    """Format an item for judge model training"""
    # Input: query + tools + plan
    input_text = f"Query: {item['query_id']}\n"  # You'd load actual query text
    input_text += f"Plan:\n"
    for step in item['plan']['steps']:
        params = ", ".join([f"{k}={v}" for k, v in step['parameters'].items()])
        input_text += f"  Step {step['step_id']}: {step['tool_name']}({params})\n"
    
    # Output: annotation
    output_text = f"<analysis>\n{item['annotation']['reasoning']}\n</analysis>\n"
    output_text += f"<judgment>\n"
    output_text += f"Quality: {item['annotation']['quality_score']}/100\n"
    output_text += f"Success: {item['annotation']['success_prediction']}\n"
    output_text += f"</judgment>"
    
    return {
        'input': input_text,
        'output': output_text
    }

# Example usage
print("""
# Load dataset
dataset = load_dataset('toolhop_candidates.json')

# Filter ground truth plans (for planner SFT)
ground_truth = filter_by_error_type(dataset, 'none')
print(f"Ground truth plans: {len(ground_truth)}")

# Filter high-quality plans (score >= 80)
high_quality = filter_by_quality_range(dataset, 80, 100)
print(f"High-quality plans: {len(high_quality)}")

# Filter low-quality plans (score < 50)
low_quality = filter_by_quality_range(dataset, 0, 49)
print(f"Low-quality plans: {len(low_quality)}")

# Create preference pairs for DPO
pairs = create_preference_pairs(dataset, min_diff=15)
print(f"Preference pairs: {len(pairs)}")

# Format for judge training
judge_data = [format_for_judge_training(item) for item in dataset['data']]
print(f"Judge training examples: {len(judge_data)}")
""")


# ============================================================================
# EXAMPLE 4: Statistics and Analysis
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Dataset Statistics")
print("="*80)

def analyze_dataset(dataset):
    """Compute statistics on the dataset"""
    data = dataset['data']
    
    # Error type distribution
    error_types = {}
    for item in data:
        et = item['plan']['error_type']
        error_types[et] = error_types.get(et, 0) + 1
    
    # Quality score distribution
    scores = [item['annotation']['quality_score'] for item in data]
    
    # Success prediction distribution
    success_preds = {}
    for item in data:
        pred = item['annotation']['success_prediction']
        success_preds[pred] = success_preds.get(pred, 0) + 1
    
    # Issue severity distribution
    severities = {}
    for item in data:
        for issue in item['annotation']['issues']:
            sev = issue['severity']
            severities[sev] = severities.get(sev, 0) + 1
    
    return {
        'error_types': error_types,
        'quality_scores': {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'median': sorted(scores)[len(scores)//2]
        },
        'success_predictions': success_preds,
        'issue_severities': severities
    }

print("""
stats = analyze_dataset(dataset)

print("Error Type Distribution:")
for et, count in stats['error_types'].items():
    print(f"  {et}: {count}")

print("\\nQuality Score Statistics:")
for key, value in stats['quality_scores'].items():
    print(f"  {key}: {value:.1f}")

print("\\nSuccess Prediction Distribution:")
for pred, count in stats['success_predictions'].items():
    print(f"  {pred}: {count}")
""")


# ============================================================================
# EXAMPLE 5: Preparing Data for Training
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 5: Preparing Data for Model Training")
print("="*80)

print("""
# For Judge SFT Training
def prepare_judge_sft_data(dataset, output_path):
    training_data = []
    
    for item in dataset['data']:
        # Format as instruction-following example
        example = {
            'instruction': 'Evaluate this tool execution plan.',
            'input': format_judge_input(item),
            'output': format_judge_output(item['annotation'])
        }
        training_data.append(example)
    
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

# For Judge DPO Training
def prepare_judge_dpo_data(dataset, output_path):
    pairs = create_preference_pairs(dataset, min_diff=15)
    
    dpo_data = []
    for pair in pairs:
        example = {
            'prompt': format_judge_input(pair['chosen']),
            'chosen': format_judge_output(pair['chosen']['annotation']),
            'rejected': format_judge_output(pair['rejected']['annotation'])
        }
        dpo_data.append(example)
    
    with open(output_path, 'w') as f:
        json.dump(dpo_data, f, indent=2)

# For Planner SFT Training
def prepare_planner_sft_data(dataset, output_path):
    # Use only ground truth plans
    ground_truth = filter_by_error_type(dataset, 'none')
    
    training_data = []
    for item in ground_truth:
        example = {
            'instruction': 'Generate a plan to solve this query.',
            'input': format_planner_input(item),
            'output': format_planner_output(item['plan'])
        }
        training_data.append(example)
    
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

# Usage
prepare_judge_sft_data(dataset, 'judge_sft_data.json')
prepare_judge_dpo_data(dataset, 'judge_dpo_data.json')
prepare_planner_sft_data(dataset, 'planner_sft_data.json')
""")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The generated dataset contains:

1. Multiple candidate plans per query (default: 10)
   - 1 ground truth plan (error_type = "none")
   - 9 plans with different error types

2. Each plan has:
   - Structured steps with tool calls and parameters
   - Dependency references ({{0}}, {{1}}, etc.)
   - Error type label

3. Each plan is annotated with:
   - Quality score (0-100)
   - Success prediction (yes/likely_yes/uncertain/likely_no/no)
   - Detailed reasoning
   - List of specific issues with severity levels
   - Confidence score

4. Use cases:
   - Judge SFT: Train judge to predict quality scores
   - Judge DPO: Teach judge to rank plans correctly
   - Planner SFT: Train planner to generate valid plans
   - Planner RL: Use judge scores as rewards
   - Evaluation: Benchmark on different error types

5. Key metrics to track:
   - Judge score MAE (mean absolute error)
   - Success prediction accuracy
   - Correlation between judge scores and actual execution
   - Error type detection precision/recall
""")

print("\nFor complete usage instructions, see README_USAGE.md")
