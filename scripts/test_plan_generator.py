"""
Test and Validation Script for ToolHop Plan Generator

This script helps you:
1. Test each component independently
2. Validate data conversion
3. Debug issues with visualization
4. Run sanity checks on generated data
"""

import json
import sys
from pathlib import Path
from toolhop_plan_generator import (
    GroundTruthParser, StaticValidator, PlanGenerator, 
    LLMJudgeAnnotator, ErrorType, Plan, ToolCall
)


class PlanGeneratorTester:
    """Comprehensive testing suite for plan generator"""
    
    def __init__(self, toolhop_path: str, api_key: str):
        self.toolhop_path = toolhop_path
        self.api_key = api_key
        
        # Load test data
        with open(toolhop_path, 'r') as f:
            self.toolhop_data = json.load(f)
        
        print(f"Loaded {len(self.toolhop_data)} examples from ToolHop")
    
    def test_ground_truth_parsing(self, n_examples: int = 5):
        """Test ground truth parsing from ToolHop format"""
        print("\n" + "="*80)
        print("TEST 1: Ground Truth Parsing")
        print("="*80)
        
        for i, example in enumerate(self.toolhop_data[:n_examples]):
            print(f"\n--- Example {i} (ID: {example['id']}) ---")
            print(f"Query: {example['question']}")
            print(f"Expected answer: {example['answer']}")
            
            try:
                plan = GroundTruthParser.parse_toolhop_example(example)
                print(f"\n✓ Successfully parsed plan with {len(plan.steps)} steps:")
                for step in plan.steps:
                    print(f"  {step}")
                
                # Validate the parsed plan
                print("\nValidation:")
                print(f"  - Number of sub-tasks: {len(example['sub_task'])}")
                print(f"  - Number of parsed steps: {len(plan.steps)}")
                print(f"  - Match: {'✓' if len(plan.steps) == len(example['sub_task']) else '✗'}")
                
            except Exception as e:
                print(f"✗ Error parsing: {e}")
                import traceback
                traceback.print_exc()
    
    def test_static_validator(self, n_examples: int = 3):
        """Test static validation on various plan types"""
        print("\n" + "="*80)
        print("TEST 2: Static Validator")
        print("="*80)
        
        for i, example in enumerate(self.toolhop_data[:n_examples]):
            print(f"\n--- Example {i} (ID: {example['id']}) ---")
            print(f"Query: {example['question']}")
            
            # Parse ground truth
            ground_truth = GroundTruthParser.parse_toolhop_example(example)
            validator = StaticValidator(example["tools"])
            
            # Test 1: Validate ground truth
            print("\n1. Validating ground truth plan:")
            is_valid, errors = validator.validate(ground_truth, example['question'])
            if is_valid:
                print("   ✓ Ground truth is valid")
            else:
                print(f"   ✗ Ground truth has {len(errors)} errors:")
                for error in errors:
                    print(f"     - {error}")
            
            # Test 2: Validate plan with type mismatch
            print("\n2. Validating plan with type mismatch:")
            invalid_plan = self._create_type_mismatch_plan(ground_truth)
            is_valid, errors = validator.validate(invalid_plan, example['question'])
            if not is_valid:
                print(f"   ✓ Correctly detected {len(errors)} errors:")
                for error in errors[:3]:
                    print(f"     - {error}")
            else:
                print("   ✗ Failed to detect type mismatch")
            
            # Test 3: Validate plan with missing dependency
            print("\n3. Validating plan with missing dependency:")
            invalid_plan = self._create_missing_dependency_plan(ground_truth)
            is_valid, errors = validator.validate(invalid_plan, example['question'])
            if not is_valid:
                print(f"   ✓ Correctly detected {len(errors)} errors:")
                for error in errors[:3]:
                    print(f"     - {error}")
            else:
                print("   ✗ Failed to detect missing dependency")
    
    def test_plan_generation(self, n_examples: int = 2):
        """Test candidate plan generation"""
        print("\n" + "="*80)
        print("TEST 3: Plan Generation (Deterministic Error Injection)")
        print("="*80)
        
        generator = PlanGenerator(api_key=self.api_key, model="gpt-4o")
        
        for i, example in enumerate(self.toolhop_data[:n_examples]):
            print(f"\n--- Example {i} (ID: {example['id']}) ---")
            print(f"Query: {example['question'][:100]}...")
            
            ground_truth = GroundTruthParser.parse_toolhop_example(example)
            print(f"\nGround truth ({len(ground_truth.steps)} steps):")
            print(ground_truth)
            
            # Generate candidates with specific error types
            error_types = [
                ErrorType.TYPE_MISMATCH,
                ErrorType.MISSING_DEPENDENCY,
                ErrorType.PARAMETER_TYPO,
                ErrorType.INEFFICIENT_ORDER,
                ErrorType.INCOMPLETE_PLAN
            ]
            
            print("\nGenerating candidates with different error types:")
            for error_type in error_types:
                candidate = generator._generate_plan_with_error(
                    example['question'],
                    example['tools'],
                    ground_truth,
                    error_type
                )
                
                print(f"\n{error_type.value}:")
                print(candidate)
                
                # Validate
                validator = StaticValidator(example["tools"])
                is_valid, errors = validator.validate(candidate, example['question'])
                print(f"  Static validation: {'✓ Valid' if is_valid else f'✗ {len(errors)} errors'}")
                if errors:
                    for error in errors[:2]:
                        print(f"    - {error}")
    
    def test_llm_annotation(self, n_examples: int = 1):
        """Test LLM annotation on a few plans"""
        print("\n" + "="*80)
        print("TEST 4: LLM Judge Annotation")
        print("="*80)
        print("Note: This will make API calls to OpenAI")
        
        response = input("\nProceed with LLM annotation test? (y/n): ")
        if response.lower() != 'y':
            print("Skipping LLM annotation test")
            return
        
        annotator = LLMJudgeAnnotator(api_key=self.api_key, model="gpt-4o")
        generator = PlanGenerator(api_key=self.api_key, model="gpt-4o")
        
        for i, example in enumerate(self.toolhop_data[:n_examples]):
            print(f"\n--- Example {i} (ID: {example['id']}) ---")
            print(f"Query: {example['question']}")
            
            ground_truth = GroundTruthParser.parse_toolhop_example(example)
            
            # Test different plan types
            test_plans = [
                ("Ground Truth", ground_truth),
                ("Type Mismatch", generator._inject_type_mismatch(ground_truth, example['tools'])),
                ("Incomplete", generator._inject_incomplete_plan(ground_truth))
            ]
            
            for plan_name, plan in test_plans:
                print(f"\n{plan_name} Plan:")
                print(plan)
                
                print("\nAnnotating...")
                annotation = annotator.annotate_plan(
                    example['question'],
                    example['tools'],
                    plan,
                    ground_truth
                )
                
                print(f"\nAnnotation:")
                print(f"  Quality Score: {annotation.quality_score}/100")
                print(f"  Success Prediction: {annotation.success_prediction}")
                print(f"  Confidence: {annotation.confidence:.2f}")
                print(f"  Number of Issues: {len(annotation.issues)}")
                
                print(f"\n  Reasoning (first 200 chars):")
                print(f"  {annotation.reasoning[:200]}...")
                
                if annotation.issues:
                    print(f"\n  Issues:")
                    for issue in annotation.issues[:3]:
                        print(f"    - [{issue['severity']}] {issue['description']}")
    
    def test_full_pipeline(self, n_examples: int = 1, n_candidates: int = 5):
        """Test the complete pipeline on a small dataset"""
        print("\n" + "="*80)
        print("TEST 5: Full Pipeline (Small Scale)")
        print("="*80)
        print(f"Processing {n_examples} examples with {n_candidates} candidates each")
        print("Note: This will make multiple API calls")
        
        response = input("\nProceed with full pipeline test? (y/n): ")
        if response.lower() != 'y':
            print("Skipping full pipeline test")
            return
        
        from toolhop_plan_generator import DatasetGenerator
        
        generator = DatasetGenerator(api_key=self.api_key, model="gpt-4o")
        
        # Create a temporary subset
        temp_path = "/tmp/toolhop_test.json"
        with open(temp_path, 'w') as f:
            json.dump(self.toolhop_data[:n_examples], f)
        
        output_path = "/tmp/test_output.json"
        
        try:
            generator.generate_dataset(
                toolhop_path=temp_path,
                output_path=output_path,
                n_candidates_per_query=n_candidates,
                validate=True
            )
            
            # Load and inspect the output
            print("\n" + "="*80)
            print("Output Inspection:")
            print("="*80)
            
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            print(f"\nMetadata:")
            for key, value in output_data['metadata'].items():
                print(f"  {key}: {value}")
            
            print(f"\nData structure:")
            if output_data['data']:
                sample = output_data['data'][0]
                print(f"  Keys: {list(sample.keys())}")
                print(f"  Plan keys: {list(sample['plan'].keys())}")
                print(f"  Annotation keys: {list(sample['annotation'].keys())}")
                
                print(f"\nFirst annotated plan:")
                print(f"  Query ID: {sample['query_id']}")
                print(f"  Error Type: {sample['plan']['error_type']}")
                print(f"  Quality Score: {sample['annotation']['quality_score']}")
                print(f"  Success Prediction: {sample['annotation']['success_prediction']}")
            
            print(f"\n✓ Full pipeline test completed successfully")
            print(f"  Output saved to: {output_path}")
            
        except Exception as e:
            print(f"\n✗ Full pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def validate_generated_dataset(self, dataset_path: str):
        """Validate a previously generated dataset"""
        print("\n" + "="*80)
        print("DATASET VALIDATION")
        print("="*80)
        
        if not Path(dataset_path).exists():
            print(f"✗ Dataset file not found: {dataset_path}")
            return
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Check structure
        print("\n1. Structure Check:")
        required_keys = ['metadata', 'data']
        for key in required_keys:
            if key in dataset:
                print(f"   ✓ {key} present")
            else:
                print(f"   ✗ {key} missing")
        
        # Check metadata
        print("\n2. Metadata Check:")
        metadata = dataset.get('metadata', {})
        print(f"   Queries: {metadata.get('n_queries')}")
        print(f"   Candidates per query: {metadata.get('n_candidates_per_query')}")
        print(f"   Total plans: {metadata.get('total_plans')}")
        print(f"   Model: {metadata.get('model')}")
        
        # Sample data
        data = dataset.get('data', [])
        print(f"\n3. Data Check:")
        print(f"   Number of annotated plans: {len(data)}")
        
        if data:
            # Check first few samples
            print("\n   Inspecting first 3 samples:")
            for i, sample in enumerate(data[:3]):
                print(f"\n   Sample {i}:")
                print(f"     Query ID: {sample.get('query_id')}")
                print(f"     Error type: {sample['plan'].get('error_type')}")
                print(f"     Steps: {len(sample['plan'].get('steps', []))}")
                print(f"     Quality score: {sample['annotation'].get('quality_score')}")
                print(f"     Success prediction: {sample['annotation'].get('success_prediction')}")
                print(f"     Issues: {len(sample['annotation'].get('issues', []))}")
        
        # Distribution analysis
        print("\n4. Distribution Analysis:")
        
        # Error type distribution
        error_types = {}
        quality_scores = []
        success_predictions = {}
        
        for sample in data:
            error_type = sample['plan'].get('error_type')
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            quality_scores.append(sample['annotation'].get('quality_score', 0))
            
            success_pred = sample['annotation'].get('success_prediction')
            success_predictions[success_pred] = success_predictions.get(success_pred, 0) + 1
        
        print("\n   Error Type Distribution:")
        for error_type, count in sorted(error_types.items()):
            pct = 100 * count / len(data)
            print(f"     {error_type:25s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n   Success Prediction Distribution:")
        for pred, count in sorted(success_predictions.items()):
            pct = 100 * count / len(data)
            print(f"     {pred:15s}: {count:4d} ({pct:5.1f}%)")
        
        if quality_scores:
            print("\n   Quality Score Statistics:")
            print(f"     Mean: {sum(quality_scores)/len(quality_scores):.1f}")
            print(f"     Min:  {min(quality_scores)}")
            print(f"     Max:  {max(quality_scores)}")
        
        print("\n✓ Dataset validation complete")
    
    # Helper methods
    def _create_type_mismatch_plan(self, plan: Plan) -> Plan:
        """Create a plan with a type mismatch error"""
        if not plan.steps:
            return plan
        
        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = ToolCall(
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters=step.parameters.copy(),
                output_variable=step.output_variable,
                expected_output=step.expected_output
            )
            
            # Inject type error in first step
            if i == 0 and new_step.parameters:
                param_name = list(new_step.parameters.keys())[0]
                if isinstance(new_step.parameters[param_name], str):
                    new_step.parameters[param_name] = 123  # String -> Number
            
            new_steps.append(new_step)
        
        return Plan(steps=new_steps, error_type=ErrorType.TYPE_MISMATCH)
    
    def _create_missing_dependency_plan(self, plan: Plan) -> Plan:
        """Create a plan with a missing dependency"""
        if len(plan.steps) < 2:
            return plan
        
        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = ToolCall(
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters=step.parameters.copy(),
                output_variable=step.output_variable,
                expected_output=step.expected_output
            )
            
            # Remove dependency in second step
            if i == 1:
                for param_name, param_value in new_step.parameters.items():
                    if isinstance(param_value, str) and "{{" in param_value:
                        new_step.parameters[param_name] = "literal_value"
                        break
            
            new_steps.append(new_step)
        
        return Plan(steps=new_steps, error_type=ErrorType.MISSING_DEPENDENCY)


def main():
    """Interactive test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and validate ToolHop plan generator")
    parser.add_argument("--toolhop-path", type=str, required=True, help="Path to ToolHop JSON")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--validate-dataset", type=str, help="Path to dataset to validate")
    parser.add_argument("--test", type=str, choices=['all', 'parsing', 'validation', 'generation', 'annotation', 'pipeline'],
                       default='all', help="Which test to run")
    
    args = parser.parse_args()
    
    tester = PlanGeneratorTester(args.toolhop_path, args.api_key)
    
    # Run validation on existing dataset if provided
    if args.validate_dataset:
        tester.validate_generated_dataset(args.validate_dataset)
        return
    
    # Run tests
    if args.test in ['all', 'parsing']:
        tester.test_ground_truth_parsing(n_examples=5)
    
    if args.test in ['all', 'validation']:
        tester.test_static_validator(n_examples=3)
    
    if args.test in ['all', 'generation']:
        tester.test_plan_generation(n_examples=2)
    
    if args.test in ['all', 'annotation']:
        tester.test_llm_annotation(n_examples=1)
    
    if args.test in ['all', 'pipeline']:
        tester.test_full_pipeline(n_examples=1, n_candidates=5)


if __name__ == "__main__":
    main()
