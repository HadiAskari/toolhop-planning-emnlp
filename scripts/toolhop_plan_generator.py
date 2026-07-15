"""
ToolHop Candidate Plan Generator with Complete Error Types — FIXED VERSION

Fixes a critical data-quality bug in the original generator: the ToolHop
dataset stores tools as a dict keyed by the natural-language sub-question,
not by the tool's API name (e.g. "Salisbury Woodland Gardens links a zoo
with which park?" instead of "geo_relationship_finder").  The actual API
name lives in tool_spec["name"].

The original code iterated `tools.items()` and used the dict KEY as the
tool name everywhere — in the prompt to GPT, in static validation, in
error injection, and in the judge prompt.  As a result every generated
plan had natural-language tool names baked in, the StaticValidator never
caught it (because it was looking those NL names up in the same dict),
and the wrong-tool injector swapped between NL-named tools.

This file fixes all of that by reindexing the tools dict by API name once
in `DatasetGenerator.generate()`, then passing the reindexed dict to every
component.  Every downstream `tools.get(...)` / `tools.items()` call now
works correctly without further changes.

This script generates N candidate plans from ToolHop dataset with:
1. LLM-based ground truth plan generation (clean API tool names)
2. All 9 error types injection
3. Static validation (now catches unknown tools correctly)
4. LLM judge annotation with comprehensive rubric
5. Validation and debugging
"""

import json
import argparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict
import openai
from tqdm import tqdm
import random


# ─── Tool reindexing helper ───────────────────────────────────────────────────
# All ToolHop tool dicts come keyed by NL sub-question.  This reindexes them
# by the actual API name (tool_spec["name"]) so every component downstream
# can use straightforward `tools.get(api_name)` lookups.

def reindex_tools_by_api_name(tools_raw: Dict) -> Dict[str, Dict]:
    """
    Convert a ToolHop tools dict keyed by NL sub-question into one keyed by
    the API name found in tool_spec["name"].

    If a tool entry has no "name" field, falls back to a sanitized version
    of its original key (lowercase, underscores, max 60 chars) so we never
    leak natural-language strings as tool names.
    """
    reindexed: Dict[str, Dict] = {}
    for nl_key, tool_spec in tools_raw.items():
        api_name = tool_spec.get("name") if isinstance(tool_spec, dict) else None
        if not api_name:
            # Fallback: sanitize the dict key into something tool-name-shaped.
            sanitized = re.sub(r"[^a-z0-9_]+", "_", nl_key.lower()).strip("_")
            api_name = sanitized[:60] if sanitized else "unnamed_tool"
        # If two entries map to the same api_name, keep the first; this
        # matches the de-duplication behavior of the inference-time prompt
        # builder in best_of_n_selection.py.
        if api_name not in reindexed:
            reindexed[api_name] = tool_spec
    return reindexed


# ─── Error types ──────────────────────────────────────────────────────────────

class ErrorType(Enum):
    """Types of errors to inject into plans"""
    NONE = "none"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_DEPENDENCY = "missing_dependency"
    WRONG_TOOL = "wrong_tool"
    PARAMETER_TYPO = "parameter_typo"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    INEFFICIENT_ORDER = "inefficient_order"
    INCOMPLETE_PLAN = "incomplete_plan"
    UNNECESSARY_STEPS = "unnecessary_steps"
    FORWARD_REFERENCE = "forward_reference"


@dataclass
class Step:
    """Represents a single step in a plan"""
    step_id: int
    tool_name: str
    parameters: Dict[str, Any]
    output_variable: str
    expected_output: Optional[str] = None

    def __str__(self):
        params_str = ", ".join([f"{k}={repr(v)}" for k, v in self.parameters.items()])
        return f"Step {self.step_id}: {self.output_variable} = {self.tool_name}({params_str})"


@dataclass
class Plan:
    """Represents a complete multi-step plan"""
    steps: List[Step]
    error_type: str = "none"

    def to_dict(self):
        return {
            "steps": [asdict(step) for step in self.steps],
            "error_type": self.error_type,
        }

    def __str__(self):
        return "\n".join([str(step) for step in self.steps])


@dataclass
class Annotation:
    """Represents the LLM judge's annotation of a plan"""
    quality_score: int
    success_prediction: str
    reasoning: str
    issues: List[Dict[str, Any]]
    confidence: float

    def to_dict(self):
        return asdict(self)


# ─── Ground truth parser ──────────────────────────────────────────────────────

class GroundTruthParser:
    """Generates ground truth plans from ToolHop format using LLM.

    Receives an API-name-keyed tools dict; uses the dict keys directly as
    tool names in the prompt and in lookups.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def parse_toolhop_example(self, example: Dict, tools_by_api_name: Dict) -> Plan:
        """
        Generate ground truth plan from a ToolHop example using LLM.

        Args:
            example: ToolHop dataset example (question, tools_raw, sub_task, answer)
            tools_by_api_name: Tools dict already reindexed by API name

        Returns:
            Ground truth Plan with API-style tool names.
        """
        query = example["question"]
        answer = example["answer"]
        sub_tasks = example.get("sub_task", {})

        prompt = self._build_ground_truth_prompt(
            query, tools_by_api_name, answer, sub_tasks
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at creating multi-step tool execution "
                            "plans. Generate plans that are syntactically correct with "
                            "all required parameters filled in."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=3000,
            )

            content = response.choices[0].message.content
            plan = self._parse_plan_from_response(content, tools_by_api_name)
            plan = self._add_expected_outputs(plan, sub_tasks)
            return plan

        except Exception as e:
            print(f"Error generating ground truth plan: {e}")
            return self._create_fallback_plan(example)

    def _add_expected_outputs(self, plan: Plan, sub_tasks: Dict[str, str]) -> Plan:
        sub_task_outputs = list(sub_tasks.values()) if sub_tasks else []
        for i, step in enumerate(plan.steps):
            if i < len(sub_task_outputs):
                step.expected_output = str(sub_task_outputs[i])
        return plan

    def _build_ground_truth_prompt(
        self, query: str, tools_by_api_name: Dict, answer: str, sub_tasks: Dict
    ) -> str:
        tools_desc = self._format_tools_for_prompt(tools_by_api_name)
        sub_tasks_desc = self._format_sub_tasks(sub_tasks)

        prompt = f"""Generate a complete, syntactically valid multi-step plan to answer the query.

**QUERY:**
{query}

**FINAL ANSWER:**
{answer}

**SUB-TASKS (intermediate steps):**
{sub_tasks_desc}

**AVAILABLE TOOLS:**
{tools_desc}

**CRITICAL REQUIREMENTS:**

1. **Use the EXACT tool name (API name) shown above.** Tool names are short
   API-style identifiers like `geo_relationship_finder`, `extract_first_name`,
   `count_letters`. NEVER use a natural-language description as a tool name.

2. **ALL required parameters MUST be provided for EVERY tool call**
   - Check each tool's "Required parameters" list
   - Never skip a required parameter
   - If you're unsure of a value, use a reasonable default or infer from context

3. **Parameter value rules:**
   - First step: Use literal values from the query (e.g., location_name='Salisbury Woodland Gardens')
   - Subsequent steps: Use {{{{N}}}} to reference prior step outputs (e.g., name='{{{{0}}}}')
   - Use the sub_tasks information to infer what values to use

4. **Output format:**
   Return a valid JSON array of steps. Each step must have:
   - step_id (integer starting from 0)
   - tool_name (EXACT match from the API names listed above)
   - parameters (dict with ALL required params + any helpful optional params)
   - output_variable (always "{{{{N}}}}" where N is the step_id)

**EXAMPLE OUTPUT FORMAT:**
```json
[
  {{
    "step_id": 0,
    "tool_name": "search_tool",
    "parameters": {{
      "query": "example query",
      "max_results": 10
    }},
    "output_variable": "{{{{0}}}}"
  }},
  {{
    "step_id": 1,
    "tool_name": "extract_tool",
    "parameters": {{
      "input": "{{{{0}}}}",
      "field": "name"
    }},
    "output_variable": "{{{{1}}}}"
  }}
]
```

**DOUBLE-CHECK BEFORE RESPONDING:**
- Did you use API tool names (NOT natural-language descriptions)?
- Did you include ALL required parameters for each tool?
- Are the parameter types correct (strings in quotes, numbers without quotes, arrays as lists)?
- Do all {{{{N}}}} references point to valid prior steps?

Generate the plan now:"""

        return prompt

    def _format_tools_for_prompt(self, tools_by_api_name: Dict) -> str:
        """Format tools for the LLM prompt using API names as the canonical tool name."""
        formatted = []
        for api_name, tool_spec in tools_by_api_name.items():
            params = tool_spec.get("parameters", {})
            required = params.get("required", [])
            properties = params.get("properties", {})

            param_desc = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                is_required = "REQUIRED" if param_name in required else "optional"
                param_desc.append(f"  - {param_name} ({param_type}) [{is_required}]")

            formatted.append(f"{api_name}:\n" + "\n".join(param_desc))

        return "\n\n".join(formatted)

    def _format_sub_tasks(self, sub_tasks: Dict) -> str:
        if not sub_tasks:
            return "No sub-tasks provided"
        formatted = []
        for i, (key, value) in enumerate(sub_tasks.items()):
            formatted.append(f"Step {i}: {value}")
        return "\n".join(formatted)

    def _parse_plan_from_response(self, content: str, tools_by_api_name: Dict) -> Plan:
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        try:
            steps_data = json.loads(json_str)
            steps = [Step(**step_data) for step_data in steps_data]
            steps = self._validate_and_fix_parameters(steps, tools_by_api_name)
            return Plan(steps=steps, error_type="none")
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response content: {content[:500]}")
            raise

    def _validate_and_fix_parameters(
        self, steps: List[Step], tools_by_api_name: Dict
    ) -> List[Step]:
        """Add default values for missing required parameters.

        Tool lookup is by API name (the dict key of `tools_by_api_name`).
        """
        for step in steps:
            tool_spec = tools_by_api_name.get(step.tool_name)
            if not tool_spec:
                # The LLM emitted a tool name we don't know about.  This can
                # happen if it hallucinates or strays from the API names; we
                # leave the step as-is so the static validator catches it.
                continue

            params = tool_spec.get("parameters", {})
            required = params.get("required", [])
            properties = params.get("properties", {})

            for req_param in required:
                if req_param not in step.parameters:
                    param_info = properties.get(req_param, {})
                    param_type = param_info.get("type", "string")

                    if param_type == "string":
                        step.parameters[req_param] = "default_value"
                    elif param_type == "number" or param_type == "integer":
                        step.parameters[req_param] = 0
                    elif param_type == "boolean":
                        step.parameters[req_param] = False
                    elif param_type == "array":
                        step.parameters[req_param] = []
                    else:
                        step.parameters[req_param] = None

                    print(
                        f"Warning: Added default for missing required param "
                        f"'{req_param}' in {step.tool_name}"
                    )

        return steps

    def _create_fallback_plan(self, example: Dict) -> Plan:
        return Plan(
            steps=[
                Step(
                    step_id=0,
                    tool_name="fallback_tool",
                    parameters={},
                    output_variable="{{0}}",
                    expected_output=example.get("answer", ""),
                )
            ],
            error_type="none",
        )


# ─── Static validator ─────────────────────────────────────────────────────────

class StaticValidator:
    """Validates plans using static analysis. Tools dict must be API-name-keyed."""

    def __init__(self, tools_by_api_name: Dict):
        self.tools = tools_by_api_name

    def validate(self, plan: Plan) -> Tuple[bool, List[str]]:
        errors = []

        for step in plan.steps:
            if step.tool_name not in self.tools:
                errors.append(f"Step {step.step_id}: Unknown tool '{step.tool_name}'")
                continue

            tool_spec = self.tools[step.tool_name]
            params_spec = tool_spec.get("parameters", {})
            required = params_spec.get("required", [])
            properties = params_spec.get("properties", {})

            for req_param in required:
                if req_param not in step.parameters:
                    errors.append(
                        f"Step {step.step_id}: Missing required parameter '{req_param}'"
                    )

            for param_name, param_value in step.parameters.items():
                if param_name not in properties:
                    continue

                expected_type = properties[param_name].get("type")
                if expected_type and not self._is_dependency_ref(param_value):
                    if not self._check_type(param_value, expected_type):
                        actual_type = type(param_value).__name__
                        errors.append(
                            f"Step {step.step_id}: Parameter '{param_name}' should "
                            f"be {expected_type}, got {actual_type}"
                        )

        dep_errors = self._check_dependencies(plan)
        errors.extend(dep_errors)

        circular = self._check_circular_dependencies(plan)
        if circular:
            errors.append(f"Circular dependency detected: {circular}")

        return len(errors) == 0, errors

    def _is_dependency_ref(self, value) -> bool:
        if isinstance(value, str):
            return "{{" in value and "}}" in value
        elif isinstance(value, list):
            return any(self._is_dependency_ref(v) for v in value)
        return False

    def _check_type(self, value, expected_type: str) -> bool:
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True
        return isinstance(value, expected_python_type)

    def _check_dependencies(self, plan: Plan) -> List[str]:
        errors = []
        max_step_id = max(step.step_id for step in plan.steps) if plan.steps else -1

        for step in plan.steps:
            for param_value in step.parameters.values():
                if isinstance(param_value, str):
                    refs = re.findall(r"\{\{(\d+)\}\}", param_value)
                    for ref in refs:
                        ref_id = int(ref)
                        if ref_id >= step.step_id:
                            errors.append(
                                f"Step {step.step_id}: Invalid forward reference to step {ref_id}"
                            )
                        if ref_id > max_step_id:
                            errors.append(
                                f"Step {step.step_id}: Reference to non-existent step {ref_id}"
                            )

        return errors

    def _check_circular_dependencies(self, plan: Plan) -> Optional[str]:
        graph = defaultdict(list)
        for step in plan.steps:
            deps = []
            for param_value in step.parameters.values():
                if isinstance(param_value, str):
                    refs = re.findall(r"\{\{(\d+)\}\}", param_value)
                    deps.extend([int(ref) for ref in refs])
            graph[step.step_id] = deps

        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        visited = set()
        for step in plan.steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id, visited, set()):
                    return f"Cycle involving step {step.step_id}"
        return None


# ─── Plan generator with error injection ──────────────────────────────────────

class PlanGenerator:
    """Generates candidate plans with error injection. Tools dict must be API-name-keyed."""

    def __init__(self, tools_by_api_name: Dict):
        self.tools = tools_by_api_name

    def generate_candidates(
        self,
        ground_truth: Plan,
        n_candidates: int = 10,
        error_distribution: Optional[Dict[str, float]] = None,
    ) -> List[Plan]:
        if error_distribution is None:
            error_distribution = {
                "none": 0.15,
                "type_mismatch": 0.10,
                "missing_dependency": 0.10,
                "wrong_tool": 0.15,
                "parameter_typo": 0.10,
                "circular_dependency": 0.05,
                "inefficient_order": 0.10,
                "incomplete_plan": 0.10,
                "unnecessary_steps": 0.10,
                "forward_reference": 0.05,
            }

        candidates = []

        # Always include the ground truth as the first candidate
        candidates.append(
            Plan(
                steps=[Step(**asdict(step)) for step in ground_truth.steps],
                error_type="none",
            )
        )

        remaining = n_candidates - 1
        error_counts = {}
        for error_type, prob in error_distribution.items():
            if error_type == "none":
                continue
            count = max(1, int(remaining * prob))
            error_counts[error_type] = count

        for error_type, count in error_counts.items():
            for _ in range(count):
                if len(candidates) >= n_candidates:
                    break
                candidate = self._inject_error(ground_truth, error_type)
                candidates.append(candidate)

        error_types = [e for e in error_distribution.keys() if e != "none"]
        while len(candidates) < n_candidates:
            error_type = random.choice(error_types)
            candidate = self._inject_error(ground_truth, error_type)
            candidates.append(candidate)

        return candidates[:n_candidates]

    def _inject_error(self, plan: Plan, error_type: str) -> Plan:
        if error_type == "type_mismatch":
            return self._inject_type_mismatch(plan)
        elif error_type == "missing_dependency":
            return self._inject_missing_dependency(plan)
        elif error_type == "wrong_tool":
            return self._inject_wrong_tool(plan)
        elif error_type == "parameter_typo":
            return self._inject_parameter_typo(plan)
        elif error_type == "circular_dependency":
            return self._inject_circular_dependency(plan)
        elif error_type == "inefficient_order":
            return self._inject_inefficient_order(plan)
        elif error_type == "incomplete_plan":
            return self._inject_incomplete_plan(plan)
        elif error_type == "unnecessary_steps":
            return self._inject_unnecessary_steps(plan)
        elif error_type == "forward_reference":
            return self._inject_forward_reference(plan)
        else:
            return Plan(
                steps=[Step(**asdict(step)) for step in plan.steps],
                error_type="none",
            )

    def _inject_type_mismatch(self, plan: Plan) -> Plan:
        new_steps = []
        injected = False

        for i, step in enumerate(plan.steps):
            new_step = Step(**asdict(step))

            if not injected and i == len(plan.steps) // 2 and new_step.parameters:
                tool_spec = self.tools.get(step.tool_name, {})
                properties = tool_spec.get("parameters", {}).get("properties", {})

                for param_name, param_value in list(new_step.parameters.items()):
                    if isinstance(param_value, str) and "{{" in param_value:
                        continue

                    param_info = properties.get(param_name, {})
                    param_type = param_info.get("type")

                    if param_type == "array" and isinstance(param_value, list):
                        new_step.parameters[param_name] = "invalid_string"
                        injected = True
                        break
                    elif param_type == "string" and isinstance(param_value, str):
                        new_step.parameters[param_name] = 42
                        injected = True
                        break
                    elif param_type in ["number", "integer"] and isinstance(
                        param_value, (int, float)
                    ):
                        new_step.parameters[param_name] = "invalid_string"
                        injected = True
                        break

            new_steps.append(new_step)

        return Plan(steps=new_steps, error_type="type_mismatch")

    def _inject_missing_dependency(self, plan: Plan) -> Plan:
        new_steps = []
        injected = False

        for i, step in enumerate(plan.steps):
            new_step = Step(**asdict(step))

            if not injected and i > 0 and i == len(plan.steps) // 2:
                for param_name, param_value in list(new_step.parameters.items()):
                    if isinstance(param_value, str) and "{{" in param_value:
                        new_step.parameters[param_name] = "hardcoded_value"
                        injected = True
                        break

            new_steps.append(new_step)

        return Plan(steps=new_steps, error_type="missing_dependency")

    def _inject_wrong_tool(self, plan: Plan) -> Plan:
        """Use the wrong tool for a step (picks from the API-name-keyed dict)."""
        new_steps = []
        injected = False

        # Pull the API names from the reindexed tools dict — this is now correct.
        tool_names = list(self.tools.keys())

        for i, step in enumerate(plan.steps):
            if not injected and i == len(plan.steps) // 2 and len(tool_names) > 1:
                other_tools = [t for t in tool_names if t != step.tool_name]
                if other_tools:
                    wrong_tool = random.choice(other_tools)
                    new_step = Step(
                        step_id=step.step_id,
                        tool_name=wrong_tool,
                        parameters=step.parameters.copy(),
                        output_variable=step.output_variable,
                        expected_output=step.expected_output,
                    )
                    new_steps.append(new_step)
                    injected = True
                    continue

            new_steps.append(Step(**asdict(step)))

        return Plan(steps=new_steps, error_type="wrong_tool")

    def _inject_parameter_typo(self, plan: Plan) -> Plan:
        new_steps = []
        injected = False

        for i, step in enumerate(plan.steps):
            new_step = Step(**asdict(step))

            if not injected and i == 0:
                for param_name, param_value in list(new_step.parameters.items()):
                    if (
                        isinstance(param_value, str)
                        and len(param_value) > 3
                        and "{{" not in param_value
                    ):
                        pos = random.randint(0, len(param_value))
                        new_step.parameters[param_name] = (
                            param_value[:pos] + "x" + param_value[pos:]
                        )
                        injected = True
                        break

            new_steps.append(new_step)

        return Plan(steps=new_steps, error_type="parameter_typo")

    def _inject_circular_dependency(self, plan: Plan) -> Plan:
        if len(plan.steps) < 2:
            return Plan(
                steps=[Step(**asdict(step)) for step in plan.steps],
                error_type="circular_dependency",
            )

        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = Step(**asdict(step))
            if i == 0 and len(plan.steps) > 1 and new_step.parameters:
                param_name = list(new_step.parameters.keys())[0]
                new_step.parameters[param_name] = "{{1}}"
            new_steps.append(new_step)

        return Plan(steps=new_steps, error_type="circular_dependency")

    def _inject_inefficient_order(self, plan: Plan) -> Plan:
        if len(plan.steps) < 3:
            return Plan(
                steps=[Step(**asdict(step)) for step in plan.steps],
                error_type="inefficient_order",
            )

        new_steps = [Step(**asdict(step)) for step in plan.steps]
        new_steps[-1], new_steps[-2] = new_steps[-2], new_steps[-1]
        for i, step in enumerate(new_steps):
            step.step_id = i

        return Plan(steps=new_steps, error_type="inefficient_order")

    def _inject_incomplete_plan(self, plan: Plan) -> Plan:
        if len(plan.steps) <= 1:
            return Plan(
                steps=[Step(**asdict(step)) for step in plan.steps],
                error_type="incomplete_plan",
            )
        new_steps = [Step(**asdict(step)) for step in plan.steps[:-1]]
        return Plan(steps=new_steps, error_type="incomplete_plan")

    def _inject_unnecessary_steps(self, plan: Plan) -> Plan:
        new_steps = [Step(**asdict(step)) for step in plan.steps]

        if len(plan.steps) > 1:
            insert_pos = len(plan.steps) // 2
            redundant_step = Step(
                step_id=insert_pos,
                tool_name=plan.steps[0].tool_name,
                parameters=plan.steps[0].parameters.copy(),
                output_variable=f"{{{{{insert_pos}}}}}",
                expected_output=None,
            )
            new_steps.insert(insert_pos, redundant_step)

            for i in range(insert_pos + 1, len(new_steps)):
                new_steps[i].step_id = i
                for param_name, param_value in new_steps[i].parameters.items():
                    if isinstance(param_value, str):
                        def increment_ref(match):
                            ref_id = int(match.group(1))
                            if ref_id >= insert_pos:
                                return f"{{{{{ref_id + 1}}}}}"
                            return match.group(0)

                        new_steps[i].parameters[param_name] = re.sub(
                            r"\{\{(\d+)\}\}", increment_ref, param_value
                        )

        return Plan(steps=new_steps, error_type="unnecessary_steps")

    def _inject_forward_reference(self, plan: Plan) -> Plan:
        if len(plan.steps) < 2:
            return Plan(
                steps=[Step(**asdict(step)) for step in plan.steps],
                error_type="forward_reference",
            )

        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = Step(**asdict(step))
            if i == 1 and len(plan.steps) > 2 and new_step.parameters:
                param_name = list(new_step.parameters.keys())[0]
                new_step.parameters[param_name] = "{{2}}"
            new_steps.append(new_step)

        return Plan(steps=new_steps, error_type="forward_reference")


# ─── Judge annotator ──────────────────────────────────────────────────────────

class LLMJudgeAnnotator:
    """Uses LLM to annotate plans with quality scores. Tools dict must be API-name-keyed."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def annotate(
        self,
        query: str,
        tools_by_api_name: Dict,
        plan: Plan,
        ground_truth: Optional[Plan] = None,
    ) -> Annotation:
        prompt = self._build_annotation_prompt(query, tools_by_api_name, plan, ground_truth)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at evaluating multi-step tool execution "
                            "plans. You only check syntactic correctness, not semantic "
                            "correctness. NEVER type-check parameters containing {{N}} - "
                            "they are dependency references and are always type-valid."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=2000,
            )

            content = response.choices[0].message.content

            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()

            annotation_data = json.loads(json_str)
            return Annotation(**annotation_data)

        except Exception as e:
            print(f"Error annotating plan: {e}")
            return Annotation(
                quality_score=50,
                success_prediction="uncertain",
                reasoning=f"Error during annotation: {str(e)}",
                issues=[],
                confidence=0.0,
            )

    def _build_annotation_prompt(
        self,
        query: str,
        tools_by_api_name: Dict[str, Dict],
        plan: Plan,
        ground_truth: Optional[Plan],
    ) -> str:
        tools_desc = self._format_tools(tools_by_api_name)
        plan_desc = self._format_plan(plan)

        prompt = f"""You are evaluating a multi-step tool execution plan using an OBJECTIVE ERROR-BASED RUBRIC.

**USER QUERY:**
{query}

**AVAILABLE TOOLS:**
{tools_desc}

**PROPOSED PLAN:**
{plan_desc}

{"**GROUND TRUTH PLAN (for reference):**" + chr(10) + self._format_plan(ground_truth) if ground_truth else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBJECTIVE SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**SCORING FORMULA:**
1. Start with a base score of 100
2. Identify ALL errors in the plan
3. For EACH error, deduct points based on severity:
   - CRITICAL error: -30 points
   - HIGH error: -20 points
   - MEDIUM error: -10 points
   - LOW error: -5 points
4. Final score = max(0, 100 - total_deductions)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: YOU ONLY CHECK SYNTAX, NOT SEMANTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your job is to check SYNTACTIC correctness ONLY. You are NOT checking if the
plan will work correctly or produce the right answer.

**SYNTACTIC (what you SHOULD check):**
✅ Are all required parameters provided?
✅ Are parameter types correct for LITERAL values WITHOUT {{{{N}}}}?
   ⚠️  CRITICAL: NEVER EVER type-check values containing {{{{N}}}}!
   ⚠️  If you see {{{{N}}}}, skip type checking completely!
✅ Are dependencies valid (no forward refs, no circular deps)?
✅ Are all steps present to complete the task?
✅ Is the tool name spelled correctly?

**SEMANTIC (what you should NOT check):**
❌ Will the tool outputs actually work together?
❌ Is the output of tool A meaningful for tool B?
❌ Will the plan produce the correct final answer?
❌ Is the data flow logically sound?
❌ Does this tool make sense for this task?
❌ Will the parameter values produce the right result?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER FLAG THESE AS ERRORS (DEPENDENCY REFERENCES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**NEVER EVER flag these as type errors:**

❌ "numbers='{{{{2}}}}' is a string but tool expects array"
✅ CORRECT: "Parameter uses dependency reference - no type error"

❌ "input_number='{{{{1}}}}' is a string but tool expects integer"
✅ CORRECT: "Parameter uses dependency reference - no type error"

❌ "value='{{{{0}}}}' should be an array not a string"
✅ CORRECT: "Parameter uses dependency reference - no type error"

**ANY parameter containing {{{{N}}}} gets ZERO type checking!**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER FLAG THESE AS ERRORS (SEMANTIC ISSUES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ "sum_digits sums digits, not factors - wrong tool"
❌ "This tool won't produce the right output for the next step"
❌ "The plan uses the wrong approach for this query"
❌ "This parameter value won't give the correct answer"
❌ "The output type doesn't match what the next tool expects"
❌ "This won't answer the query correctly"

**Example of what NOT to flag:**

Query: "What is the sum of the prime factors?"
Plan:
  Step 2: {{{{2}}}} = prime_factorization(number='{{{{1}}}}')
  Step 3: {{{{3}}}} = sum_digits(number='{{{{2}}}}')

❌ WRONG: "sum_digits sums digits, not factors - this is wrong tool error"
✅ CORRECT: "All required params provided, valid dependency - no error"

Even though sum_digits is semantically wrong (it sums digits not factors),
it's SYNTACTICALLY VALID! The tool exists, all required parameters are
provided, and the dependency reference is valid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: Type Checking Rules for Dependency References
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**ABSOLUTE RULE: If parameter contains {{{{N}}}}, NEVER EVER check its type!**

Any parameter value that contains {{{{N}}}} is a dependency reference and is
ALWAYS type-valid. Period. No exceptions. The quotes and parameter type are
IRRELEVANT when {{{{N}}}} is present.

**These are ALL dependency references (ZERO type checking allowed):**
✅ number='{{{{1}}}}'     → Valid dependency (NOT a string!)
✅ number="{{{{1}}}}"     → Valid dependency (NOT a string!)
✅ number={{{{1}}}}       → Valid dependency (NOT missing quotes!)
✅ input_number='{{{{2}}}}' → Valid dependency (NOT a string!)
✅ numbers='{{{{0}}}}'    → Valid dependency (NOT a string!)
✅ array_param='{{{{1}}}}' → Valid dependency (NOT a string!)

**THE RULE:**
If you see {{{{N}}}} anywhere in the parameter value:
  1. DO NOT check the type
  2. DO NOT complain about quotes
  3. DO NOT flag any type mismatch
  4. ONLY check that N refers to a valid prior step

**Quotes DO NOT make {{{{N}}}} a "literal string"!**
The system will substitute {{{{N}}}} with the actual output at runtime.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ERROR SEVERITY DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL SEVERITY (-30 points each):
 • Missing required parameters (tool cannot execute)
 • Circular dependency ({{{{N}}}} references itself/cycle)
 • Non-existent tool used
 • Forward reference ({{{{N}}}} where N >= current step)
 • Type mismatch for LITERAL values WITHOUT {{{{N}}}}
 • Missing critical steps (plan cannot achieve goal)

HIGH SEVERITY (-20 points each):
 • Missing dependency: hardcoded value instead of {{{{N}}}}
 • Incorrect parameter value that changes output significantly
 • Missing non-critical but important step

MEDIUM SEVERITY (-10 points each):
 • Inefficient step ordering (works but suboptimal)
 • Typo in parameter value (might still work)
 • Unnecessary step that doesn't break the plan
 • Redundant computation

LOW SEVERITY (-5 points each):
 • Minor formatting issues
 • Non-standard but functional parameter format
 • Minor inefficiency that barely impacts performance
 • Style issues (but plan is correct)

**SUCCESS PREDICTION MAPPING:**
- 90-100: "yes"
- 75-89: "likely_yes"
- 50-74: "uncertain"
- 25-49: "likely_no"
- 0-24: "no"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **IDENTIFY ALL ERRORS**: Go through the plan step-by-step
2. **CLASSIFY SEVERITY**: Use the rubric above
3. **CALCULATE SCORE**: Start at 100, deduct points per error
4. **SHOW YOUR WORK**: In reasoning, list each error and deduction

**RESPONSE FORMAT (JSON only, no markdown):**

{{
  "quality_score": <0-100>,
  "success_prediction": "<yes|likely_yes|uncertain|likely_no|no>",
  "confidence": <0.0-1.0>,
  "reasoning": "Base score: 100\\nError 1: [severity] description (-X points)\\nError 2: [severity] description (-X points)\\n...\\nFinal score: 100 - X - Y = Z",
  "issues": [
    {{
      "type": "type_mismatch|missing_dependency|wrong_tool|parameter_typo|circular_dependency|inefficient_order|incomplete_plan|unnecessary_steps|forward_reference|other",
      "severity": "critical|high|medium|low",
      "step": <step_number or null>,
      "description": "Specific description of the error",
      "suggestion": "How to fix this error",
      "points_deducted": <30|20|10|5>
    }}
  ]
}}

**IMPORTANT SCORING RULES:**
1. If plan has 0 errors → Score = 95-100 (perfect plan)
2. Be consistent: Same error type = same severity = same deduction
3. Count EACH occurrence separately (3 type errors = 3 deductions)
4. **NEVER EVER flag parameters containing {{{{N}}}} as type errors!**
5. **NEVER flag semantic issues - only syntactic errors!**
6. **Only flag literal values for type errors** (values without {{{{N}}}})

Begin your analysis:"""

        return prompt

    def _format_tools(self, tools_by_api_name: Dict) -> str:
        """Format tools for display using API names as the canonical tool name."""
        formatted = []
        for api_name, tool_spec in tools_by_api_name.items():
            params = tool_spec.get("parameters", {})
            required = params.get("required", [])
            properties = params.get("properties", {})

            param_lines = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                is_req = " (REQUIRED)" if param_name in required else ""
                param_lines.append(f"    - {param_name}: {param_type}{is_req}")

            formatted.append(
                f"  {api_name}:\n"
                + (
                    f"    Required parameters: {', '.join(required)}\n"
                    if required
                    else ""
                )
                + "    Parameters:\n"
                + "\n".join(param_lines)
            )

        return "\n\n".join(formatted)

    def _format_plan(self, plan: Plan) -> str:
        if not plan:
            return "No plan provided"

        lines = []
        for step in plan.steps:
            params_str = ", ".join(
                f"{k}={repr(v)}" for k, v in step.parameters.items()
            )
            lines.append(
                f"Step {step.step_id}: {step.output_variable} = {step.tool_name}({params_str})"
            )

        return "\n".join(lines)


# ─── Dataset generator (entry point) ──────────────────────────────────────────

class DatasetGenerator:
    """Main dataset generator. Reindexes tools by API name once per query
    and passes the clean dict to every component.
    """

    def __init__(
        self,
        toolhop_path: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        n_candidates: int = 10,
    ):
        self.toolhop_path = toolhop_path
        self.n_candidates = n_candidates

        self.ground_truth_parser = GroundTruthParser(api_key, model)
        self.static_validator = None  # set per query
        self.plan_generator = None    # set per query
        self.llm_judge = LLMJudgeAnnotator(api_key, model)

        print("Loading ToolHop dataset...")
        with open(toolhop_path, "r") as f:
            self.toolhop_data = json.load(f)

    def generate(self, max_queries: Optional[int] = None) -> Dict:
        queries = self.toolhop_data[:max_queries] if max_queries else self.toolhop_data

        results = []

        print(f"Processing {len(queries)} queries...")
        for query_idx, example in enumerate(tqdm(queries, desc="Generating plans")):
            print("\n" + "=" * 80)
            print(f"Query {query_idx}: {example['question']}")
            print("=" * 80)

            # ★ Critical fix: reindex this query's tools by API name BEFORE
            #   passing them to any component.  Every downstream module now
            #   receives a dict where the keys are the actual API names
            #   (e.g. "geo_relationship_finder") rather than NL questions.
            tools_raw = example["tools"]
            tools = reindex_tools_by_api_name(tools_raw)

            # Sanity-check that we recovered API names successfully.
            if any(" " in name or name.endswith("?") for name in tools.keys()):
                print(
                    f"  ⚠  Warning: some tool names still look natural-language-like: "
                    f"{[n for n in tools.keys() if ' ' in n or n.endswith('?')]}"
                )

            self.static_validator = StaticValidator(tools)
            self.plan_generator = PlanGenerator(tools)

            # Generate ground truth plan
            print("\nGenerating ground truth plan with LLM...")
            ground_truth = self.ground_truth_parser.parse_toolhop_example(
                example, tools
            )

            print(f"\nGround truth plan ({len(ground_truth.steps)} steps):")
            for step in ground_truth.steps:
                params_str = ", ".join(
                    f"{k}={repr(v)}" for k, v in step.parameters.items()
                )
                print(
                    f"Step {step.step_id}: {step.output_variable} = "
                    f"{step.tool_name}({params_str})"
                )

            # Generate candidate plans
            print(f"\nGenerating {self.n_candidates} candidate plans...")
            candidates = self.plan_generator.generate_candidates(
                ground_truth, self.n_candidates
            )

            # Validate plans
            print("\nRunning static validation...")
            for i, plan in enumerate(candidates):
                is_valid, errors = self.static_validator.validate(plan)
                status = "✓ VALID" if is_valid else f"✗ INVALID ({len(errors)} errors)"
                print(f"  Plan {i} [{plan.error_type}]: {status}")
                if errors:
                    for error in errors[:3]:
                        print(f"    - {error}")

            # Annotate with LLM judge
            print("\nAnnotating plans with LLM judge...")
            for i, plan in enumerate(tqdm(candidates, desc="  ", leave=False)):
                annotation = self.llm_judge.annotate(
                    query=example["question"],
                    tools_by_api_name=tools,
                    plan=plan,
                    ground_truth=ground_truth,
                )

                results.append({
                    "query_id": query_idx,
                    "query": example["question"],
                    "plan": {
                        "steps": [asdict(step) for step in plan.steps],
                        "error_type": plan.error_type,
                    },
                    "annotation": asdict(annotation),
                })

                print(f"\n  Plan {i} [{plan.error_type}]:")
                print(f"    Quality: {annotation.quality_score}/100")
                print(f"    Success: {annotation.success_prediction}")
                print(f"    Issues: {len(annotation.issues)}")
                for issue in annotation.issues[:2]:
                    print(f"      - [{issue['severity']}] {issue['description']}")

        dataset = {
            "metadata": {
                "n_queries": len(queries),
                "n_candidates_per_query": self.n_candidates,
                "total_plans": len(results),
                "model": self.llm_judge.model,
                "tool_name_format": "api_name",  # explicit marker for downstream
            },
            "data": results,
        }

        return dataset

    def save_dataset(self, dataset: Dict, output_path: str):
        print(f"\nSaving dataset to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Dataset saved: {len(dataset['data'])} annotated plans")

    def print_statistics(self, dataset: Dict):
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        data = dataset["data"]

        by_error_type = {}
        for item in data:
            error_type = item["plan"]["error_type"]
            if error_type not in by_error_type:
                by_error_type[error_type] = []
            by_error_type[error_type].append(item["annotation"])

        print("\nError Type Distribution:")
        for error_type, annotations in sorted(by_error_type.items()):
            avg_score = sum(a["quality_score"] for a in annotations) / len(annotations)
            avg_issues = sum(len(a["issues"]) for a in annotations) / len(annotations)
            print(
                f"  {error_type:20s}: {len(annotations):3d} plans "
                f"(avg score: {avg_score:5.1f}, avg issues: {avg_issues:.1f})"
            )

        scores = [item["annotation"]["quality_score"] for item in data]
        print(f"\nQuality Score Statistics:")
        print(f"  Mean:   {sum(scores)/len(scores):.1f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.1f}")
        print(f"  Min:    {min(scores):.1f}")
        print(f"  Max:    {max(scores):.1f}")

        ranges = {
            "90-100 (Excellent)": lambda s: 90 <= s <= 100,
            "75-89 (Good)": lambda s: 75 <= s < 90,
            "50-74 (Fair)": lambda s: 50 <= s < 75,
            "25-49 (Poor)": lambda s: 25 <= s < 50,
            "0-24 (Critical)": lambda s: 0 <= s < 25,
        }

        print(f"\nScore Distribution:")
        for range_name, predicate in ranges.items():
            count = sum(1 for s in scores if predicate(s))
            pct = 100 * count / len(scores)
            print(f"  {range_name:20s}: {count:4d} ({pct:5.1f}%)")

        predictions = [item["annotation"]["success_prediction"] for item in data]
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        print(f"\nSuccess Prediction Distribution:")
        for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
            count = pred_counts.get(pred, 0)
            pct = 100 * count / len(predictions) if predictions else 0
            print(f"  {pred:12s}: {count:4d} ({pct:5.1f}%)")

        all_issues = [issue for item in data for issue in item["annotation"]["issues"]]
        print(f"\nIssue Statistics:")
        print(f"  Total issues detected: {len(all_issues)}")
        print(f"  Avg issues per plan: {len(all_issues)/len(data):.2f}")

        if all_issues:
            severity_counts = {}
            for issue in all_issues:
                sev = issue["severity"]
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            print(f"\n  By Severity:")
            for sev in ["critical", "high", "medium", "low"]:
                count = severity_counts.get(sev, 0)
                pct = 100 * count / len(all_issues) if all_issues else 0
                print(f"    {sev:8s}: {count:4d} ({pct:5.1f}%)")

            type_counts = {}
            for issue in all_issues:
                issue_type = issue.get("type", "other")
                type_counts[issue_type] = type_counts.get(issue_type, 0) + 1

            print(f"\n  By Type:")
            for issue_type in sorted(
                type_counts.keys(), key=lambda x: type_counts[x], reverse=True
            ):
                count = type_counts[issue_type]
                pct = 100 * count / len(all_issues)
                print(f"    {issue_type:20s}: {count:4d} ({pct:5.1f}%)")

        # Tool-name sanity check on the saved dataset itself
        all_tool_names = []
        for item in data:
            for s in item["plan"]["steps"]:
                all_tool_names.append(s["tool_name"])
        nl_looking = [t for t in all_tool_names if " " in t or t.endswith("?")]
        print(f"\nTool-Name Sanity Check:")
        print(f"  Total tool-name occurrences: {len(all_tool_names)}")
        print(
            f"  NL-looking tool names (contain spaces or '?'): "
            f"{len(nl_looking)} ({100*len(nl_looking)/max(1,len(all_tool_names)):.1f}%)"
        )
        if nl_looking:
            print(f"  ✗ WARNING: dataset still contains NL-looking tool names")
            print(f"    examples: {nl_looking[:3]}")
        else:
            print(f"  ✓ all tool names look like API identifiers")

        none_scores = [a["quality_score"] for a in by_error_type.get("none", [])]
        error_scores = [
            item["annotation"]["quality_score"]
            for item in data
            if item["plan"]["error_type"] != "none"
        ]

        if none_scores and error_scores:
            avg_none = sum(none_scores) / len(none_scores)
            avg_error = sum(error_scores) / len(error_scores)
            print(f"\nRubric Consistency Check:")
            print(f"  (Verifying that error_type='none' has highest scores)")
            print(f"    Ground truth avg: {avg_none:.1f}")
            print(f"    Error plans avg:  {avg_error:.1f}")
            if avg_none > avg_error:
                print(f"    ✓ Rubric working: Ground truth scored {avg_none - avg_error:.1f} points higher")
            else:
                print(f"    ✗ Rubric issue: Ground truth NOT scoring higher")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ToolHop annotated plans dataset with all 9 error types (API-name fix)"
    )
    parser.add_argument("--toolhop-path", required=True, help="Path to ToolHop dataset JSON")
    parser.add_argument("--output-path", required=True, help="Path to save annotated dataset")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use for generation")
    parser.add_argument(
        "--n-candidates", type=int, default=10, help="Number of candidate plans per query"
    )
    parser.add_argument(
        "--max-queries", type=int, default=None, help="Maximum number of queries to process"
    )

    args = parser.parse_args()

    generator = DatasetGenerator(
        toolhop_path=args.toolhop_path,
        api_key=args.api_key,
        model=args.model,
        n_candidates=args.n_candidates,
    )

    dataset = generator.generate(max_queries=args.max_queries)
    generator.save_dataset(dataset, args.output_path)
    generator.print_statistics(dataset)


if __name__ == "__main__":
    main()