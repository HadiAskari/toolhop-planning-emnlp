"""
NESTFUL Candidate Plan Generator — v2 (bug-fixed)

Drop-in replacement for nestful_annotator.py.  Only the PlanGenerator class
and the candidate-orchestration logic in DatasetGenerator have changed; the
schema adapter, static validator, and LLM judge prompt are byte-for-byte
identical to v1 so that judge models trained against v1 ToolHop data remain
compatible if you also regenerate ToolHop with the same fixes applied.

Bugs fixed (vs v1)
==================
A. _inject_wrong_tool: now prefers replacement tools whose required-param
   names are DISJOINT from the original step's params.  v1 randomly picked
   any other tool; ~22 % of injections landed on a same-arity, same-param-
   names replacement that the judge correctly evaluated as syntactically
   valid (score 100), silently contaminating the wrong_tool training class.

C. _inject_inefficient_order: now scans for adjacent independent pairs and
   swaps without renumbering step_ids.  v1 always swapped the last two steps
   and then renumbered, producing a forward-reference whenever the dropped
   step's output was used downstream — score 70 (critical FR) instead of 90
   (medium inefficient ordering).

D. _inject_type_mismatch: now uses the tool spec's declared type to pick a
   provably incompatible replacement value, covering dict/object, boolean,
   and any other literal type.  v1 had only three passes (numeric→string,
   list→string, string→int) and silently no-op'd on plans whose only
   non-dep-ref literal was a dict (e.g. make_json_string queries).

E. _inject_forward_reference: now handles 2-step plans (the v1 guard
   `len(plan.steps) > 2` skipped them entirely, producing a labeled-but-
   unmodified plan that scored 100).

F. Orchestration: when an injector cannot find a valid injection point
   (e.g. inefficient_order on a strict dependency chain), it now returns
   was_modified=False, and generate_candidates retries with a different
   error type.  v1 emitted "labeled X but actually unchanged" plans that
   scored 100 — a major source of label/score inconsistency.

Bug B (circular_dependency mislabeled as forward_reference by the judge) is
NOT fixed here — it would require a prompt change, and the score (-30
critical) is correct in either case.  The labeling collapse only affects
per-error-type analysis figures, not the trained judge's reward signal.

CLI
===
  # Smoke test (50 queries, ~$0.50)
  python nestful_annotator_v2.py \
      --nestful-path data_v2/nestful_data.jsonl \
      --output-path nestful_annotated_v2_smoke.json \
      --api-key sk-... \
      --max-queries 50

  # Self-test (no API calls; ~3s; verifies each injector produces a
  # syntactically valid, statically-detectable error)
  python nestful_annotator_v2.py --self-test

  # Full run
  python nestful_annotator_v2.py \
      --nestful-path data_v2/nestful_data.jsonl \
      --output-path nestful_annotated_v2.json \
      --api-key sk-...
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import openai
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Data structures (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────


class ErrorType(Enum):
    NONE                = "none"
    TYPE_MISMATCH       = "type_mismatch"
    MISSING_DEPENDENCY  = "missing_dependency"
    WRONG_TOOL          = "wrong_tool"
    PARAMETER_TYPO      = "parameter_typo"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    INEFFICIENT_ORDER   = "inefficient_order"
    INCOMPLETE_PLAN     = "incomplete_plan"
    UNNECESSARY_STEPS   = "unnecessary_steps"
    FORWARD_REFERENCE   = "forward_reference"


@dataclass
class Step:
    step_id: int
    tool_name: str
    parameters: Dict[str, Any]
    output_variable: str
    expected_output: Optional[str] = None

    def __str__(self):
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in self.parameters.items())
        return f"Step {self.step_id}: {self.output_variable} = {self.tool_name}({params_str})"


@dataclass
class Plan:
    steps: List[Step]
    error_type: str = "none"

    def to_dict(self):
        return {"steps": [asdict(s) for s in self.steps], "error_type": self.error_type}

    def __str__(self):
        return "\n".join(str(s) for s in self.steps)


@dataclass
class Annotation:
    quality_score: int
    success_prediction: str
    reasoning: str
    issues: List[Dict[str, Any]]
    confidence: float

    def to_dict(self):
        return asdict(self)


# Injector return type: (modified plan, did we actually inject the requested error?)
InjectorResult = Tuple[Plan, bool]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: recursive substring search inside nested params
# ──────────────────────────────────────────────────────────────────────────────


def _contains_ref(value: Any, ref: str) -> bool:
    """True if `ref` (e.g. '{{0}}') appears anywhere inside `value` recursively."""
    if isinstance(value, str):
        return ref in value
    if isinstance(value, list):
        return any(_contains_ref(v, ref) for v in value)
    if isinstance(value, dict):
        return any(_contains_ref(v, ref) for v in value.values())
    return False


def _is_dep_ref(value: Any) -> bool:
    """True if `value` is or contains a {{N}} dependency reference."""
    if isinstance(value, str):
        return bool(re.search(r"\{\{\d+\}\}", value))
    if isinstance(value, list):
        return any(_is_dep_ref(v) for v in value)
    if isinstance(value, dict):
        return any(_is_dep_ref(v) for v in value.values())
    return False


def _clone_steps(steps: List[Step]) -> List[Step]:
    return [Step(**asdict(s)) for s in steps]


# ──────────────────────────────────────────────────────────────────────────────
# NESTFUL schema normalizer (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────


class NestfulSchemaAdapter:
    _TYPE_MAP = {
        "int or float": "number",
        "int":          "integer",
        "float":        "number",
        "integer":      "integer",
        "string":       "string",
        "str":          "string",
        "bool":         "boolean",
        "boolean":      "boolean",
        "list":         "array",
        "array":        "array",
        "dict":         "object",
        "object":       "object",
    }

    @classmethod
    def _resolve_type(cls, raw_type: Any) -> str:
        if raw_type is None:
            return "string"
        if isinstance(raw_type, list):
            non_null = [t for t in raw_type if t != "null"]
            raw_type = non_null[0] if non_null else "string"
        if not isinstance(raw_type, str):
            return "string"
        return cls._TYPE_MAP.get(raw_type.lower().strip(), "string")

    @classmethod
    def _normalize_param(cls, pinfo: Any) -> Dict[str, Any]:
        # Some corpus tools carry malformed schemas where a property value is a
        # bare string (a description or type name) instead of a spec dict.
        if not isinstance(pinfo, dict):
            if isinstance(pinfo, str) and pinfo.lower().strip() in cls._TYPE_MAP:
                return {"type": cls._resolve_type(pinfo), "description": ""}
            return {"type": "string",
                    "description": pinfo if isinstance(pinfo, str) else ""}
        return {"type":        cls._resolve_type(pinfo.get("type")),
                "description": pinfo.get("description", "")}

    @classmethod
    def normalize_tools(cls, tools_list: List[Dict]) -> Dict[str, Dict]:
        tools_dict: Dict[str, Dict] = {}
        for tool in tools_list:
            name = tool["name"]
            raw_params = tool.get("parameters", {})

            if "properties" in raw_params:
                sc2_props    = raw_params.get("properties", {})
                sc2_required = raw_params.get("required", list(sc2_props.keys()))
                properties: Dict[str, Dict] = {
                    pname: cls._normalize_param(pinfo)
                    for pname, pinfo in sc2_props.items()
                }
                required_list = sc2_required
            else:
                properties = {
                    arg_name: cls._normalize_param(arg_info)
                    for arg_name, arg_info in raw_params.items()
                }
                required_list = list(properties.keys())

            tools_dict[name] = {
                "description": tool.get("description", ""),
                "parameters": {
                    "required":   required_list,
                    "properties": properties,
                },
            }
        return tools_dict

    @classmethod
    def normalize_plan(cls, output_list: List[Dict], gold_answer: Any) -> Plan:
        label_to_idx: Dict[str, int] = {}
        for i, call in enumerate(output_list):
            label = call.get("label", f"$var_{i+1}")
            label_to_idx[label] = i
            if re.search(r"\$var_\d+", label):
                alias = re.sub(r"\$var_(\d+)", r"$var\1", label)
            else:
                alias = re.sub(r"\$var(\d+)", r"$var_\1", label)
            label_to_idx.setdefault(alias, i)

        steps: List[Step] = []
        for i, call in enumerate(output_list):
            raw_args = call.get("arguments", {})
            norm_args = {
                k: cls._normalize_ref(v, label_to_idx)
                for k, v in raw_args.items()
            }

            for k, v in norm_args.items():
                if isinstance(v, str) and re.search(r"\$var_?\d+\.\w+\$", v):
                    print(f"  [WARN] Step {i}: could not resolve reference in "
                          f"'{k}={v}' — leaving as literal (data quality issue)")

            exp_out = str(gold_answer) if i == len(output_list) - 1 else None

            steps.append(Step(
                step_id=i,
                tool_name=call["name"],
                parameters=norm_args,
                output_variable=f"{{{{{i}}}}}",
                expected_output=exp_out,
            ))

        return Plan(steps=steps, error_type="none")

    @classmethod
    def _normalize_ref(cls, value: Any, label_to_idx: Dict[str, int]) -> Any:
        if isinstance(value, str):
            pattern = r"(\$var_?\d+)\.(\w+)\$"

            def replace_label(match):
                raw_label = match.group(1)
                idx = label_to_idx.get(raw_label)
                if idx is None:
                    if re.search(r"\$var_\d+", raw_label):
                        alias = re.sub(r"\$var_(\d+)", r"$var\1", raw_label)
                    else:
                        alias = re.sub(r"\$var(\d+)", r"$var_\1", raw_label)
                    idx = label_to_idx.get(alias)
                if idx is not None:
                    return f"{{{{{idx}}}}}"
                return match.group(0)

            return re.sub(pattern, replace_label, value)
        elif isinstance(value, list):
            return [cls._normalize_ref(v, label_to_idx) for v in value]
        elif isinstance(value, dict):
            return {k: cls._normalize_ref(v, label_to_idx) for k, v in value.items()}
        return value


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth parser (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────


class GroundTruthParser:
    def parse_nestful_example(self, example: Dict) -> Plan:
        plan = NestfulSchemaAdapter.normalize_plan(
            output_list=example["output"],
            gold_answer=example.get("gold_answer", ""),
        )
        return plan


# ──────────────────────────────────────────────────────────────────────────────
# Static validator (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────


class StaticValidator:
    def __init__(self, tools: Dict):
        self.tools = tools

    def validate(self, plan: Plan) -> Tuple[bool, List[str]]:
        errors = []

        for step in plan.steps:
            if step.tool_name not in self.tools:
                errors.append(f"Step {step.step_id}: Unknown tool '{step.tool_name}'")
                continue

            tool_spec = self.tools[step.tool_name]
            params_spec = tool_spec.get("parameters", {})
            required    = params_spec.get("required", [])
            properties  = params_spec.get("properties", {})

            for req in required:
                if req not in step.parameters:
                    errors.append(f"Step {step.step_id}: Missing required parameter '{req}'")

            for param_name, param_value in step.parameters.items():
                if param_name not in properties:
                    continue
                expected_type = properties[param_name].get("type")
                if expected_type and not self._is_dependency_ref(param_value):
                    if not self._check_type(param_value, expected_type):
                        errors.append(
                            f"Step {step.step_id}: Parameter '{param_name}' should be "
                            f"{expected_type}, got {type(param_value).__name__}"
                        )

        errors.extend(self._check_dependencies(plan))

        circ = self._check_circular_dependencies(plan)
        if circ:
            errors.append(f"Circular dependency detected: {circ}")

        return len(errors) == 0, errors

    def _is_dependency_ref(self, value) -> bool:
        if isinstance(value, str):
            return "{{" in value and "}}" in value
        if isinstance(value, list):
            return any(self._is_dependency_ref(v) for v in value)
        return False

    def _check_type(self, value, expected_type: str) -> bool:
        type_map = {
            "string":  str,
            "number":  (int, float),
            "integer": int,
            "boolean": bool,
            "array":   list,
            "object":  dict,
        }
        expected = type_map.get(expected_type)
        return True if expected is None else isinstance(value, expected)

    def _check_dependencies(self, plan: Plan) -> List[str]:
        errors = []
        max_id = max((s.step_id for s in plan.steps), default=-1)

        for step in plan.steps:
            for val in step.parameters.values():
                if isinstance(val, str):
                    for ref in re.findall(r"\{\{(\d+)\}\}", val):
                        ref_id = int(ref)
                        if ref_id >= step.step_id:
                            errors.append(
                                f"Step {step.step_id}: Invalid forward reference to step {ref_id}"
                            )
                        if ref_id > max_id:
                            errors.append(
                                f"Step {step.step_id}: Reference to non-existent step {ref_id}"
                            )
        return errors

    def _check_circular_dependencies(self, plan: Plan) -> Optional[str]:
        graph: Dict[int, List[int]] = defaultdict(list)
        for step in plan.steps:
            for val in step.parameters.values():
                if isinstance(val, str):
                    graph[step.step_id].extend(
                        int(r) for r in re.findall(r"\{\{(\d+)\}\}", val)
                    )

        def has_cycle(node, visited, stack):
            visited.add(node)
            stack.add(node)
            for nb in graph.get(node, []):
                if nb not in visited:
                    if has_cycle(nb, visited, stack):
                        return True
                elif nb in stack:
                    return True
            stack.remove(node)
            return False

        visited: set = set()
        for step in plan.steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id, visited, set()):
                    return f"Cycle involving step {step.step_id}"
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Plan generator (v2 — all 5 injection-logic fixes applied)
# ──────────────────────────────────────────────────────────────────────────────


class PlanGenerator:
    """Generates candidate plans with error injection. Each injector returns
    (plan, was_modified) so the orchestrator can retry with a different error
    type when an injection isn't applicable."""

    # Maps a tool's declared param type to a literal of incompatible type.
    # Used by _inject_type_mismatch to produce a detectable mismatch regardless
    # of whether the literal was originally a dict, bool, list, etc.
    _INCOMPATIBLE_LITERAL = {
        "number":  "invalid_string",
        "integer": "invalid_string",
        "string":  42,
        "array":   "invalid_string",
        "object":  "invalid_string",
        "boolean": "invalid_string",
    }

    def __init__(self, tools: Dict):
        self.tools = tools

    # ── public API ──────────────────────────────────────────────────────────

    def generate_candidates(
        self,
        ground_truth: Plan,
        n_candidates: int = 10,
        error_distribution: Optional[Dict[str, float]] = None,
    ) -> List[Plan]:
        if error_distribution is None:
            error_distribution = {
                "none":                0.15,
                "type_mismatch":       0.10,
                "missing_dependency":  0.10,
                "wrong_tool":          0.15,
                "parameter_typo":      0.10,
                "circular_dependency": 0.05,
                "inefficient_order":   0.10,
                "incomplete_plan":     0.10,
                "unnecessary_steps":   0.10,
                "forward_reference":   0.05,
            }

        candidates = [Plan(steps=_clone_steps(ground_truth.steps), error_type="none")]
        remaining = n_candidates - 1
        error_counts = {
            et: max(1, int(remaining * p))
            for et, p in error_distribution.items()
            if et != "none"
        }
        error_types_list = [e for e in error_distribution if e != "none"]

        for error_type, count in error_counts.items():
            for _ in range(count):
                if len(candidates) >= n_candidates:
                    break
                plan = self._inject_with_retry(ground_truth, error_type, error_types_list)
                candidates.append(plan)

        # Top up if quotas didn't fill all slots
        while len(candidates) < n_candidates:
            ft = random.choice(error_types_list)
            plan = self._inject_with_retry(ground_truth, ft, error_types_list)
            candidates.append(plan)

        return candidates[:n_candidates]

    def _inject_with_retry(
        self,
        plan: Plan,
        primary_error_type: str,
        all_error_types: List[str],
    ) -> Plan:
        """Try to inject `primary_error_type`; if that injector reports it
        couldn't apply to this plan, fall back to other types in random order.
        Final fallback: return the unmodified plan as 'none' (clean labeling)."""
        injected, ok = self._inject_error(plan, primary_error_type)
        if ok:
            return injected
        # Retry with shuffled fallback types
        for ft in random.sample(all_error_types, len(all_error_types)):
            if ft == primary_error_type:
                continue
            injected, ok = self._inject_error(plan, ft)
            if ok:
                return injected
        # All injections failed (extreme degenerate case) — emit clean GT plan
        return Plan(steps=_clone_steps(plan.steps), error_type="none")

    def _inject_error(self, plan: Plan, error_type: str) -> InjectorResult:
        dispatch = {
            "type_mismatch":       self._inject_type_mismatch,
            "missing_dependency":  self._inject_missing_dependency,
            "wrong_tool":          self._inject_wrong_tool,
            "parameter_typo":      self._inject_parameter_typo,
            "circular_dependency": self._inject_circular_dependency,
            "inefficient_order":   self._inject_inefficient_order,
            "incomplete_plan":     self._inject_incomplete_plan,
            "unnecessary_steps":   self._inject_unnecessary_steps,
            "forward_reference":   self._inject_forward_reference,
        }
        fn = dispatch.get(error_type)
        if not fn:
            return Plan(steps=_clone_steps(plan.steps), error_type="none"), True
        return fn(plan)

    # ── individual injectors ────────────────────────────────────────────────

    def _inject_type_mismatch(self, plan: Plan) -> InjectorResult:
        """Bug D fix: walk all steps, find any non-dep-ref literal, and replace
        with a value whose type is provably incompatible with the tool's
        declared param type.  Handles dict, bool, list, string, number alike."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n == 0:
            return Plan(steps=new_steps, error_type="none"), False
        mid = n // 2
        order = sorted(range(n), key=lambda i: abs(i - mid))

        for i in order:
            step = new_steps[i]
            tool_spec = self.tools.get(step.tool_name, {})
            props = tool_spec.get("parameters", {}).get("properties", {})
            for pname, pval in list(step.parameters.items()):
                if _is_dep_ref(pval):
                    continue
                param_type = props.get(pname, {}).get("type", "string")
                replacement = self._INCOMPATIBLE_LITERAL.get(param_type, "invalid_string")
                # Make sure replacement is actually different in type from current value.
                # (Edge: tool spec says "string" and current value is already a string —
                # replacing with 42 still creates a mismatch.)
                step.parameters[pname] = replacement
                return Plan(steps=new_steps, error_type="type_mismatch"), True

        return Plan(steps=new_steps, error_type="none"), False

    def _inject_missing_dependency(self, plan: Plan) -> InjectorResult:
        """Replace a {{N}} dep ref with a hardcoded literal."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n < 2:
            return Plan(steps=new_steps, error_type="none"), False
        mid = n // 2
        order = sorted(range(1, n), key=lambda i: abs(i - mid))

        for i in order:
            for pname, pval in list(new_steps[i].parameters.items()):
                if isinstance(pval, str) and "{{" in pval:
                    new_steps[i].parameters[pname] = "hardcoded_value"
                    return Plan(steps=new_steps, error_type="missing_dependency"), True

        return Plan(steps=new_steps, error_type="none"), False

    def _inject_wrong_tool(self, plan: Plan) -> InjectorResult:
        """Bug A fix: prefer replacement tools whose required-param names are
        DISJOINT from the original step's params.  This forces the judge to
        see missing-required-param + unknown-extra-params, which it labels
        as wrong_tool with score -30 critical.

        Avoids: tools whose required-param signature exactly matches the
        original — those produce score 100 because the judge sees no
        syntactic error (only semantic, which the rubric forbids it from
        flagging)."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n < 1 or len(self.tools) < 2:
            return Plan(steps=new_steps, error_type="none"), False

        mid = n // 2
        target = new_steps[mid]
        orig_params = set(target.parameters.keys())
        orig_tool = target.tool_name

        disjoint: List[str] = []         # best signal: judge labels wrong_tool
        partial_overlap: List[str] = []  # OK: judge labels "other" but score correct
        full_match: List[str] = []       # avoid: judge sees no syntactic error

        for tname, spec in self.tools.items():
            if tname == orig_tool:
                continue
            required = set(spec.get("parameters", {}).get("required", []))
            if not required:
                # Tool with no required params accepts anything → judge sees
                # only "unknown extra params" which it may downplay.  Treat
                # as full_match to deprioritize.
                full_match.append(tname)
                continue
            if required == orig_params:
                full_match.append(tname)
            elif required & orig_params:
                partial_overlap.append(tname)
            else:
                disjoint.append(tname)

        pool = disjoint or partial_overlap or full_match
        if not pool:
            return Plan(steps=new_steps, error_type="none"), False
        target.tool_name = random.choice(pool)
        return Plan(steps=new_steps, error_type="wrong_tool"), True

    def _inject_parameter_typo(self, plan: Plan) -> InjectorResult:
        """Subtle perturbation: insert a character in a string literal, or
        ±1 a numeric literal.  Stays syntactically valid (judge scores 100,
        which IS correct per the syntax-only rubric).  This injector exists
        to populate the parameter_typo class with semantically-wrong but
        syntactically-clean plans."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n == 0:
            return Plan(steps=new_steps, error_type="none"), False

        # Try string literals on step 0 first (StarCoder2 style)
        step0 = new_steps[0]
        for pname, pval in list(step0.parameters.items()):
            if isinstance(pval, str) and len(pval) > 3 and "{{" not in pval:
                pos = random.randint(0, len(pval))
                step0.parameters[pname] = pval[:pos] + "x" + pval[pos:]
                return Plan(steps=new_steps, error_type="parameter_typo"), True
            if isinstance(pval, (int, float)) and not isinstance(pval, bool):
                if isinstance(pval, int):
                    step0.parameters[pname] = pval + random.choice([-1, 1])
                else:
                    delta = max(0.1, abs(pval) * 0.1)
                    step0.parameters[pname] = round(pval + random.choice([-delta, delta]), 6)
                return Plan(steps=new_steps, error_type="parameter_typo"), True

        # Walk other steps if step 0 only has dep refs
        for i in range(1, n):
            for pname, pval in list(new_steps[i].parameters.items()):
                if isinstance(pval, (int, float)) and not isinstance(pval, bool):
                    if isinstance(pval, int):
                        new_steps[i].parameters[pname] = pval + random.choice([-1, 1])
                    else:
                        delta = max(0.1, abs(pval) * 0.1)
                        new_steps[i].parameters[pname] = round(
                            pval + random.choice([-delta, delta]), 6
                        )
                    return Plan(steps=new_steps, error_type="parameter_typo"), True

        return Plan(steps=new_steps, error_type="none"), False

    def _inject_circular_dependency(self, plan: Plan) -> InjectorResult:
        """Inject a cycle.  Strategy: find (i, t) with i < t where step t
        already references step i (back-edge), then add forward edge i → t
        to close the loop.  Falls back to a self-reference on the last step
        if no back-edge can be exploited (e.g. 2-step plans).

        Note: the LLM judge often labels the resulting structure as
        forward_reference rather than circular_dependency — this is a known
        labeling issue that does NOT affect the score (-30 critical in either
        case) and we accept it to preserve byte-identical prompt vs ToolHop."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n < 2:
            return Plan(steps=new_steps, error_type="none"), False

        # Look for (i, t) with i < t and step t already refs step i
        for i in range(n - 1):
            if not new_steps[i].parameters:
                continue
            i_var = f"{{{{{i}}}}}"
            for t in range(i + 1, n):
                if _contains_ref(new_steps[t].parameters, i_var):
                    # Add forward edge i → t
                    pname = list(new_steps[i].parameters.keys())[0]
                    new_steps[i].parameters[pname] = f"{{{{{t}}}}}"
                    return Plan(steps=new_steps, error_type="circular_dependency"), True

        # Fallback: self-reference on last step with parameters
        for i in range(n - 1, -1, -1):
            if new_steps[i].parameters:
                pname = list(new_steps[i].parameters.keys())[0]
                new_steps[i].parameters[pname] = f"{{{{{i}}}}}"
                return Plan(steps=new_steps, error_type="circular_dependency"), True

        return Plan(steps=new_steps, error_type="none"), False

    def _inject_inefficient_order(self, plan: Plan) -> InjectorResult:
        """Bug C fix: scan adjacent positions (i, i+1) for a pair where the
        step at position i+1 does NOT reference the step at position i's
        output.  Swap LIST POSITIONS without renumbering step_ids — this
        produces a plan whose printed listing has step_ids out of order,
        which the judge correctly interprets as inefficient ordering rather
        than as a forward reference.

        v1 always swapped the last two positions and renumbered, which
        introduced a forward reference whenever the dropped step's output
        was used downstream — the judge then scored these as critical FR
        (70) instead of medium inefficient_order (90)."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n < 2:
            return Plan(steps=new_steps, error_type="none"), False

        for i in range(n - 1):
            step_i_var = new_steps[i].output_variable
            if _contains_ref(new_steps[i + 1].parameters, step_i_var):
                continue  # would create FR — skip
            # Independent! Swap list positions; leave step_ids alone.
            new_steps[i], new_steps[i + 1] = new_steps[i + 1], new_steps[i]
            return Plan(steps=new_steps, error_type="inefficient_order"), True

        # Strict dependency chain — no valid reorder exists.
        return Plan(steps=new_steps, error_type="none"), False

    def _inject_incomplete_plan(self, plan: Plan) -> InjectorResult:
        """Drop the final step."""
        if len(plan.steps) <= 1:
            return Plan(steps=_clone_steps(plan.steps), error_type="none"), False
        truncated = _clone_steps(plan.steps[:-1])
        return Plan(steps=truncated, error_type="incomplete_plan"), True

    def _inject_unnecessary_steps(self, plan: Plan) -> InjectorResult:
        """Insert a redundant computation in the middle, renumbering downstream
        step_ids and dependency refs."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n < 2:
            return Plan(steps=new_steps, error_type="none"), False

        ins = n // 2
        redundant = Step(
            step_id=ins,
            tool_name=plan.steps[0].tool_name,
            parameters=dict(plan.steps[0].parameters),
            output_variable=f"{{{{{ins}}}}}",
            expected_output=None,
        )
        new_steps.insert(ins, redundant)

        for i in range(ins + 1, len(new_steps)):
            new_steps[i].step_id = i
            new_steps[i].output_variable = f"{{{{{i}}}}}"
            for pname, pval in new_steps[i].parameters.items():
                if isinstance(pval, str):
                    def inc(m, _ins=ins):
                        rid = int(m.group(1))
                        return f"{{{{{rid + 1}}}}}" if rid >= _ins else m.group(0)
                    new_steps[i].parameters[pname] = re.sub(
                        r"\{\{(\d+)\}\}", inc, pval
                    )
        return Plan(steps=new_steps, error_type="unnecessary_steps"), True

    def _inject_forward_reference(self, plan: Plan) -> InjectorResult:
        """Bug E fix: handle 2-step plans.  v1 required len > 2, silently
        no-op'ing on 2-step plans and emitting unmodified plans labeled FR.

        Strategy: find the earliest step with at least one parameter, and
        replace its first param with a reference to the next step.  Works
        for any plan with len >= 2."""
        new_steps = _clone_steps(plan.steps)
        n = len(new_steps)
        if n < 2:
            return Plan(steps=new_steps, error_type="none"), False

        for i in range(n - 1):
            if not new_steps[i].parameters:
                continue
            future_id = i + 1
            pname = list(new_steps[i].parameters.keys())[0]
            new_steps[i].parameters[pname] = f"{{{{{future_id}}}}}"
            return Plan(steps=new_steps, error_type="forward_reference"), True

        return Plan(steps=new_steps, error_type="none"), False


# ──────────────────────────────────────────────────────────────────────────────
# LLM judge annotator (prompt UNCHANGED from v1 to preserve cross-dataset
# compatibility for OOD experiments)
# ──────────────────────────────────────────────────────────────────────────────


class LLMJudgeAnnotator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def annotate(
        self,
        query: str,
        tools: Dict,
        plan: Plan,
        ground_truth: Optional[Plan] = None,
    ) -> Annotation:
        prompt = self._build_annotation_prompt(query, tools, plan, ground_truth)
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
            data = json.loads(json_str)
            return Annotation(**data)
        except Exception as e:
            print(f"Error annotating plan: {e}")
            return Annotation(
                quality_score=50,
                success_prediction="uncertain",
                reasoning=f"Error during annotation: {e}",
                issues=[],
                confidence=0.0,
            )

    def _build_annotation_prompt(
        self,
        query: str,
        tools: Dict,
        plan: Plan,
        ground_truth: Optional[Plan],
    ) -> str:
        tools_desc = self._format_tools(tools)
        plan_desc  = self._format_plan(plan)

        return f"""You are evaluating a multi-step tool execution plan using an OBJECTIVE ERROR-BASED RUBRIC.

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: Type Checking Rules for Dependency References
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**ABSOLUTE RULE: If parameter contains {{{{N}}}}, NEVER EVER check its type!**

Any parameter value that contains {{{{N}}}} is a dependency reference and is 
ALWAYS type-valid. Period. No exceptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ERROR SEVERITY DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────┐
│ CRITICAL SEVERITY (-30 points each)                             │
├─────────────────────────────────────────────────────────────────┤
│ • Missing required parameters (tool cannot execute)             │
│ • Circular dependency ({{{{N}}}} references itself/cycle)           │
│ • Non-existent tool used                                        │
│ • Forward reference ({{{{N}}}} where N >= current step)             │
│ • Type mismatch for LITERAL values WITHOUT {{{{N}}}}              │
│ • Missing critical steps (plan cannot achieve goal)             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ HIGH SEVERITY (-20 points each)                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Missing dependency: hardcoded value instead of {{{{N}}}}          │
│ • Incorrect parameter value that changes output significantly   │
│ • Missing non-critical but important step                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ MEDIUM SEVERITY (-10 points each)                               │
├─────────────────────────────────────────────────────────────────┤
│ • Inefficient step ordering (works but suboptimal)              │
│ • Typo in parameter value (might still work)                    │
│ • Unnecessary step that doesn't break the plan                  │
│ • Redundant computation                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LOW SEVERITY (-5 points each)                                   │
├─────────────────────────────────────────────────────────────────┤
│ • Minor formatting issues                                       │
│ • Non-standard but functional parameter format                  │
│ • Minor inefficiency that barely impacts performance            │
│ • Style issues (but plan is correct)                            │
└─────────────────────────────────────────────────────────────────┘

**SUCCESS PREDICTION MAPPING:**
- 90-100: "yes"
- 75-89:  "likely_yes"
- 50-74:  "uncertain"
- 25-49:  "likely_no"
- 0-24:   "no"

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
  "reasoning": "Base score: 100\\nError 1: [severity] description (-X points)\\n...\\nFinal score: Z",
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
1. If plan has 0 errors → Score = 95-100
2. Same error type = same severity = same deduction
3. Count EACH occurrence separately
4. **NEVER flag parameters containing {{{{N}}}} as type errors!**
5. **NEVER flag semantic issues — only syntactic errors!**
6. **Only flag literal values for type errors**

Begin your analysis:"""

    def _format_tools(self, tools: Dict) -> str:
        lines = []
        for tname, spec in tools.items():
            params   = spec.get("parameters", {})
            required = params.get("required", [])
            props    = params.get("properties", {})
            param_lines = [
                f"    - {pn}: {pi.get('type','any')}"
                + (" (REQUIRED)" if pn in required else "")
                for pn, pi in props.items()
            ]
            lines.append(
                f"  {tname}:\n"
                + (f"    Required parameters: {', '.join(required)}\n" if required else "")
                + "    Parameters:\n"
                + "\n".join(param_lines)
            )
        return "\n\n".join(lines)

    def _format_plan(self, plan: Optional[Plan]) -> str:
        if not plan:
            return "No plan provided"
        return "\n".join(
            f"Step {s.step_id}: {s.output_variable} = "
            f"{s.tool_name}({', '.join(f'{k}={repr(v)}' for k, v in s.parameters.items())})"
            for s in plan.steps
        )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset generator (unchanged orchestration; metadata bumped to v2)
# ──────────────────────────────────────────────────────────────────────────────


class DatasetGenerator:
    def __init__(
        self,
        nestful_path: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        n_candidates: int = 10,
    ):
        self.nestful_path  = nestful_path
        self.n_candidates  = n_candidates
        self.gt_parser     = GroundTruthParser()
        self.llm_judge     = LLMJudgeAnnotator(api_key, model)

        print("Loading NESTFUL dataset...")
        self.nestful_data: List[Dict] = []
        with open(nestful_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.nestful_data.append(json.loads(line))
        print(f"  Loaded {len(self.nestful_data)} samples.")

    def generate(self, max_queries: Optional[int] = None) -> Dict:
        samples = self.nestful_data[:max_queries] if max_queries else self.nestful_data
        results = []
        skipped_bad_spec = 0
        skipped_sample_ids: List[str] = []

        print(f"\nProcessing {len(samples)} queries...")
        for idx, example in enumerate(tqdm(samples, desc="Generating plans")):
            tools_norm = NestfulSchemaAdapter.normalize_tools(example["tools"])
            ground_truth = self.gt_parser.parse_nestful_example(example)

            gt_validator = StaticValidator(tools_norm)
            gt_ok, gt_errs = gt_validator.validate(ground_truth)
            if not gt_ok:
                skipped_bad_spec += 1
                skipped_sample_ids.append(example["sample_id"])
                continue

            plan_gen   = PlanGenerator(tools_norm)
            candidates = plan_gen.generate_candidates(ground_truth, self.n_candidates)

            for i, plan in enumerate(candidates):
                ann = self.llm_judge.annotate(
                    query=example["input"],
                    tools=tools_norm,
                    plan=plan,
                    ground_truth=ground_truth,
                )
                results.append({
                    "query_id":   idx,
                    "sample_id":  example["sample_id"],
                    "query":      example["input"],
                    "gold_answer": example.get("gold_answer"),
                    "plan": {
                        "steps":      [asdict(s) for s in plan.steps],
                        "error_type": plan.error_type,
                    },
                    "annotation": asdict(ann),
                })

        n_processed = len(samples) - skipped_bad_spec

        return {
            "metadata": {
                "dataset":                "NESTFUL",
                "annotator_version":      "v2",
                "annotator_fixes":        ["A_wrong_tool_disjoint_pool",
                                           "C_inefficient_order_no_FR",
                                           "D_type_mismatch_dict_handling",
                                           "E_forward_reference_2step",
                                           "F_failed_injection_retry"],
                "n_queries":              n_processed,
                "n_candidates_per_query": self.n_candidates,
                "total_plans":            len(results),
                "model":                  self.llm_judge.model,
                "ref_syntax":             "{{N}} (normalized from $var_N.result$)",
                "skipped_bad_spec":       skipped_bad_spec,
                "skipped_sample_ids":     skipped_sample_ids,
            },
            "data": results,
        }

    def save_dataset(self, dataset: Dict, output_path: str):
        print(f"\nSaving to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Saved {len(dataset['data'])} annotated plans.")

    def print_statistics(self, dataset: Dict):
        print("\n" + "=" * 80)
        print("VALIDATION REPORT (v2)")
        print("=" * 80)

        data = dataset["data"]
        if not data:
            print("No data to report on.")
            return

        by_type: Dict[str, List] = defaultdict(list)
        for item in data:
            by_type[item["plan"]["error_type"]].append(item["annotation"])

        print("\nError Type Distribution:")
        for et, anns in sorted(by_type.items()):
            avg_s = sum(a["quality_score"] for a in anns) / len(anns)
            avg_i = sum(len(a["issues"]) for a in anns) / len(anns)
            print(f"  {et:25s}: {len(anns):4d} plans  "
                  f"avg_score={avg_s:5.1f}  avg_issues={avg_i:.1f}")

        # v2-specific sanity check: no error_type should have avg_score > 95
        # (except 'none' and 'parameter_typo', which are syntactically clean
        # by design).  This catches regressions where injection silently fails.
        SYNTAX_CLEAN = {"none", "parameter_typo"}
        suspicious = [(et, sum(a["quality_score"] for a in anns) / len(anns))
                      for et, anns in by_type.items()
                      if et not in SYNTAX_CLEAN and anns
                      and sum(a["quality_score"] for a in anns) / len(anns) > 95]
        if suspicious:
            print("\n⚠️  WARNING: error types with avg score > 95 (possible silent injection failure):")
            for et, s in suspicious:
                print(f"     {et}: avg {s:.1f}")
        else:
            print("\n✓ No silent-injection-failure regression detected.")

        scores = [d["annotation"]["quality_score"] for d in data]
        print(f"\nQuality Score Statistics:")
        print(f"  Mean:   {sum(scores)/len(scores):.1f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.1f}")
        print(f"  Min:    {min(scores)}")
        print(f"  Max:    {max(scores)}")

        none_scores  = [a["quality_score"] for a in by_type.get("none", [])]
        error_scores = [d["annotation"]["quality_score"]
                        for d in data if d["plan"]["error_type"] != "none"]
        if none_scores and error_scores:
            avg_none  = sum(none_scores)  / len(none_scores)
            avg_error = sum(error_scores) / len(error_scores)
            print(f"\nRubric Sanity Check:")
            print(f"  Ground truth avg : {avg_none:.1f}")
            print(f"  Error plans avg  : {avg_error:.1f}")
            marker = "✓" if avg_none > avg_error else "✗"
            print(f"  {marker} GT > error by {avg_none - avg_error:+.1f} pts")

        print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Self-tests (no API calls)
# ──────────────────────────────────────────────────────────────────────────────


def _make_test_tools() -> Dict:
    """Synthetic tool catalog covering all type categories and arities."""
    return {
        "add":       {"parameters": {"required": ["arg_0", "arg_1"],
                                      "properties": {"arg_0": {"type": "number"},
                                                     "arg_1": {"type": "number"}}}},
        "multiply":  {"parameters": {"required": ["arg_0", "arg_1"],
                                      "properties": {"arg_0": {"type": "number"},
                                                     "arg_1": {"type": "number"}}}},
        "negate":    {"parameters": {"required": ["arg_0"],
                                      "properties": {"arg_0": {"type": "number"}}}},
        "make_json": {"parameters": {"required": ["dictionary"],
                                      "properties": {"dictionary": {"type": "object"}}}},
        "tokenize":  {"parameters": {"required": ["sentence"],
                                      "properties": {"sentence": {"type": "string"}}}},
        "find_kth":  {"parameters": {"required": ["nums", "k"],
                                      "properties": {"nums": {"type": "array"},
                                                     "k":    {"type": "integer"}}}},
        "swap_idx":  {"parameters": {"required": ["values", "i", "j"],
                                      "properties": {"values": {"type": "array"},
                                                     "i":      {"type": "integer"},
                                                     "j":      {"type": "integer"}}}},
    }


def _make_test_plan_3step() -> Plan:
    return Plan(steps=[
        Step(0, "add",      {"arg_0": 1, "arg_1": 2},        "{{0}}"),
        Step(1, "multiply", {"arg_0": "{{0}}", "arg_1": 3},  "{{1}}"),
        Step(2, "add",      {"arg_0": "{{1}}", "arg_1": 5},  "{{2}}", "8"),
    ])


def _make_test_plan_2step_dict() -> Plan:
    """Replicates the make_json_string + tokenize_sentence shape that
    silently no-op'd in v1's type_mismatch injector."""
    return Plan(steps=[
        Step(0, "make_json", {"dictionary": {"a": 1, "b": 2}}, "{{0}}"),
        Step(1, "tokenize",  {"sentence": "{{0}}"},            "{{1}}", "..."),
    ])


def _make_test_plan_4step_independent() -> Plan:
    """Has an independent adjacent pair (steps 2 & 3) — inefficient_order
    should swap them safely."""
    return Plan(steps=[
        Step(0, "add",      {"arg_0": 30, "arg_1": 50},        "{{0}}"),
        Step(1, "multiply", {"arg_0": "{{0}}", "arg_1": 10},   "{{1}}"),
        Step(2, "multiply", {"arg_0": "{{1}}", "arg_1": 100},  "{{2}}"),
        Step(3, "multiply", {"arg_0": 10, "arg_1": 25},        "{{3}}"),
        Step(4, "multiply", {"arg_0": "{{2}}", "arg_1": "{{3}}"}, "{{4}}", "..."),
    ])


def run_self_tests() -> int:
    """Returns 0 on success, nonzero on failure."""
    tools = _make_test_tools()
    pg = PlanGenerator(tools)
    sv = StaticValidator(tools)
    failures: List[str] = []

    def check(cond: bool, msg: str):
        if not cond:
            failures.append(msg)
            print(f"  ✗ {msg}")
        else:
            print(f"  ✓ {msg}")

    # ── Bug A: wrong_tool prefers disjoint param-name pool ──
    print("\n[Test A] wrong_tool injection")
    plan = _make_test_plan_3step()
    for _ in range(20):  # try multiple times since random.choice is involved
        injected, ok = pg._inject_wrong_tool(plan)
        check(ok, "wrong_tool returns ok=True")
        # Mid step (step_id=1) was multiply; check it changed
        new_tool = injected.steps[1].tool_name
        check(new_tool != "multiply", f"replacement is not original tool (got {new_tool})")
        # If a disjoint tool exists, replacement should be from disjoint pool
        # In our tools, multiply has params {arg_0, arg_1}. Disjoint pool:
        # make_json (dictionary), tokenize (sentence), find_kth (nums,k), swap_idx (values,i,j)
        DISJOINT_TOOLS = {"make_json", "tokenize", "find_kth", "swap_idx"}
        if new_tool in DISJOINT_TOOLS:
            break
    else:
        check(False, "wrong_tool picks from disjoint pool when available")
    check(new_tool in DISJOINT_TOOLS,
          f"final pick {new_tool} is from disjoint pool")

    # ── Bug C: inefficient_order swaps independent pairs without FR ──
    print("\n[Test C] inefficient_order injection")
    plan = _make_test_plan_4step_independent()
    injected, ok = pg._inject_inefficient_order(plan)
    check(ok, "inefficient_order returns ok=True for plan with independent pair")
    is_valid, errs = sv.validate(injected)
    check(is_valid,
          f"swapped plan is statically valid (no FR introduced): errs={errs}")
    # Check that step_ids appear out of order in the list (sign of swap)
    list_ids = [s.step_id for s in injected.steps]
    check(list_ids != sorted(list_ids),
          f"step_ids out of monotonic order in list: {list_ids}")

    # Edge case: strict dependency chain — no swap possible
    chain = _make_test_plan_3step()  # 0 → 1 → 2 strict chain
    injected, ok = pg._inject_inefficient_order(chain)
    check(not ok,
          "inefficient_order returns ok=False for strict dependency chain")

    # ── Bug D: type_mismatch on dict-only literal plan ──
    print("\n[Test D] type_mismatch injection")
    plan = _make_test_plan_2step_dict()
    injected, ok = pg._inject_type_mismatch(plan)
    check(ok, "type_mismatch returns ok=True for dict-only plan")
    # Step 0's `dictionary` param should now have an incompatible type
    new_val = injected.steps[0].parameters["dictionary"]
    check(not isinstance(new_val, dict),
          f"dictionary param replaced (got type {type(new_val).__name__})")
    # Static validator should detect the type error
    is_valid, errs = sv.validate(injected)
    check(not is_valid, f"static validator detects the injected mismatch: errs={errs}")

    # ── Bug E: forward_reference works on 2-step plans ──
    print("\n[Test E] forward_reference injection on 2-step plan")
    plan2 = Plan(steps=[
        Step(0, "add", {"arg_0": 1, "arg_1": 2}, "{{0}}"),
        Step(1, "negate", {"arg_0": "{{0}}"},     "{{1}}", "..."),
    ])
    injected, ok = pg._inject_forward_reference(plan2)
    check(ok, "forward_reference returns ok=True for 2-step plan")
    # Step 0's first param should now reference {{1}} (forward)
    p0 = injected.steps[0].parameters
    has_fr = any(isinstance(v, str) and "{{1}}" in v for v in p0.values())
    check(has_fr, f"step 0 contains {{{{1}}}} forward reference: params={p0}")
    is_valid, errs = sv.validate(injected)
    check(not is_valid, f"static validator detects FR: errs={errs}")

    # ── Bug F: failed injection orchestrator falls back ──
    print("\n[Test F] orchestrator retries when injection fails")
    chain = _make_test_plan_3step()  # strict chain
    candidates = pg.generate_candidates(chain, n_candidates=10)
    check(len(candidates) == 10, f"got {len(candidates)} candidates")
    # All candidates should have a valid error_type label
    for i, c in enumerate(candidates):
        check(c.error_type in {"none", "type_mismatch", "missing_dependency",
                                "wrong_tool", "parameter_typo", "circular_dependency",
                                "inefficient_order", "incomplete_plan",
                                "unnecessary_steps", "forward_reference"},
              f"candidate {i} has valid error_type label: {c.error_type}")

    # ── Sanity: GT plan validates clean ──
    print("\n[Sanity] ground-truth validation")
    is_valid, errs = sv.validate(_make_test_plan_3step())
    check(is_valid, f"GT plan validates: errs={errs}")
    is_valid, errs = sv.validate(_make_test_plan_2step_dict())
    check(is_valid, f"GT dict plan validates: errs={errs}")
    is_valid, errs = sv.validate(_make_test_plan_4step_independent())
    check(is_valid, f"GT 4-step plan validates: errs={errs}")

    print("\n" + "=" * 60)
    if failures:
        print(f"❌ {len(failures)} self-test(s) FAILED:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"✅ All self-tests passed.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    if "--self-test" in sys.argv:
        sys.exit(run_self_tests())

    parser = argparse.ArgumentParser(
        description="Generate NESTFUL annotated plans dataset (v2 — bug fixes applied)"
    )
    parser.add_argument("--nestful-path",  required=True,
                        help="Path to nestful_data.jsonl")
    parser.add_argument("--output-path",   required=True,
                        help="Path to save annotated JSON dataset")
    parser.add_argument("--api-key",       required=True,
                        help="OpenAI API key")
    parser.add_argument("--model",         default="gpt-4o-mini",
                        help="Model to use for judge annotation")
    parser.add_argument("--n-candidates",  type=int, default=10,
                        help="Candidate plans per query")
    parser.add_argument("--max-queries",   type=int, default=None,
                        help="Cap on number of queries to process")
    args = parser.parse_args()

    gen = DatasetGenerator(
        nestful_path=args.nestful_path,
        api_key=args.api_key,
        model=args.model,
        n_candidates=args.n_candidates,
    )
    dataset = gen.generate(max_queries=args.max_queries)
    gen.save_dataset(dataset, args.output_path)
    gen.print_statistics(dataset)


if __name__ == "__main__":
    main()