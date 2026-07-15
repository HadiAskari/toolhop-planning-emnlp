"""
E1 — Execution harness core: run tool plans and check end-task answers.

ToolHop  : every ToolHop.json entry ships per-query executable Python tool
           implementations (entry['functions'], list of source strings) and a
           gold final answer (entry['answer'], str).
NESTFUL  : the math subset (MathQA-derived, ~40 tools) is executed against the
           official IBM implementations vendored in nestful_exec/basic_functions.py.
           The coding subset can optionally be executed by pointing
           --nestful-exec-dir at a clone of IBM/NESTFUL's
           data_v2/executable_functions directory (func_file_map.json +
           py_code_file_*.py).

Plans are parsed with the same parse_plan_steps used for the paper's
structural metrics, then executed step-by-step with {{N}} dependency
resolution. A plan's final answer is the output of its last step.

Failure taxonomy of execution (recorded per plan):
  parse_empty        plan text yielded no parseable steps
  unknown_tool       a step names a tool with no implementation
  unresolved_ref     a {{N}} reference to a step that has not executed
                     (covers forward references and circular chains)
  call_error         the tool raised (bad params, missing args, internal error)
  timeout            the plan exceeded the wall-clock budget
  ok                 all steps executed; answer then compared to gold
"""

import ast
import math
import re
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import parse_plan_steps  # noqa: E402

REF_RE = re.compile(r"\{\{(\d+)\}\}")
MARKER_RE = re.compile(r"__REF_(\d+)__")

# Lenient step line: "Step 0: tool(args)" with the "{{0}} = " assignment
# optional. Some baselines (e.g. ToolPlanner-qwen3B on NESTFUL) omit the
# output-variable assignment; the paper's Table-2 numbers for those runs were
# produced via rescue_plans.py's robust re-parse, so E1 mirrors that.
LENIENT_STEP_RE = re.compile(
    r"Step\s+(\d+)\s*:\s*(?:(\{\{\d+\}\})\s*=\s*)?([A-Za-z_]\w*)\s*\((.*)\)\s*$")


def _split_params(params_str: str) -> Dict[str, str]:
    """Depth/quote-aware split of 'k1=v1, k2=v2' — same algorithm as the
    canonical parse_plan_steps."""
    params: Dict[str, str] = {}
    if not params_str:
        return params
    param_parts = []
    current = ""
    depth = 0
    in_str = False
    str_char = None
    for ch in params_str:
        if ch in ('"', "'") and (not in_str or ch == str_char):
            in_str = not in_str
            str_char = ch if in_str else None
        if not in_str:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                param_parts.append(current.strip())
                current = ""
                continue
        current += ch
    if current.strip():
        param_parts.append(current.strip())
    for part in param_parts:
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip()] = v.strip()
    return params


def parse_plan_steps_lenient(plan_text: str) -> List[Dict]:
    """Fallback parser for step lines missing the '{{N}} =' assignment."""
    steps = []
    for line in str(plan_text).split("\n"):
        m = LENIENT_STEP_RE.match(line.strip())
        if not m:
            continue
        sid = int(m.group(1))
        steps.append({
            "step_id": sid,
            "output_variable": m.group(2) or f"{{{{{sid}}}}}",
            "tool_name": m.group(3).strip(),
            "parameters": _split_params(m.group(4).strip()),
        })
    return steps


def parse_plan_any(plan: Any) -> List[Dict]:
    """Canonical parse; lenient fallback only when it finds nothing."""
    if not isinstance(plan, str):
        return plan
    steps = parse_plan_steps(plan)
    if not steps:
        steps = parse_plan_steps_lenient(plan)
    return steps


# ══════════════════════════════════════════════════════════════════════════════
# Timeouts (POSIX signal-based; harness must run in the main thread)
# ══════════════════════════════════════════════════════════════════════════════

class PlanTimeout(Exception):
    pass


class _timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        def handler(signum, frame):
            raise PlanTimeout()
        self._old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# {{N}} dependency resolution
# ══════════════════════════════════════════════════════════════════════════════

class UnresolvedRef(Exception):
    def __init__(self, idx: int):
        self.idx = idx
        super().__init__(f"reference to step {idx} which has not executed")


def _try_literal(s: str) -> Any:
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
            return s[1:-1]
        return s


def _sub_markers(obj: Any, outputs: Dict[int, Any]) -> Any:
    """Recursively replace __REF_j__ markers with step outputs.

    A string that IS exactly a marker becomes the native output object
    (type-preserving); a string that merely CONTAINS markers gets string
    substitution."""
    if isinstance(obj, str):
        m = MARKER_RE.fullmatch(obj.strip())
        if m:
            j = int(m.group(1))
            if j not in outputs:
                raise UnresolvedRef(j)
            return outputs[j]
        if MARKER_RE.search(obj):
            def repl(mm):
                j = int(mm.group(1))
                if j not in outputs:
                    raise UnresolvedRef(j)
                return str(outputs[j])
            return MARKER_RE.sub(repl, obj)
        return obj
    if isinstance(obj, list):
        return [_sub_markers(x, outputs) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_sub_markers(x, outputs) for x in obj)
    if isinstance(obj, dict):
        return {k: _sub_markers(v, outputs) for k, v in obj.items()}
    return obj


def resolve_param_value(value: Any, outputs: Dict[int, Any]) -> Any:
    """
    Resolve one parameter value, which may be:
      - a native object from corpus step dicts (list/int/str/... possibly
        containing "{{N}}" strings at any nesting level), or
      - a raw source-text string from parse_plan_steps (e.g. "'Kaylee Frye'",
        "[1, 2, 3]", "{{0}}", "'{{0}}'", "['{{0}}', 'Alan Dale']").
    """
    if isinstance(value, str):
        if not REF_RE.search(value):
            return _try_literal(value)
        marked = REF_RE.sub(lambda m: f"__REF_{m.group(1)}__", value).strip()
        # bare (possibly quoted) single reference -> native output object
        stripped = marked
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
            stripped = stripped[1:-1].strip()
        m = MARKER_RE.fullmatch(stripped)
        if m:
            j = int(m.group(1))
            if j not in outputs:
                raise UnresolvedRef(j)
            return outputs[j]
        try:
            obj = ast.literal_eval(marked)
        except Exception:
            # not a python literal: treat as plain string with refs substituted
            return _sub_markers(stripped if stripped != marked else marked, outputs)
        return _sub_markers(obj, outputs)
    if isinstance(value, list):
        return [resolve_param_value(v, outputs) for v in value]
    if isinstance(value, dict):
        return {k: resolve_param_value(v, outputs) for k, v in value.items()}
    return value


# ══════════════════════════════════════════════════════════════════════════════
# Tool registries
# ══════════════════════════════════════════════════════════════════════════════

_SAFE_PRELUDE = (
    "import math, re, json, datetime, itertools, collections, string, random, functools\n"
    "import typing\n"
    "from typing import List, Dict, Tuple, Optional, Union, Any\n"
)


def build_toolhop_registry(entry: Dict[str, Any]) -> Tuple[Dict[str, Callable], Dict[str, str]]:
    """
    Compile one ToolHop entry's function sources into callables.

    Returns (registry, alias_map):
      registry : {api_function_name: callable}
      alias_map: {nl_subquestion_or_spec_name: api_function_name} for remapping
                 plans that use non-API tool names.
    """
    ns: Dict[str, Any] = {}
    exec(_SAFE_PRELUDE, ns)
    for src in entry.get("functions", []):
        try:
            exec(src, ns)
        except Exception:
            continue  # skip an uncompilable tool; its calls will be unknown_tool

    # take exactly the top-level function names defined by the sources
    defined = set()
    for src in entry.get("functions", []):
        for m in re.finditer(r"^def\s+(\w+)", src, flags=re.MULTILINE):
            defined.add(m.group(1))
    registry = {k: ns[k] for k in defined if k in ns and callable(ns[k])}

    alias_map: Dict[str, str] = {}
    for nl_key, spec in entry.get("tools", {}).items():
        api = spec.get("name")
        if api:
            alias_map[nl_key] = api
            alias_map[api] = api
    return registry, alias_map


_NESTFUL_BASIC_REGISTRY: Optional[Dict[str, Callable]] = None


def load_nestful_basic_registry() -> Dict[str, Callable]:
    """Import the vendored IBM basic_functions.py (math subset, ~40 tools)."""
    global _NESTFUL_BASIC_REGISTRY
    if _NESTFUL_BASIC_REGISTRY is not None:
        return _NESTFUL_BASIC_REGISTRY
    import importlib.util
    p = Path(__file__).resolve().parent / "nestful_exec" / "basic_functions.py"
    spec = importlib.util.spec_from_file_location("nestful_basic_functions", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _NESTFUL_BASIC_REGISTRY = {
        name: fn for name, fn in vars(mod).items()
        if callable(fn) and not name.startswith("_") and getattr(fn, "__module__", "") == "nestful_basic_functions"
    }
    return _NESTFUL_BASIC_REGISTRY


class NestfulExtendedRegistry:
    """
    Lazy loader for the coding-subset functions from a local clone of
    IBM/NESTFUL data_v2/executable_functions (func_file_map.json + py files).
    """

    def __init__(self, exec_dir: str):
        import json as _json
        self.exec_dir = Path(exec_dir)
        map_path = self.exec_dir / "func_file_map.json"
        with open(map_path) as f:
            self.func_file_map: Dict[str, str] = _json.load(f)
        self._cache: Dict[str, Callable] = {}

    def get(self, name: str) -> Optional[Callable]:
        if name in self._cache:
            return self._cache[name]
        fname = self.func_file_map.get(name)
        if not fname:
            return None
        path = self.exec_dir / fname
        if not path.exists():
            return None
        ns: Dict[str, Any] = {}
        try:
            exec(_SAFE_PRELUDE, ns)
            exec(path.read_text(), ns)
        except Exception:
            return None
        fn = ns.get(name)
        if callable(fn):
            self._cache[name] = fn
            return fn
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Plan execution
# ══════════════════════════════════════════════════════════════════════════════

_ARG_N_RE = re.compile(r"^arg_(\d+)$")

# ToolHop tools signal failure by RETURNING error payloads rather than raising.
_ERROR_KEYS = {"error", "error_message"}
_UNWRAP_KEYS = ["result", "answer", "output", "value", "data"]


def _is_error_payload(v: Any) -> Optional[str]:
    """Return the error message if v is a tool-signaled error, else None."""
    if isinstance(v, dict):
        for k in _ERROR_KEYS:
            if k in v:
                return str(v[k])[:200]
    if isinstance(v, str) and re.match(r"^\s*error\b", v, re.IGNORECASE):
        return v[:200]
    return None


def _unwrap_output(v: Any) -> Any:
    """
    Unwrap structured tool outputs to the value the next step needs.
    In the interactive ToolHop benchmark the agent reads the tool's JSON
    output and extracts the value itself; in one-shot plan execution the
    executor must do this: {'closest_palindrome': 1881} -> 1881,
    {'result': '2011'} -> '2011'. Iterates because unwrapping can nest.
    """
    for _ in range(3):
        if isinstance(v, dict) and len(v) == 1:
            v = next(iter(v.values()))
            continue
        if isinstance(v, dict):
            for k in _UNWRAP_KEYS:
                if k in v:
                    v = v[k]
                    break
            else:
                break
            continue
        break
    return v


def _coerce_to_spec_type(value: Any, spec_type: Optional[str]) -> Any:
    """
    Gentle coercion of a resolved argument to its declared schema type.
    Mirrors what the interactive agent does naturally when reading a tool
    output like '2011' (str) and passing it to a number parameter.
    """
    if spec_type in ("integer", "number") and isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            f = float(s)
            if spec_type == "integer" and f.is_integer():
                return int(f)
            return int(f) if f.is_integer() and "." not in s else f
        except ValueError:
            return value
    if spec_type == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if spec_type == "string" and isinstance(value, (int, float)):
        return str(value)
    return value


def _call_tool(fn: Callable, kwargs: Dict[str, Any]) -> Any:
    """
    Call a tool implementation with the plan's keyword arguments.

    NESTFUL tool SPECS name parameters arg_0/arg_1/... while several IBM
    IMPLEMENTATIONS use semantic names (power(base, exponent), speed(distance,
    time), ...). Plans legitimately use the spec names, so when a direct
    keyword call fails with an unexpected-keyword TypeError and every key is
    arg_N-style, retry positionally ordered by N.
    """
    try:
        return fn(**kwargs)
    except TypeError as e:
        msg = str(e)
        if ("unexpected keyword argument" in msg or "required positional" in msg) \
                and kwargs and all(_ARG_N_RE.match(k) for k in kwargs):
            ordered = [kwargs[k] for k in
                       sorted(kwargs, key=lambda k: int(_ARG_N_RE.match(k).group(1)))]
            return fn(*ordered)
        raise


def _call_tool_with_retry(fn: Callable, kwargs: Dict[str, Any],
                          required: Optional[List[str]] = None) -> Any:
    """
    Retry ladder modeling the recovery available to interactive agents
    (who see a tool's validation error and re-call). Applied identically to
    every method, so comparisons stay fair:

      1. call with the plan's full arguments;
      2. if the tool rejects the call (error payload or ValueError/KeyError)
         and the error message names exactly one OPTIONAL argument, drop it
         and retry;
      3. if still rejected, retry with only the schema-required arguments.

    Returns the first non-error result; otherwise the last result/exception.
    """
    def attempt(kw):
        out = _call_tool(fn, kw)
        return out, _is_error_payload(out)

    try:
        result, err = attempt(kwargs)
        if err is None:
            return result
    except PlanTimeout:
        raise
    except (ValueError, KeyError, TypeError) as e:
        result, err = None, str(e)
    except Exception:
        raise

    required = list(required or [])
    optional_keys = [k for k in kwargs if k not in required]

    # 2. targeted drop: the error names exactly one optional argument
    named = [k for k in optional_keys if k.lower() in err.lower()]
    if len(named) == 1:
        try:
            result2, err2 = attempt({k: v for k, v in kwargs.items() if k != named[0]})
            if err2 is None:
                return result2
        except PlanTimeout:
            raise
        except Exception:
            pass

    # 3. required-only
    if required and optional_keys and any(k in kwargs for k in required):
        try:
            result3, err3 = attempt({k: v for k, v in kwargs.items() if k in required})
            if err3 is None:
                return result3
        except PlanTimeout:
            raise
        except Exception:
            pass

    if result is not None:
        return result          # last error payload; caller detects it
    raise ValueError(err)      # original exception path


def execute_plan(steps: List[Dict[str, Any]],
                 registry: Dict[str, Callable],
                 alias_map: Optional[Dict[str, str]] = None,
                 extended: Optional["NestfulExtendedRegistry"] = None,
                 param_types: Optional[Dict[str, Dict[str, Any]]] = None,
                 plan_timeout_s: int = 20) -> Dict[str, Any]:
    """
    Execute parsed steps in listed order. Returns a dict:
      {"status": "ok"|"parse_empty"|"unknown_tool"|"unresolved_ref"|
                 "call_error"|"tool_error"|"timeout",
       "final_output": Any (only when ok),
       "n_steps_executed": int,
       "failed_step": int|None, "error": str}

    param_types: optional {tool_name: {"types": {param: schema_type},
                                       "required": [param, ...]}} used to
    coerce resolved values and drive the optional-arg retry ladder.
    """
    if not steps:
        return {"status": "parse_empty", "n_steps_executed": 0,
                "failed_step": None, "error": "no parseable steps"}

    outputs: Dict[int, Any] = {}
    executed = 0
    try:
        with _timeout(plan_timeout_s):
            for pos, step in enumerate(steps):
                tool_name = str(step["tool_name"]).strip()
                fn = registry.get(tool_name)
                if fn is None and alias_map and tool_name in alias_map:
                    fn = registry.get(alias_map[tool_name])
                if fn is None and extended is not None:
                    fn = extended.get(tool_name)
                if fn is None:
                    return {"status": "unknown_tool", "n_steps_executed": executed,
                            "failed_step": step.get("step_id", pos),
                            "error": f"no implementation for tool '{tool_name}'"}

                try:
                    kwargs = {k: resolve_param_value(v, outputs)
                              for k, v in step["parameters"].items()}
                except UnresolvedRef as e:
                    return {"status": "unresolved_ref", "n_steps_executed": executed,
                            "failed_step": step.get("step_id", pos), "error": str(e)}

                spec = None
                if param_types:
                    spec = (param_types.get(tool_name)
                            or (param_types.get(alias_map[tool_name])
                                if alias_map and tool_name in alias_map else None))
                if spec and spec.get("types"):
                    kwargs = {k: _coerce_to_spec_type(v, spec["types"].get(k))
                              for k, v in kwargs.items()}

                try:
                    result = _call_tool_with_retry(
                        fn, kwargs, required=spec.get("required") if spec else None)
                except PlanTimeout:
                    raise
                except Exception as e:
                    return {"status": "call_error", "n_steps_executed": executed,
                            "failed_step": step.get("step_id", pos),
                            "error": f"{type(e).__name__}: {e}"}

                err_msg = _is_error_payload(result)
                if err_msg is not None:
                    return {"status": "tool_error", "n_steps_executed": executed,
                            "failed_step": step.get("step_id", pos),
                            "error": f"tool returned error: {err_msg}"}
                result = _unwrap_output(result)

                # index outputs by the step's declared output variable {{N}};
                # fall back to positional index
                m = REF_RE.fullmatch(str(step.get("output_variable", "")).strip())
                out_idx = int(m.group(1)) if m else step.get("step_id", pos)
                outputs[out_idx] = result
                executed += 1
    except PlanTimeout:
        return {"status": "timeout", "n_steps_executed": executed,
                "failed_step": None, "error": f"exceeded {plan_timeout_s}s"}

    last = steps[-1]
    m = REF_RE.fullmatch(str(last.get("output_variable", "")).strip())
    last_idx = int(m.group(1)) if m else last.get("step_id", len(steps) - 1)
    return {"status": "ok", "final_output": outputs.get(last_idx),
            "n_steps_executed": executed, "failed_step": None, "error": ""}


# ══════════════════════════════════════════════════════════════════════════════
# Answer comparison
# ══════════════════════════════════════════════════════════════════════════════

def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except OverflowError:  # e.g. huge ints from corrupted factorial/power
            return None
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        try:
            return float(s)
        except Exception:
            return None
    return None


def _norm_str(x: Any) -> str:
    try:
        s = str(x)
    except Exception:  # e.g. int-to-str digit limit on huge ints (py3.11+)
        return "<unprintable>"
    return " ".join(s.strip().strip("\"'").lower().split())


def answers_match(predicted: Any, gold: Any) -> Dict[str, bool]:
    """
    Returns {"strict": bool, "lenient": bool}.

    strict : numeric closeness (rel 1e-4 / abs 1e-6) or 2-decimal rounding
             for number pairs; normalized string equality otherwise.
    lenient: strict OR containment of the normalized gold string in the
             normalized predicted string (useful for tools that return
             sentences like "The answer is 4").
    """
    pf, gf = _as_float(predicted), _as_float(gold)
    if pf is not None and gf is not None:
        # match at the gold answer's own decimal precision: NESTFUL gold
        # answers are pre-rounded (e.g. 71.4 for a computed 71.42857...)
        gold_str = str(gold).strip()
        decimals = len(gold_str.split(".")[1]) if "." in gold_str else 0
        strict = (math.isclose(pf, gf, rel_tol=1e-4, abs_tol=1e-6)
                  or round(pf, 2) == round(gf, 2)
                  or round(pf, decimals) == gf)
        return {"strict": strict, "lenient": strict}

    ps, gs = _norm_str(predicted), _norm_str(gold)
    strict = ps == gs and gs != ""
    lenient = strict or (gs != "" and gs in ps)
    return {"strict": strict, "lenient": lenient}


# ══════════════════════════════════════════════════════════════════════════════
# Per-plan convenience wrappers
# ══════════════════════════════════════════════════════════════════════════════

def run_plan_toolhop(plan: Any, entry: Dict[str, Any],
                     plan_timeout_s: int = 20) -> Dict[str, Any]:
    """
    plan: either a plan string ("Step N: {{N}} = tool(...)") or a list of
          corpus-style step dicts. entry: the ToolHop.json entry (by query_id).
    """
    steps = parse_plan_any(plan)
    registry, alias_map = build_toolhop_registry(entry)
    param_types: Dict[str, Dict[str, Any]] = {}
    for spec in entry.get("tools", {}).values():
        name = spec.get("name")
        params = spec.get("parameters", {})
        props = params.get("properties", {})
        if name:
            param_types[name] = {
                "types": {p: info.get("type") for p, info in props.items()},
                "required": list(params.get("required", [])),
            }
    res = execute_plan(steps, registry, alias_map=alias_map,
                       param_types=param_types, plan_timeout_s=plan_timeout_s)
    out = {**res, "n_steps": len(steps)}
    if res["status"] == "ok":
        out["answer_check"] = answers_match(res["final_output"], entry["answer"])
        out["final_output_repr"] = repr(res["final_output"])[:300]
    else:
        out["answer_check"] = {"strict": False, "lenient": False}
        out["final_output_repr"] = None
    out["gold_answer"] = str(entry["answer"])
    return out


def run_plan_toolhop_grounded(plan: Any, entry: Dict[str, Any],
                              plan_timeout_s: int = 10) -> Dict[str, Any]:
    """
    Grounded-execution protocol for ToolHop.

    Rationale: ToolHop's per-query tools are GPT-written simulators whose
    internal lookup tables are keyed by exact strings; even gold reference
    plans rarely execute against them verbatim (the original benchmark runs
    them interactively, letting the agent read errors and retry). Grounded
    mode removes only the brittle string-keyed lookup layer while keeping
    execution semantics that discriminate real structural failures:

      - each tool maps to its sub-question hop k, whose gold sub-answer is
        provided by the benchmark (entry['sub_task']);
      - a call to the hop-k tool SUCCEEDS only if its resolved arguments are
        consistent with the hop k-1 gold sub-answer (dependency threading is
        genuinely tested: hardcoded wrong values, mis-ordered steps, skipped
        hops, and unresolvable {{N}} refs all fail), and then returns the
        hop-k gold sub-answer;
      - the plan's final output must equal the query's gold answer, so
        incomplete plans fail.

    Applied identically to every method.
    """
    steps = parse_plan_any(plan)
    if not steps:
        return {"status": "parse_empty", "n_steps_executed": 0, "failed_step": None,
                "error": "no parseable steps", "n_steps": 0,
                "answer_check": {"strict": False, "lenient": False},
                "final_output_repr": None, "gold_answer": str(entry["answer"]),
                "covered": True}

    subqs = list(entry.get("sub_task", {}).keys())
    toolq = list(entry.get("tools", {}).keys())
    if set(subqs) != set(toolq) or not subqs:
        # 3/995 entries have mismatched sub_task/tools keys: not gradable here
        return {"status": "uncovered", "n_steps_executed": 0, "failed_step": None,
                "error": "entry tools/sub_task keys mismatch", "n_steps": len(steps),
                "answer_check": {"strict": False, "lenient": False},
                "final_output_repr": None, "gold_answer": str(entry["answer"]),
                "covered": False}

    hop_answers = [entry["sub_task"][q] for q in subqs]
    question_norm = _norm_str(entry.get("question", ""))
    hops_of_tool: Dict[str, List[int]] = {}
    for k, q in enumerate(subqs):
        spec = entry["tools"][q]
        name = spec.get("name")
        if name:
            hops_of_tool.setdefault(name, []).append(k)
        hops_of_tool.setdefault(q, []).append(k)  # NL alias

    def _atoms(x: Any) -> List[str]:
        if isinstance(x, (list, tuple, set)):
            out = []
            for i in x:
                out.extend(_atoms(i))
            return out
        if isinstance(x, dict):
            out = []
            for v in x.values():
                out.extend(_atoms(v))
            return out
        return [_norm_str(x)]

    def _consistent(kwargs: Dict[str, Any], expected_any: List[Any]) -> bool:
        """Arguments are consistent if they carry the gold sub-answer of at
        least one EARLIER hop, or a multi-word entity taken from the question
        text itself (queries are multi-root DAGs: a hop's input may be any
        earlier hop's output or a question entity). Hardcoding a correct
        value also passes — exactly as it would against a real lookup tool."""
        atoms = []
        for v in kwargs.values():
            atoms.extend(_atoms(v))
            atoms.append(_norm_str(v))
        for expected in expected_any:
            exp = _norm_str(expected)
            if not exp:
                return True
            if any(a == exp for a in atoms):
                return True
            if len(exp) >= 4 and any(exp in a or (len(a) >= 4 and a in exp)
                                     for a in atoms):
                return True
        # question-derived root entity: multi-word only, to avoid matching
        # generic formatting values ('letters', 'first', ...) against the query
        return any(len(a) >= 4 and " " in a and a in question_norm for a in atoms)

    outputs: Dict[int, Any] = {}
    executed = 0
    used_hops: set = set()
    for pos, step in enumerate(steps):
        tool_name = str(step["tool_name"]).strip()
        if tool_name not in hops_of_tool:
            return {"status": "unknown_tool", "n_steps_executed": executed,
                    "failed_step": step.get("step_id", pos),
                    "error": f"no such tool '{tool_name}'", "n_steps": len(steps),
                    "answer_check": {"strict": False, "lenient": False},
                    "final_output_repr": None, "gold_answer": str(entry["answer"]),
                    "covered": True}
        try:
            kwargs = {p: resolve_param_value(v, outputs)
                      for p, v in step["parameters"].items()}
        except UnresolvedRef as e:
            return {"status": "unresolved_ref", "n_steps_executed": executed,
                    "failed_step": step.get("step_id", pos), "error": str(e),
                    "n_steps": len(steps),
                    "answer_check": {"strict": False, "lenient": False},
                    "final_output_repr": None, "gold_answer": str(entry["answer"]),
                    "covered": True}

        # a tool name may serve several hops; take the first candidate whose
        # input-consistency holds, preferring hops not yet executed
        candidates = hops_of_tool[tool_name]
        ordered = ([k for k in candidates if k not in used_hops]
                   + [k for k in candidates if k in used_hops])
        chosen = None
        for k in ordered:
            if k == 0 or _consistent(kwargs, hop_answers[:k]):
                chosen = k
                break
        if chosen is None:
            return {"status": "tool_error", "n_steps_executed": executed,
                    "failed_step": step.get("step_id", pos),
                    "error": (f"arguments to tool '{tool_name}' carry no "
                              f"upstream hop result or question entity"),
                    "n_steps": len(steps),
                    "answer_check": {"strict": False, "lenient": False},
                    "final_output_repr": None, "gold_answer": str(entry["answer"]),
                    "covered": True}
        used_hops.add(chosen)
        m = REF_RE.fullmatch(str(step.get("output_variable", "")).strip())
        out_idx = int(m.group(1)) if m else step.get("step_id", pos)
        outputs[out_idx] = hop_answers[chosen]
        executed += 1

    last = steps[-1]
    m = REF_RE.fullmatch(str(last.get("output_variable", "")).strip())
    last_idx = int(m.group(1)) if m else last.get("step_id", len(steps) - 1)
    final = outputs.get(last_idx)
    return {"status": "ok", "n_steps_executed": executed, "failed_step": None,
            "error": "", "n_steps": len(steps),
            "answer_check": answers_match(final, entry["answer"]),
            "final_output_repr": repr(final)[:300],
            "gold_answer": str(entry["answer"]), "covered": True}


def run_plan_nestful(plan: Any, gold_answer: Any,
                     extended: Optional[NestfulExtendedRegistry] = None,
                     plan_timeout_s: int = 20) -> Dict[str, Any]:
    steps = parse_plan_any(plan)
    registry = load_nestful_basic_registry()

    # coverage check: are all tools implementable?
    missing = [s["tool_name"] for s in steps
               if s["tool_name"] not in registry
               and (extended is None or extended.get(s["tool_name"]) is None)]
    covered = len(missing) == 0 or not steps

    res = execute_plan(steps, registry, extended=extended,
                       plan_timeout_s=plan_timeout_s)
    out = {**res, "n_steps": len(steps), "tools_covered": covered,
           "uncovered_tools": missing[:5]}
    if res["status"] == "ok":
        out["answer_check"] = answers_match(res["final_output"], gold_answer)
        out["final_output_repr"] = repr(res["final_output"])[:300]
    else:
        out["answer_check"] = {"strict": False, "lenient": False}
        out["final_output_repr"] = None
    out["gold_answer"] = str(gold_answer)
    return out


def nestful_gold_tools_covered(gold_steps: List[Dict[str, Any]],
                               extended: Optional[NestfulExtendedRegistry] = None) -> bool:
    """True iff every tool in the GOLD plan has an implementation available.
    Used to define the executable subset (denominator) per query."""
    registry = load_nestful_basic_registry()
    for s in gold_steps:
        name = s["tool_name"]
        if name in registry:
            continue
        if extended is not None and extended.get(name) is not None:
            continue
        return False
    return True
