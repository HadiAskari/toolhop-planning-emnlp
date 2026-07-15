#!/usr/bin/env python3
"""
α-UMi Baseline for ToolHop Tool Planning — FIXED VERSION (NESTFUL-aware)

Implements the α-UMi multi-LLM agent framework (Shen et al., EMNLP 2024)
as a baseline for the PPO-trained planner.

α-UMi decomposes tool learning into three specialized LLMs:
  - Planner : generates rationale + next decision (Caller / Conclusion / Give up)
  - Caller  : generates the tool call given the planner's rationale
  - Summarizer : (unused in offline eval — no final answer needed)

Since ToolHop has no real execution environment, we run in OFFLINE mode:
the Planner→Caller loop executes without real observations between steps.
We insert a placeholder observation token so the planner can see the
trajectory shape and decide when to stop. The planner prompt is rewritten
to make explicit that placeholder observations do NOT signal task completion
— the planner must enumerate ALL required tool calls before concluding.
This is the key adaptation from online α-UMi: without execution feedback,
the planner cannot know when the task is "done" by observing real outputs,
so completeness must be assessed against the query itself.

TWO EXECUTION VARIANTS:
  --mode single-model   One model plays all three roles (different system prompts).
                        Cheapest; tests whether role separation via prompting helps.
  --mode multi-model    Separate model paths for planner and caller.
                        Pass --caller-model if it differs from --planner-model.
                        Reproduces the original α-UMi paper setting as closely
                        as possible without fine-tuning separate models.

Output format EXACTLY matches best_of_n_selection.py so stats can be compared
side-by-side via the same compute_stats() function.

KNOWN DATASET ARTIFACT (ToolHop):
  Ground truth tool names are natural-language sub-questions.
  Exact tool_accuracy will be ~0. Use functional_tool_accuracy instead.

═══════════════════════════════════════════════════════════════════════════════
FIXES APPLIED IN THIS VERSION (vs. prior):
═══════════════════════════════════════════════════════════════════════════════

  FIX 1 — _normalize_step_line() injects "{{N}} =" into caller output.
    Llama-3.2-3B (and to a lesser extent other models) treats the "{{N}}" in
    the caller prompt as a template variable rather than as literal characters
    to copy verbatim, emitting:
        Step 0: tool_name(param1=value1, ...)
    instead of the canonical:
        Step 0: {{0}} = tool_name(param1=value1, ...)
    Without {{N}} = , the structural parser rejected every line and forced
    generated_n_steps=0 and every metric to 0. The normalizer detects the
    missing assignment in caller output and synthesizes it from step_idx
    before the line is appended to the trajectory. Canonical output =>
    canonical metrics.

  FIX 2 — parse_plan_steps() is now tolerant of missing "{{N}} =".
    Defense in depth: even if a caller emits a Step line without {{N}} = ,
    the parser synthesizes the output variable from the step_id rather than
    silently dropping the line. This salvages any existing JSON outputs that
    were generated before FIX 1.

  FIX 3 — LLMModel emits Llama-3.x <|eot_id|> as an additional EOS.
    Llama-3.2's tokenizer.eos_token_id typically points to <|end_of_text|>,
    but the model emits <|eot_id|> at the end of assistant turns. If only
    eos_token_id is passed to generate(), generation runs until max_new_tokens
    rather than stopping at the proper turn boundary. The fix adds <|eot_id|>
    to the eos_token_id list when the tokenizer has it, and is a no-op for
    non-Llama-3 models (Qwen, etc.) whose tokenizers don't recognize the token.

═══════════════════════════════════════════════════════════════════════════════

Usage:
    # Start judge server first:
    CUDA_VISIBLE_DEVICES=5 python judge_server.py \
        --model ${FORTE_ROOT}/judge_finetuning/models/judge/merged --port 8003 --batch-size 32 --batch-timeout-ms 200

    # Single-model variant (same LLM for planner + caller):
    CUDA_VISIBLE_DEVICES=0 python alpha_umi_baseline.py \
        --planner-model Qwen/Qwen2.5-7B-Instruct \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_nestful_full/test.parquet \
        --mode single-model --full \
        --judge_url http://localhost:8002/v1/chat/completions \
        --output qwen7b/alpha_umi_qwen-7B-Instruct_results.json
        
    # Multi-model variant (separate planner + caller models):
    CUDA_VISIBLE_DEVICES=3 python alpha_umi_baseline.py \
        --planner-model ${FORTE_ROOT}/planner_finetuning/checkpoints_planner_sft-7b-updated-hf \
        --caller-model  ${FORTE_ROOT}/planner_finetuning/checkpoints_planner_sft-7b-updated-hf \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --mode multi-model --full \
        --judge_url http://localhost:8002/v1/chat/completions \
        --output qwen3b/alpha_umi_multi_nestful_results.json
"""

import json
import re
import time
import argparse
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset_utils import resolve_dataset, dataset_label

# Error types where the GT is deliberately flawed in a way that makes the
# model's CORRECT plan look wrong under structural comparison.
# For these, we compare against the perfect plan for the same query instead.
ARTIFACT_ERROR_TYPES = {
    "circular_dependency",
    "forward_reference",
    "incomplete_plan",
    "inefficient_order",
    "missing_dependency",
    "parameter_typo",
    "type_mismatch",
    "unnecessary_steps",
    "wrong_tool",
}


# ── α-UMi System Prompts (adapted for offline planning) ─────────────────────

PLANNER_SYSTEM_PROMPT = (
    "You are the Planner in a multi-step tool-execution system. Your job is to "
    "decompose the user's query into a sequence of tool calls and decide when "
    "the decomposition is complete.\n\n"
    "IMPORTANT: This system runs in OFFLINE planning mode. The [Observation] "
    "entries you see in the conversation history are PLACEHOLDERS — the tools "
    "have NOT actually been executed and the placeholders carry no real "
    "information. You must judge completeness by checking whether every "
    "sub-question implied by the user's query has been addressed by a planned "
    "tool call, NOT by reading observation content.\n\n"
    "On each turn, generate a brief thought about what tool call should happen "
    "next, then output a final decision line that is EXACTLY one of:\n"
    "  Next: caller       (a new tool call is needed)\n"
    "  Next: conclusion   (every sub-question in the query has been planned)\n"
    "  Next: give up      (only if the query is genuinely impossible)\n\n"
    "Rules:\n"
    "- Do NOT output 'Next: conclusion' on the first step. The query always "
    "requires at least one tool call.\n"
    "- Do NOT output 'Next: give up' unless the query is impossible to "
    "decompose into the available tools.\n"
    "- The 'Next:' line MUST be the last line of your output.\n"
    "- Plan ALL required steps before concluding. ToolHop queries typically "
    "require 3–5 sequential tool calls.\n\n"
    "Output format:\n"
    "<one or two sentences of reasoning about the next tool call>\n"
    "Next: caller"
)

CALLER_SYSTEM_PROMPT = (
    "You are the Caller responsible for emitting tool invocations. "
    "Given the conversation history and the planner's thought for this step, "
    "generate exactly ONE tool call in the following format:\n\n"
    "Step N: {{N}} = tool_name(param1=value1, param2=value2, ...)\n\n"
    "The double-brace syntax `{{N}}` is LITERAL — copy it verbatim, do not "
    "substitute or omit it. Use {{N}} (e.g. {{0}}, {{1}}) for the output "
    "variable assignment on the LEFT of the equals sign, and to reference "
    "previous step outputs on the RIGHT. Generate ONLY the Step line — no "
    "extra commentary, no markdown, no code fences.\n\n"
    "Worked example:\n"
    "  Step 0: {{0}} = lookup_actor(name=\"Jane Doe\", series=\"Firefly\")\n"
    "  Step 1: {{1}} = find_year(actor={{0}}, genre=\"sci-fi\")"
)

# Judge server (shared with other baselines)
JUDGE_SERVER_URL = "http://localhost:8002/v1/chat/completions"
JUDGE_SYSTEM_PROMPT = """You are an expert judge for evaluating tool execution plans. Your task is to:
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
- 0-39: Critical errors, will fail"""

# Max planning steps — prevents infinite loops in offline mode
MAX_PLANNER_STEPS = 8


# ── Tool formatting ──────────────────────────────────────────────────────────


def load_perfect_gt_from_parquet(parquet_path: str) -> dict:
    """
    Build {query_id: ground_truth_str} for error_type='none', quality>=100.
    Used to give artifact error types a fair structural comparison target.
    """
    import json as _json
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    perfect_gt: dict = {}
    for ei, rm in zip(extra_infos, reward_models):
        if isinstance(ei, str):
            ei = _json.loads(ei)
        if isinstance(rm, str):
            rm = _json.loads(rm)
        if not isinstance(ei, dict) or not isinstance(rm, dict):
            continue
        if (str(ei.get("error_type", "")) == "none"
                and int(ei.get("quality_score", 0)) >= 100):
            qid = ei.get("query_id", -1)
            gt_str = rm.get("ground_truth", "")
            if gt_str and qid not in perfect_gt:
                perfect_gt[qid] = gt_str
    print(f"  Loaded perfect GT for {len(perfect_gt)} query_ids.")
    return perfect_gt


def format_tools(tools: Dict[str, Any]) -> str:
    lines = ["Available Tools:"]
    seen: Dict[str, Any] = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in seen:
            seen[name] = tool_info
    for name, tool_info in seen.items():
        props    = tool_info.get("parameters", {}).get("properties", {})
        required = tool_info.get("parameters", {}).get("required", [])
        params   = ", ".join(
            f"{k}: {v.get('type', 'any')}{' (required)' if k in required else ''}"
            for k, v in props.items()
        )
        lines.append(f"- {name}({params})")
    return "\n".join(lines)


# ── Trajectory helpers ───────────────────────────────────────────────────────

def _trajectory_to_text(trajectory: List[Dict]) -> str:
    """Render the execution trajectory as readable text for the planner/caller prompts."""
    parts = []
    for entry in trajectory:
        role = entry["role"]
        if role == "planner":
            parts.append(f"[Planner] {entry['content']}")
        elif role == "caller":
            parts.append(f"[Caller] {entry['content']}")
        elif role == "observation":
            parts.append(f"[Observation] {entry['content']}")
    return "\n".join(parts)


# ── FIX 1 — Caller output normalization ─────────────────────────────────────

def _normalize_step_line(line: str, step_idx: int) -> str:
    """
    Inject "{{N}} =" into a Step line if the model omitted the output-variable
    assignment.

    Llama-3.2-3B and some other instruction-tuned models treat the "{{N}}" in
    the caller prompt as a template variable rather than as literal characters
    to copy. They emit:
        Step 0: tool_name(args)
    instead of the canonical:
        Step 0: {{0}} = tool_name(args)
    The downstream parser requires "{{N}} =" to extract the tool call, so the
    omission silently drops the line and forces all structural metrics to zero.

    This normalizer detects a missing "{{N}} =" assignment and synthesizes it
    from the step index, producing canonical output. It is a no-op when:
      - The line is already in canonical form ("{{N}} =" present)
      - The line doesn't match the "Step N:" prefix at all

    Returns the (possibly rewritten) line.
    """
    m = re.match(r"^\s*Step\s+(\d+)\s*:\s*(.*)$", line)
    if not m:
        return line
    n = int(m.group(1))
    body = m.group(2).strip()
    # Already canonical — leave it alone
    if re.match(r"\{\{\d+\}\}\s*=", body):
        return line
    return f"Step {n}: {{{{{n}}}}} = {body}"


# ── Decision parser ─────────────────────────────────────────────────────────

def _parse_planner_decision(raw: str, step_idx: int) -> str:
    """
    Parse the planner's decision robustly.

    Rules (in order):
    1. Find the LAST line that starts with 'next:' (case-insensitive, after
       leading whitespace stripped). Markdown wrappers (e.g. **Next:**) are
       tolerated.
    2. Inspect the suffix after 'next:' — the FIRST keyword to appear wins:
       'conclusion' → conclusion
       'give up' / 'giveup' → give up
       otherwise → caller
    3. If no 'Next:' line exists, default to 'caller' (do NOT silently stop).
    4. On step 0, never honor 'conclusion' or 'give up' — force 'caller' so
       we always plan at least one tool call.
    """
    decision = "caller"  # safe default — keep planning
    # Walk lines in reverse so the LAST Next: line wins (model may revise).
    for line in reversed(raw.split("\n")):
        line_low = line.strip().lower()
        # Tolerate light markdown wrappers like **Next:** caller
        line_low = re.sub(r"^\*+\s*", "", line_low)
        line_low = re.sub(r"\s*\*+$", "", line_low)
        if line_low.startswith("next:"):
            suffix = line_low[len("next:"):].strip()
            # FIRST keyword in the suffix decides
            conc_idx = suffix.find("conclusion")
            give_idx = suffix.find("give up")
            if give_idx == -1:
                give_idx = suffix.find("giveup")
            if conc_idx != -1 and (give_idx == -1 or conc_idx < give_idx):
                decision = "conclusion"
            elif give_idx != -1:
                decision = "give up"
            else:
                decision = "caller"
            break

    # Step-0 guard: planner must attempt at least one tool call.
    if step_idx == 0 and decision in ("conclusion", "give up"):
        decision = "caller"

    return decision


# ── Model wrapper ────────────────────────────────────────────────────────────

class LLMModel:
    """Thin wrapper around a HuggingFace causal LM."""

    def __init__(self, model_path: str, device: str = "cuda", label: str = "model"):
        print(f"Loading {label} from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # ── FIX 3 — Llama-3.x EOS token handling ─────────────────────────────
        # Llama-3.x emits <|eot_id|> at the end of assistant turns, but its
        # tokenizer.eos_token_id points to <|end_of_text|>, which the model
        # never emits during conversation. Without <|eot_id|> in the eos list,
        # generation runs until max_new_tokens and produces trailing garbage.
        # No-op for non-Llama-3 models (e.g. Qwen) where the token isn't
        # recognized — convert_tokens_to_ids returns unk_token_id, which we
        # filter out.
        self.eos_token_ids = self.tokenizer.eos_token_id
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if (eot_id is not None
                    and eot_id != self.tokenizer.unk_token_id
                    and eot_id != self.tokenizer.eos_token_id):
                self.eos_token_ids = [self.tokenizer.eos_token_id, eot_id]
                print(f"  ✓ Llama-3-style <|eot_id|> detected (id={eot_id}); "
                      f"added to eos list.")
        except Exception:
            # Older tokenizers without this token — leave eos_token_ids as default
            pass

        print(f"  ✓ {label} loaded on {self.device}")

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-6),
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_ids,  # FIX 3
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# ── α-UMi agent ──────────────────────────────────────────────────────────────

class AlphaUMiAgent:
    """
    Offline α-UMi agent.

    Planner→Caller loop without real observations.
    Inserts an explicit non-completion-signaling observation so the planner
    knows the trajectory is partial and must continue until all sub-questions
    are addressed.
    """

    def __init__(
        self,
        planner: LLMModel,
        caller:  Optional[LLMModel] = None,   # None → reuse planner (single-model mode)
    ):
        self.planner = planner
        self.caller  = caller if caller is not None else planner  # single-model fallback

    # ── Planner ────────────────────────────────────────────────────────────

    def _run_planner(
        self,
        question: str,
        tools: Dict[str, Any],
        trajectory: List[Dict],
        step_idx: int,
    ) -> Tuple[str, str]:
        """
        Returns (rationale_text, next_decision) where next_decision in
        {"caller", "conclusion", "give up"}.
        """
        tools_str    = format_tools(tools)
        history_text = _trajectory_to_text(trajectory)

        user_content = (
            f"Query: {question}\n\n"
            f"{tools_str}\n\n"
            f"Conversation history (offline mode — observations are placeholders):\n"
            f"{history_text if history_text else '(none yet — this is the first step)'}\n\n"
            f"Current step index: {step_idx}\n"
            f"Steps planned so far: {sum(1 for t in trajectory if t['role']=='caller')}\n\n"
            f"Decide what should happen next. Remember: judge completeness by "
            f"comparing the planned tool calls against the sub-questions in the "
            f"query, NOT by reading observation content. End your response with "
            f"a single 'Next:' line."
        )

        raw = self.planner.generate(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.0,
            max_new_tokens=200,
        )

        decision = _parse_planner_decision(raw, step_idx)
        return raw, decision

    # ── Caller ─────────────────────────────────────────────────────────────

    def _run_caller(
        self,
        question: str,
        tools: Dict[str, Any],
        trajectory: List[Dict],
        rationale: str,
        step_idx: int,
    ) -> str:
        """
        Returns the Step N: {{N}} = tool_name(...) line.

        FIX 1 applied: caller output is normalized via _normalize_step_line()
        before return. If the model omits the "{{N}} =" assignment (common with
        Llama-3.2-3B), the normalizer synthesizes it from step_idx so the
        downstream parser can extract the tool call.
        """
        tools_str    = format_tools(tools)
        history_text = _trajectory_to_text(trajectory)

        user_content = (
            f"Query: {question}\n\n"
            f"{tools_str}\n\n"
            f"Conversation history:\n{history_text if history_text else '(none yet)'}\n\n"
            f"Planner's thought for step {step_idx}: {rationale}\n\n"
            f"Generate the tool call for step {step_idx}. "
            f"Use the format: Step {step_idx}: {{{{{step_idx}}}}} = tool_name(param1=value1, ...)"
        )

        raw = self.caller.generate(
            system_prompt=CALLER_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.0,
            max_new_tokens=200,
        )

        # Extract only the Step N: line; normalize {{N}} = if missing
        for line in raw.split("\n"):
            line = line.strip()
            if re.match(r"Step \d+:", line):
                return _normalize_step_line(line, step_idx)

        # Fallback: return raw trimmed (also normalized)
        fallback = raw.strip().split("\n")[0]
        return _normalize_step_line(fallback, step_idx)

    # ── Full rollout ────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        tools: Dict[str, Any],
        max_steps: int = MAX_PLANNER_STEPS,
    ) -> Tuple[str, List[Dict]]:
        """
        Execute the Planner→Caller loop.

        Returns:
          plan_str   : concatenated "Step N: ..." lines (judge-compatible)
          trajectory : full raw trajectory (for debugging)
        """
        trajectory: List[Dict] = []
        step_lines: List[str]  = []

        for step_idx in range(max_steps):
            # 1. Planner
            rationale, decision = self._run_planner(
                question, tools, trajectory, step_idx
            )
            trajectory.append({"role": "planner", "content": rationale})

            if decision in ("conclusion", "give up"):
                break

            # 2. Caller
            step_line = self._run_caller(
                question, tools, trajectory, rationale, step_idx
            )
            trajectory.append({"role": "caller", "content": step_line})
            step_lines.append(step_line)

            # 3. Placeholder observation that explicitly does NOT signal completion.
            obs_text = (
                f"[Step {step_idx} planned. NOTE: No real tool execution occurred. "
                f"This placeholder does NOT indicate the task is complete. "
                f"Continue planning subsequent steps until every sub-question in "
                f"the original query has been addressed.]"
            )
            trajectory.append({"role": "observation", "content": obs_text})

        plan_str = "\n".join(step_lines)
        return plan_str, trajectory


# ── Shared utilities (mirrored from best_of_n_selection.py) ─────────────────

def _is_nl_tool_name(name: str) -> bool:
    return len(name.split()) > 4 or name.endswith("?")


def _functional_tool_match(gen_name: str, gt_name: str) -> float:
    STOP = {"what", "is", "the", "of", "in", "a", "an", "and", "or", "to",
            "how", "many", "who", "which", "are", "was", "were", "be", "been",
            "at", "on", "for", "with", "that", "this", "it", "its", "from"}

    def keywords(s: str) -> set:
        words = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
        return {w for w in words if w not in STOP and len(w) > 2}

    gen_kw = keywords(gen_name)
    gt_kw  = keywords(gt_name)
    if not gen_kw or not gt_kw:
        return 0.0
    intersection = gen_kw & gt_kw
    union        = gen_kw | gt_kw
    return round(len(intersection) / len(union), 3)


def normalize_value(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())


def parse_plan_steps(plan_text: str) -> List[Dict]:
    """
    Parse Step-N lines from a generated plan into structured dicts.

    FIX 2 applied: tolerates missing "{{N}} =" output-variable assignments.
    When the canonical "Step N: {{N}} = tool(args)" pattern fails to match,
    falls back to "Step N: tool(args)" and synthesizes the output variable
    from the step index. This salvages caller outputs from models that drop
    the {{N}} = prefix (notably Llama-3.2-3B).

    Both strict and lenient patterns produce identical downstream dicts —
    {step_id, output_variable, tool_name, parameters} — so eval code does
    not need any changes.
    """
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("Step "):
            continue
        try:
            step_match = re.match(r"Step (\d+):", line)
            if not step_match:
                continue
            step_id = int(step_match.group(1))

            # ── Try canonical: Step N: {{N}} = tool(args) ─────────────────
            var_match = re.search(r"(\{\{\d+\}\})\s*=", line)
            if var_match:
                output_var = var_match.group(1)
                tool_match = re.search(r"=\s*([^\(]+)\((.*)\)\s*$", line)
                tool_match_empty = re.search(r"=\s*([^\(]+)\(\)\s*$", line)
            else:
                # ── Lenient: Step N: tool(args) without {{N}} = ───────────
                # FIX 2: synthesize the output variable from the step index.
                output_var = f"{{{{{step_id}}}}}"
                # Tool name immediately after "Step N:" — restrict to a Python
                # identifier so we don't false-positive on free-form text.
                tool_match = re.search(
                    r"^Step \d+:\s*([a-zA-Z_]\w*)\((.*)\)\s*$", line
                )
                tool_match_empty = re.search(
                    r"^Step \d+:\s*([a-zA-Z_]\w*)\(\)\s*$", line
                )

            if not tool_match:
                if tool_match_empty:
                    tool_name = tool_match_empty.group(1).strip()
                    params: Dict[str, str] = {}
                else:
                    continue
            else:
                tool_name  = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                params: Dict[str, str] = {}
                if params_str:
                    param_parts: List[str] = []
                    current = ""
                    depth   = 0
                    in_str  = False
                    str_char: Optional[str] = None
                    for ch in params_str:
                        if ch in ('"', "'") and (not in_str or ch == str_char):
                            in_str   = not in_str
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

            steps.append({
                "step_id":         step_id,
                "output_variable": output_var,
                "tool_name":       tool_name,
                "parameters":      params,
            })
        except Exception:
            continue
    return steps


def _remap_gt_tool_name(nl_name: str, tools: Dict[str, Any]) -> str:
    """
    ToolHop GT plans use the sub_question dict key as the tool name, but the
    model is shown (and trained to output) tool_info['name'] — the API-style name.
    This maps the NL sub-question back to the API name for fair structural comparison.
    Returns the API name if found, else the original NL name unchanged.
    """
    # Direct lookup
    if nl_name in tools:
        api_name = tools[nl_name].get("name")
        if api_name:
            return api_name
    # Fuzzy substring match
    for key, tool_info in tools.items():
        if nl_name in key or key in nl_name:
            api_name = tool_info.get("name")
            if api_name:
                return api_name
    return nl_name


def evaluate_plan_vs_gt(gen_steps: List[Dict], gt_steps: List[Dict],
                        tools: Dict[str, Any] = None) -> Dict:
    empty = {
        "valid": False, "error": "",
        "step_count_match": False,
        "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
        "param_accuracy": 0.0, "dependency_accuracy": 0.0,
        "exact_match": False, "functional_match": False, "param_only_match": False,
        "gt_uses_nl_tool_names": False, "step_details": [],
    }
    if not gen_steps:
        return {**empty, "error": "no steps generated"}
    if not gt_steps:
        return {**empty, "error": "no ground truth steps"}

    gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)

    # Remap GT tool names from NL sub-questions to API names if needed
    gen_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gen_steps)
    if gt_uses_nl and tools and not gen_uses_nl:
        gt_steps = [
            {**s, "tool_name": _remap_gt_tool_name(s["tool_name"], tools)}
            for s in gt_steps
        ]
        gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)

    step_count_match     = len(gen_steps) == len(gt_steps)
    correct_exact_tools  = 0
    total_functional     = 0.0
    total_params_correct = 0
    total_params         = 0
    correct_deps         = 0
    total_deps           = 0
    step_details: List[Dict] = []

    for i in range(max(len(gen_steps), len(gt_steps))):
        gen    = gen_steps[i] if i < len(gen_steps) else None
        gt     = gt_steps[i]  if i < len(gt_steps)  else None
        detail: Dict[str, Any] = {"step_id": i}

        if gen and gt:
            exact_ok = gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower()
            detail["exact_tool_correct"] = exact_ok
            if exact_ok:
                correct_exact_tools += 1

            func_score = _functional_tool_match(gen["tool_name"], gt["tool_name"])
            detail["functional_tool_score"] = func_score
            total_functional += func_score

            gt_keys  = set(gt["parameters"].keys())
            gen_keys = set(gen["parameters"].keys())
            common   = gt_keys & gen_keys
            incorrect: List[Dict] = []
            for k in common:
                gt_v  = normalize_value(gt["parameters"][k])
                gen_v = normalize_value(gen["parameters"][k])
                if gt_v == gen_v or gt_v in gen_v or gen_v in gt_v:
                    total_params_correct += 1
                else:
                    incorrect.append({
                        "param": k,
                        "generated":    gen["parameters"][k],
                        "ground_truth": gt["parameters"][k],
                    })
            total_params += len(gt_keys)
            detail["param_comparison"] = {
                "total_gt_params": len(gt_keys),
                "correct":  len(common) - len(incorrect),
                "missing":  list(gt_keys - gen_keys),
                "extra":    list(gen_keys - gt_keys),
                "incorrect": incorrect,
            }

            gt_refs  = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            total_deps  += len(gt_refs)
            correct_deps += len(gt_refs & gen_refs)
            detail["dependency_refs_match"] = (gt_refs == gen_refs)
        else:
            detail["exact_tool_correct"]    = False
            detail["functional_tool_score"] = 0.0
            detail["param_comparison"]      = None
            detail["dependency_refs_match"] = False

        step_details.append(detail)

    n_gt           = len(gt_steps)
    exact_tool_acc = correct_exact_tools / n_gt
    functional_acc = total_functional / n_gt
    param_accuracy = total_params_correct / total_params if total_params > 0 else 0.0
    dep_accuracy   = correct_deps / total_deps if total_deps > 0 else 1.0

    return {
        "valid":                    True,
        "gt_uses_nl_tool_names":    gt_uses_nl,
        "step_count_match":         step_count_match,
        "generated_steps":          len(gen_steps),
        "ground_truth_steps":       n_gt,
        "exact_tool_accuracy":      exact_tool_acc,
        "functional_tool_accuracy": functional_acc,
        "param_accuracy":           param_accuracy,
        "dependency_accuracy":      dep_accuracy,
        "exact_match":              step_count_match and exact_tool_acc == 1.0 and param_accuracy == 1.0,
        "functional_match":         step_count_match and functional_acc >= 0.5 and param_accuracy >= 0.5,
        "param_only_match":         param_accuracy >= 0.5,
        "step_details":             step_details,
    }


# ── Judge client ─────────────────────────────────────────────────────────────

def _format_tools_for_judge(tools: Dict[str, Any]) -> str:
    if not tools:
        return ""
    lines = ["Available Tools:"]
    seen: Dict[str, Any] = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in seen:
            seen[name] = tool_info
    for name, tool_info in seen.items():
        props      = tool_info.get("parameters", {}).get("properties", {})
        params_str = ", ".join(f"{k}: {v.get('type','any')}" for k, v in props.items())
        lines.append(f"- {name}({params_str})")
    return "\n".join(lines)


def score_plan_via_judge(
    query: str,
    plan_str: str,
    tools: Dict,
    max_tokens: int = 300,
    retries: int = 3,
) -> Dict[str, Any]:
    tools_str    = _format_tools_for_judge(tools)
    user_content = f"""Query: {query}

{tools_str}

Plan to Evaluate:
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
  "issues": [...],
  "confidence": <float>
}}"""

    payload = {
        "model": "judge",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens":  max_tokens,
    }

    content = ""
    for attempt in range(retries):
        try:
            resp = requests.post(JUDGE_SERVER_URL, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "```json" in content:
                content = content[content.find("```json") + 7 : content.rfind("```")].strip()
            elif "```" in content:
                content = content[content.find("```") + 3 : content.rfind("```")].strip()
            if not content.endswith("}"):
                last = content.rfind("}")
                if last != -1:
                    content = content[:last + 1]
            annotation = json.loads(content)
            annotation["quality_score"] = max(0, min(100, int(annotation.get("quality_score", 50))))
            annotation["confidence"]    = max(0.0, min(1.0, float(annotation.get("confidence", 0.5))))
            annotation["_full_parse"]   = True
            return annotation
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt < retries - 1:
                time.sleep(2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            try:
                m = re.search(r'"quality_score"\s*:\s*(\d+)', content)
                if m:
                    return {
                        "quality_score": max(0, min(100, int(m.group(1)))),
                        "success_prediction": "uncertain",
                        "reasoning":    "partial parse",
                        "issues": [], "confidence": 0.5, "_full_parse": False,
                    }
            except Exception:
                pass
            break

    return {
        "quality_score": 0, "success_prediction": "no",
        "reasoning": "judge call failed", "issues": [], "confidence": 0.0,
        "_full_parse": False,
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_test_parquet(parquet_path: str, perfect_only: bool = False) -> List[Dict]:
    import pyarrow.parquet as pq
    table         = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    examples: List[Dict] = []

    for i in range(len(extra_infos)):
        extra_info = extra_infos[i]
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        if not isinstance(extra_info, dict):
            extra_info = {}

        reward_model = reward_models[i]
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        if not isinstance(reward_model, dict):
            reward_model = {}

        data_json = extra_info.get("data_json", "{}")
        if isinstance(data_json, str):
            data_json = json.loads(data_json)
        if not isinstance(data_json, dict):
            data_json = {}

        error_type    = str(extra_info.get("error_type",    "none"))
        quality_score = int(extra_info.get("quality_score", 0))

        if perfect_only and not (error_type == "none" and quality_score >= 100):
            continue

        examples.append({
            "question":      data_json.get("question", ""),
            "tools":         data_json.get("tools", {}),
            "ground_truth":  reward_model.get("ground_truth", ""),
            "error_type":    error_type,
            "quality_score": quality_score,
            "query_id":      extra_info.get("query_id", -1),
        })

    return examples


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_alpha_umi(
    agent: AlphaUMiAgent,
    examples: List[Dict],
    perfect_gt_by_qid: Dict[int, str], dataset: str,
    max_steps: int = MAX_PLANNER_STEPS,
    judge_max_tokens: int = 300,
    mode_label: str = "single-model",
    return_trajectory: bool = False,
) -> List[Dict]:
    results: List[Dict] = []
    empty_plan_count = 0

    for ex in tqdm(examples, desc=f"α-UMi ({mode_label}) evaluation"):
        question     = ex["question"]
        tools        = ex["tools"]
        ground_truth = ex["ground_truth"]
        gt_steps     = parse_plan_steps(ground_truth)

        # Run α-UMi agent
        plan_str, trajectory = agent.run(question, tools, max_steps=max_steps)

        if not plan_str.strip():
            empty_plan_count += 1

        # Score via judge
        judge_ann = score_plan_via_judge(
            question, plan_str or "(empty plan)", tools,
            max_tokens=judge_max_tokens,
        )

        # Structural evaluation
        gen_steps = parse_plan_steps(plan_str)
        # Perfect-GT fix: artifact error types have deliberately flawed GT
        if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                and ex["query_id"] in perfect_gt_by_qid):
            _struct_gt_steps = parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        else:
            _struct_gt_steps = gt_steps
        struct_eval = evaluate_plan_vs_gt(gen_steps, _struct_gt_steps, tools=tools)

        judge_success  = judge_ann["quality_score"] >= 80
        ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)

        error_type_handled = (
            judge_success
            if ref_is_perfect
            else judge_ann["quality_score"] >= ex["quality_score"]
        )

        result: Dict[str, Any] = {
            # Identifiers
            "dataset": dataset,
            "query_id":          ex["query_id"],
            "question":          question,
            "error_type":        ex["error_type"],
            "ref_quality_score": ex["quality_score"],
            "ref_is_perfect":    ref_is_perfect,

            # Plans
            "ground_truth":   ground_truth,
            "generated_plan": plan_str,
            "n_planner_steps": len([t for t in trajectory if t["role"] == "planner"]),
            "n_caller_steps":  len([t for t in trajectory if t["role"] == "caller"]),

            # Judge scores
            "judge_success":      judge_success,
            "judge_score":        judge_ann["quality_score"],
            "judge_success_pred": judge_ann["success_prediction"],
            "judge_confidence":   judge_ann["confidence"],
            "judge_full_parse":   judge_ann.get("_full_parse", False),

            # Structural metrics vs ground truth
            "gt_uses_nl_tool_names":      struct_eval["gt_uses_nl_tool_names"],
            "exact_match":                struct_eval["exact_match"],
            "functional_match":           struct_eval["functional_match"],
            "param_only_match":           struct_eval["param_only_match"],
            "step_count_match":           struct_eval["step_count_match"],
            "generated_n_steps":          struct_eval.get("generated_steps", 0),
            "gt_n_steps":                 struct_eval.get("ground_truth_steps", len(gt_steps)),
            "exact_tool_accuracy":        struct_eval["exact_tool_accuracy"],
            "functional_tool_accuracy":   struct_eval["functional_tool_accuracy"],
            "param_accuracy":             struct_eval["param_accuracy"],
            "dependency_accuracy":        struct_eval["dependency_accuracy"],

            # Reference agreement
            "error_type_handled":    error_type_handled,
            "judge_agrees_with_ref": (ref_is_perfect == judge_success),

            # α-UMi metadata
            "alpha_umi_mode":      mode_label,
            "max_planner_steps":   max_steps,
        }

        if return_trajectory:
            result["raw_trajectory"] = trajectory

        results.append(result)

    if empty_plan_count > 0:
        print(f"\n  ⚠  {empty_plan_count}/{len(examples)} examples produced no parseable plan steps.")
        print("     Check the caller's output format.")

    return results


# ── Statistics (mirrors react_baseline.py compute_stats) ─────────────────────

def compute_stats(results: List[Dict], label: str) -> Dict:
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}

    scores     = [r["judge_score"]              for r in results]
    func_tools = [r["functional_tool_accuracy"] for r in results]
    param_accs = [r["param_accuracy"]           for r in results]
    dep_accs   = [r["dependency_accuracy"]      for r in results]

    judge_success_rate    = float(np.mean([r["judge_success"]      for r in results]))
    error_handled_rate    = float(np.mean([r["error_type_handled"] for r in results]))
    exact_match_rate      = float(np.mean([r["exact_match"]        for r in results]))
    functional_match_rate = float(np.mean([r["functional_match"]   for r in results]))
    param_only_match_rate = float(np.mean([r["param_only_match"]   for r in results]))
    step_match_rate       = float(np.mean([r["step_count_match"]   for r in results]))
    full_parse_rate       = float(np.mean([r.get("judge_full_parse", False) for r in results]))
    empty_plan_rate       = float(np.mean([r["generated_n_steps"] == 0 for r in results]))

    error_types = sorted(set(r["error_type"] for r in results))
    per_error: Dict[str, Any] = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n":                       len(sub),
            "judge_success_rate":      float(np.mean([r["judge_success"]             for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"]        for r in sub])),
            "mean_judge_score":        float(np.mean([r["judge_score"]               for r in sub])),
            "functional_tool_acc":     float(np.mean([r["functional_tool_accuracy"]  for r in sub])),
            "mean_param_accuracy":     float(np.mean([r["param_accuracy"]            for r in sub])),
            "exact_match_rate":        float(np.mean([r["exact_match"]               for r in sub])),
            "functional_match_rate":   float(np.mean([r["functional_match"]          for r in sub])),
            "param_only_match_rate":   float(np.mean([r["param_only_match"]          for r in sub])),
            "step_count_match_rate":   float(np.mean([r["step_count_match"]          for r in sub])),
        }

    success_dist: Dict[str, Any] = {}
    for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100 * c / n, 1)}

    stats = {
        "label":              label,
        "dataset": results[0].get("dataset", "unknown"),
        "method":             "α-UMi (prompting)",
        "alpha_umi_mode":     results[0].get("alpha_umi_mode", "unknown"),
        "n_examples":         n,
        "gt_uses_nl_tools":   bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(full_parse_rate, 3),
        "empty_plan_rate":    round(empty_plan_rate, 3),

        "accuracy": {
            "judge_success_rate":  round(judge_success_rate, 3),
            "error_handled_rate":  round(error_handled_rate, 3),
        },
        "judge_scores": {
            "mean":       round(float(np.mean(scores)), 2),
            "median":     round(float(np.median(scores)), 2),
            "std":        round(float(np.std(scores)), 2),
            "pct_gte_80": round(100 * sum(s >= 80  for s in scores) / n, 1),
            "pct_eq_100": round(100 * sum(s == 100 for s in scores) / n, 1),
        },
        "structural": {
            "exact_match_rate":         round(exact_match_rate, 3),
            "functional_match_rate":    round(functional_match_rate, 3),
            "param_only_match_rate":    round(param_only_match_rate, 3),
            "step_count_match_rate":    round(step_match_rate, 3),
            "mean_functional_tool_acc": round(float(np.mean(func_tools)), 3),
            "mean_param_accuracy":      round(float(np.mean(param_accs)), 3),
            "mean_dependency_accuracy": round(float(np.mean(dep_accs)), 3),
        },
        "success_prediction_dist": success_dist,
        "per_error_type":          per_error,
    }

    # ── Print ────────────────────────────────────────────────────────────────
    W = 70
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  Method    : α-UMi ({results[0].get('alpha_umi_mode','?')} mode)")
    print(f"  N examples: {n}")
    if stats["gt_uses_nl_tools"]:
        print(f"  ⚠  GT uses NL tool names — exact_tool_accuracy unreliable")
    if empty_plan_rate > 0.05:
        print(f"  ⚠  Empty plan rate: {100*empty_plan_rate:.1f}% — caller not following format")
    if full_parse_rate < 0.9:
        print(f"  ⚠  Judge full-parse rate: {100*full_parse_rate:.0f}%")

    print(f"\n  ── Primary Accuracy ──────────────────────────────────────────")
    print(f"  Judge success (score≥80) : {100*judge_success_rate:.1f}%")
    print(f"  Error type handled       : {100*error_handled_rate:.1f}%")

    print(f"\n  ── Judge Scores ──────────────────────────────────────────────")
    print(f"  Mean / Median / Std : {stats['judge_scores']['mean']:.1f} / "
          f"{stats['judge_scores']['median']:.1f} / {stats['judge_scores']['std']:.1f}")
    print(f"  ≥80 (good)          : {stats['judge_scores']['pct_gte_80']:.1f}%")
    print(f"  =100 (perfect)      : {stats['judge_scores']['pct_eq_100']:.1f}%")

    print(f"\n  ── Structural Metrics (vs GT) ─────────────────────────────────")
    print(f"  Exact match              : {100*exact_match_rate:.1f}%")
    print(f"  Functional match         : {100*functional_match_rate:.1f}%")
    print(f"  Param-only match (≥50%) : {100*param_only_match_rate:.1f}%")
    print(f"  Step count match         : {100*step_match_rate:.1f}%")
    print(f"  Mean functional tool acc : {np.mean(func_tools):.3f}")
    print(f"  Mean param accuracy      : {np.mean(param_accs):.3f}")
    print(f"  Mean dependency accuracy : {np.mean(dep_accs):.3f}")

    print(f"\n  ── Success Prediction ────────────────────────────────────────")
    for pred, d in success_dist.items():
        print(f"    {pred:12s}: {d['count']:3d}  ({d['pct']:.1f}%)")

    if len(error_types) > 1:
        print(f"\n  ── Per Error-Type ─────────────────────────────────────────────")
        hdr = (f"  {'Error Type':28s}  {'N':>4}  {'Success%':>8}  "
               f"{'Handled%':>8}  {'Judge':>6}  {'FuncTool%':>9}  "
               f"{'Param%':>6}  {'FuncMatch%':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for et, d in per_error.items():
            print(
                f"  {et:28s}  {d['n']:>4}  "
                f"{100*d['judge_success_rate']:>8.1f}  "
                f"{100*d['error_type_handled_rate']:>8.1f}  "
                f"{d['mean_judge_score']:>6.1f}  "
                f"{100*d['functional_tool_acc']:>9.1f}  "
                f"{100*d['mean_param_accuracy']:>6.1f}  "
                f"{100*d['functional_match_rate']:>10.1f}"
            )
    print()
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="α-UMi baseline for ToolHop tool planning (offline mode)"
    )
    parser.add_argument("--planner-model", required=True,
                        help="HF model path for the planner (and caller in single-model mode)")
    parser.add_argument("--caller-model",  default=None,
                        help="HF model path for the caller (multi-model mode only)")
    parser.add_argument("--test-parquet",  required=True)
    parser.add_argument("--mode",          default="single-model",
                        choices=["single-model", "multi-model"],
                        help="single-model: one model plays both planner+caller roles. "
                             "multi-model: separate models (pass --caller-model).")
    parser.add_argument("--max-steps",     type=int,   default=MAX_PLANNER_STEPS,
                        help="Max Planner→Caller iterations per query")
    parser.add_argument("--judge-max-tokens", type=int, default=300)
    parser.add_argument("--output",        default="alpha_umi_results.json")
    parser.add_argument("--stats-output",  default=None)
    parser.add_argument("--return-trajectory", action="store_true",
                        help="Save raw Planner/Caller/Observation trajectory in results")
    parser.add_argument("--planner-device", default="cuda:0")
    parser.add_argument("--caller-device",  default="cuda:0")
    parser.add_argument("--judge_url",     default="http://localhost:8002/v1/chat/completions")
    parser.add_argument("--perfect-only",  action="store_true")
    parser.add_argument("--full",          action="store_true")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Evaluate only first N examples (sanity check)")
    parser.add_argument("--dataset", default="auto",
                        choices=["auto", "toolhop", "nestful"],
                        help="Dataset for labels and metadata. "
                             "'auto' detects from the parquet's "
                             "data_source field.")
    args = parser.parse_args()

    # Resolve dataset (auto-detect from parquet data_source, or explicit)
    dataset = resolve_dataset(args.dataset, args.test_parquet)
    print(f"\nDataset: {dataset_label(dataset)} ({dataset})")

    if not args.perfect_only and not args.full:
        parser.error("Specify at least one of --perfect-only or --full")

    if args.mode == "multi-model" and args.caller_model is None:
        parser.error("--mode multi-model requires --caller-model")

    stats_output = args.stats_output or args.output.replace(".json", ".stats.json")

    global JUDGE_SERVER_URL
    JUDGE_SERVER_URL = args.judge_url

    # Check judge
    try:
        r = requests.get(JUDGE_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5)
        print(f"✅ Judge server healthy: {r.json()}")
    except Exception as e:
        print(f"❌ Judge server not reachable: {e}")
        return

    # Load models
    planner_model = LLMModel(args.planner_model, device=args.planner_device, label="Planner")

    if args.mode == "multi-model":
        caller_model = LLMModel(args.caller_model, device=args.caller_device, label="Caller")
    else:
        caller_model = None  # reuse planner

    agent      = AlphaUMiAgent(planner=planner_model, caller=caller_model)
    mode_label = args.mode
    all_output = {"config": {**vars(args), "resolved_dataset": dataset}, "runs": {}}
    all_stats  = {"config": {**vars(args), "resolved_dataset": dataset}, "runs": {}}

    # Load perfect GT lookup (needed for artifact error type structural comparison)
    perfect_gt_by_qid = load_perfect_gt_from_parquet(args.test_parquet)

    if args.perfect_only:
        print("\nLoading perfect-only test examples...")
        examples = load_test_parquet(args.test_parquet, perfect_only=True)
        if args.limit:
            examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_alpha_umi(
            agent, examples,
            perfect_gt_by_qid=perfect_gt_by_qid, dataset=dataset,
            max_steps=args.max_steps,
            judge_max_tokens=args.judge_max_tokens,
            mode_label=mode_label,
            return_trajectory=args.return_trajectory,
        )
        stats = compute_stats(results, f"PERFECT-ONLY  α-UMi ({mode_label}) — {dataset_label(dataset)}")
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"]  = stats

    if args.full:
        print("\nLoading full test set...")
        examples = load_test_parquet(args.test_parquet, perfect_only=False)
        if args.limit:
            examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_alpha_umi(
            agent, examples,
            perfect_gt_by_qid=perfect_gt_by_qid, dataset=dataset,
            max_steps=args.max_steps,
            judge_max_tokens=args.judge_max_tokens,
            mode_label=mode_label,
            return_trajectory=args.return_trajectory,
        )
        stats = compute_stats(results, f"FULL TEST SET  α-UMi ({mode_label}) — {dataset_label(dataset)}")
        all_output["runs"]["full"] = results
        all_stats["runs"]["full"]  = stats

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"Results saved to: {args.output}")

    with open(stats_output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats   saved to: {stats_output}")


if __name__ == "__main__":
    main()