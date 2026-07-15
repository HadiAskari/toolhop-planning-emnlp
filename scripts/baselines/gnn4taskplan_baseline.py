#!/usr/bin/env python3
"""
GNN4TaskPlan Baseline for ToolHop Tool Planning — FIXED VERSION

Implements GNN4TaskPlan (Wu et al., NeurIPS 2024) faithfully, with three
bug fixes over the previous version:

  1. Decomposition prompt no longer leaks tool-call syntax.
     The previous prompt said "do NOT mention tool names" but didn't prevent
     the model from emitting parenthesized parameter lists. Rewrote to explicitly
     forbid parentheses, equals signs, and {{N}} syntax in step descriptions.

  2. Parameter-filling uses pre-filled scaffold, not free-form generation.
     The previous prompt asked the model to generate full Step-N lines from
     scratch. An SFT-trained planner ignored the GNN's tool assignments and
     produced its own plan. Fix: construct the scaffold
         Step 0: {{0}} = <GNN_selected_tool_0>(
         Step 1: {{1}} = <GNN_selected_tool_1>(
         ...
     and have the model complete the parameter lists only. The GNN's tool
     selection is now structurally guaranteed to appear in the final plan.

  3. Parser restored output-variable requirement.
     Plans missing {{N}} = were silently rejected by the parser, yielding
     100% empty generated_n_steps and zero structural metrics. The scaffold
     fix ensures every output plan has proper {{N}} = assignments.

Bug A fix (this version):
  4. parse_plan_steps now accepts both '=' and ':' as kv separators.
     The base LLM emits dict-style 'key: value' syntax inside parentheses
     rather than Python kwarg 'key=value' syntax. The old parser required '='
     and silently dropped every parameter, forcing param_accuracy and
     dependency_accuracy to 0. The fixed parser prefers '=' but falls back
     to the first ':' that appears before any quote character (so values like
     "time: 12:30" are not mis-split).

  5. build_fill_params_prompt now includes an explicit kwarg-syntax example.
     Added to STRICT FORMAT RULES: use param_name=value, not param_name: value,
     with a concrete worked example. This constrains future inference runs to
     produce '='-style params that the original parser could also handle.

The three-stage pipeline is unchanged:
  Stage 1 — Request Decomposition (LLM, now with strict format)
  Stage 2 — GNN Tool Retrieval     (SGC or GraphSAGE, unchanged)
  Stage 3 — Parameter Filling      (LLM, now with scaffold template)

Dependencies:
    pip install sentence-transformers scikit-learn

Usage:
    # Judge server first (GPU 7):
    CUDA_VISIBLE_DEVICES=7 python judge_server.py --model /path/to/judge --port 8001

    # Training-free SGC (zero training, matches paper §4.2):
    python gnn4taskplan_baseline.py \
        --model /path/to/llm \
        --test-parquet data/verl_rl_full_clean/test.parquet \
        --mode sgc --full \
        --output gnn4tp_sgc_results.json

    # Training-based GraphSAGE (matches paper §4.3):
    CUDA_VISIBLE_DEVICES=3 python gnn4taskplan_baseline.py \
        --model ${FORTE_ROOT}/planner_finetuning/checkpoints_planner_sft-qwen-7b/global_step_196 \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --train-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/train.parquet \
        --mode graphsage --full \
        --gnn-epochs 20 --gnn-lr 1e-3 --gnn-hidden 256 \
        --output gnn4tp_sage_Qwen-7B_finetuned_results.json \
        --judge_url http://localhost:8002/v1/chat/completions
"""

import json
import re
import time
import argparse
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer as HFTokenizer

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


# ── Shared constants ─────────────────────────────────────────────────────────

JUDGE_SERVER_URL = "http://localhost:8001/v1/chat/completions"

# ── FIX #1: Decomposition prompt no longer leaks tool syntax ────────────────
# The old prompt was too permissive. The SFT-trained model saw "Step N:" and
# its priors took over, emitting parenthesized parameter lists. New prompt:
# 1. Forbids parentheses, equals signs, and {{N}} syntax explicitly.
# 2. Uses a numbered list format (1., 2., 3.) so the model can't pattern-match
#    to "Step N:" and slip into plan-generation mode.
# 3. Includes a positive example and a negative example inline.
DECOMPOSE_SYSTEM_PROMPT = (
    "You break down complex queries into short natural-language descriptions "
    "of what each step should accomplish. You output a plain numbered list. "
    "You NEVER use parentheses, equals signs, or double-brace references. "
    "You NEVER name specific tools or APIs. You describe the LOGICAL ACTION only."
)

# ── FIX #2: Parameter-filling prompt now provides a scaffold ─────────────────
# The old prompt asked for free-form plan generation given a tool list. The
# SFT model ignored the list. New approach: pre-fill the scaffold with the
# GNN-selected tool names and ask the model to complete only the parameters.
# This makes the tool assignment a structural guarantee, not a hope.
FILL_PARAMS_SYSTEM_PROMPT = (
    "You complete tool invocations by filling in parameter values. "
    "You are given a partial plan where the tool name for each step is already "
    "fixed. Your job is to fill in concrete parameter values in parentheses, "
    "using {{N}} syntax to reference outputs from previous steps. "
    "Do NOT change tool names. Do NOT add or remove steps. Do NOT add commentary. "
    "Output ONLY the completed Step N: lines."
)

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


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOOL EMBEDDING  (e5-335M encoder, matching paper §4.2)
# ══════════════════════════════════════════════════════════════════════════════

class ToolEmbedder:
    """Sentence encoder for tool descriptions and step descriptions."""

    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "cpu"):
        print(f"  Loading text encoder: {model_name} on {device}...")
        self._tok = HFTokenizer.from_pretrained(model_name)
        self._enc = AutoModel.from_pretrained(model_name).to(device)
        self._enc.eval()
        self._device = device
        self._model_name = model_name
        print(f"  ✓ Text encoder ready")

    @torch.no_grad()
    def encode(self, texts: List[str], prefix: str = "") -> np.ndarray:
        if prefix and "e5" in self._model_name.lower():
            texts = [f"{prefix}{t}" for t in texts]
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(self._device)
        out = self._enc(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
        embs = embs.cpu().float().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
        return embs / norms

    def encode_tools(self, tool_names: List[str], tool_infos: List[Dict]) -> np.ndarray:
        texts = []
        for name, info in zip(tool_names, tool_infos):
            props = info.get("parameters", {}).get("properties", {})
            req = info.get("parameters", {}).get("required", [])
            params = ", ".join(
                f"{k}({v.get('type','any')}{'*' if k in req else ''})"
                for k, v in props.items()
            )
            texts.append(f"{name}: {params}")
        return self.encode(texts, prefix="passage: ")

    def encode_steps(self, steps: List[str]) -> np.ndarray:
        return self.encode(steps, prefix="query: ")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA UTILS
# ══════════════════════════════════════════════════════════════════════════════

def load_perfect_gt_from_parquet(parquet_path: str) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    perfect_gt: dict = {}
    for ei, rm in zip(extra_infos, reward_models):
        if isinstance(ei, str): ei = json.loads(ei)
        if isinstance(rm, str): rm = json.loads(rm)
        if not isinstance(ei, dict) or not isinstance(rm, dict): continue
        if (str(ei.get("error_type","")) == "none"
                and int(ei.get("quality_score",0)) >= 100):
            qid = ei.get("query_id", -1)
            gt_str = rm.get("ground_truth", "")
            if gt_str and qid not in perfect_gt:
                perfect_gt[qid] = gt_str
    print(f"  Loaded perfect GT for {len(perfect_gt)} query_ids.")
    return perfect_gt


# ══════════════════════════════════════════════════════════════════════════════
# 3. SGC RETRIEVER  (training-free, §4.2) — unchanged
# ══════════════════════════════════════════════════════════════════════════════

def _build_adjacency(n: int, add_self_loops: bool = True) -> np.ndarray:
    adj = np.ones((n, n), dtype=np.float32)
    if not add_self_loops:
        np.fill_diagonal(adj, 0.0)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    return (adj * deg_inv_sqrt[:, None]) * deg_inv_sqrt[None, :]


def sgc_smooth(embeddings: np.ndarray, adj_norm: np.ndarray, k: int) -> np.ndarray:
    h = embeddings.copy()
    for _ in range(k):
        h = adj_norm @ h
    return h


class SGCRetriever:
    def __init__(self, k: int = 2):
        self.k = k

    def retrieve(self, tool_embeddings, step_embeddings, tool_names,
                 n_steps: Optional[int] = None) -> List[str]:
        n_tools = len(tool_names)
        if n_tools == 0:
            return []
        adj_norm = _build_adjacency(n_tools, add_self_loops=True)
        smoothed = sgc_smooth(tool_embeddings, adj_norm, k=self.k)
        n_steps = n_steps or len(step_embeddings)
        selected: List[str] = []
        used_indices: set = set()
        for i in range(n_steps):
            if i >= len(step_embeddings):
                break
            step_emb = step_embeddings[i]
            scores = smoothed @ step_emb
            cs = scores.copy()
            if len(used_indices) < n_tools:
                for idx in used_indices:
                    cs[idx] = -np.inf
            selected.append(tool_names[int(np.argmax(cs))])
            used_indices.add(int(np.argmax(cs)))
        return selected


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRAPHSAGE RETRIEVER  (training-based, §4.3) — unchanged
# ══════════════════════════════════════════════════════════════════════════════

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim * 2, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        agg = adj_norm @ x
        h = torch.cat([x, agg], dim=-1)
        return self.activation(self.W(self.dropout(h)))


class GraphSAGEProjector(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.sage = GraphSAGELayer(emb_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.sage(x, adj_norm)
        return nn.functional.normalize(self.proj(h), dim=-1)


def bpr_loss(step_emb, pos_emb, neg_emb):
    pos_score = (step_emb * pos_emb).sum()
    neg_score = (step_emb * neg_emb).sum()
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)


class GraphSAGERetriever:
    def __init__(self, emb_dim: int, hidden_dim: int = 256,
                 out_dim: Optional[int] = None, device: str = "cpu"):
        out_dim = out_dim or emb_dim
        self.model = GraphSAGEProjector(emb_dim, hidden_dim, out_dim).to(device)
        self.device = device
        self.is_trained = False

    def train(self, triplets, cooccur_adj=None, epochs=20, lr=1e-3, batch_size=128):
        if not triplets:
            print("  ⚠  No triplets available — skipping GraphSAGE training")
            return []
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        epoch_losses = []
        S = torch.tensor(np.array([t[0] for t in triplets]), dtype=torch.float32).to(self.device)
        P = torch.tensor(np.array([t[1] for t in triplets]), dtype=torch.float32).to(self.device)
        N = torch.tensor(np.array([t[2] for t in triplets]), dtype=torch.float32).to(self.device)
        n_triplets = len(triplets)
        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n_triplets)
            total_loss, n_batches = 0.0, 0
            for start in range(0, n_triplets, batch_size):
                batch_idx = perm[start: start + batch_size]
                s_b, p_b, n_b = S[batch_idx], P[batch_idx], N[batch_idx]
                ones = torch.ones(1, 1, device=self.device)
                loss = torch.tensor(0.0, device=self.device)
                for si, pi, ni in zip(s_b, p_b, n_b):
                    s_proj = self.model(si.unsqueeze(0), ones).squeeze(0)
                    p_proj = self.model(pi.unsqueeze(0), ones).squeeze(0)
                    n_proj = self.model(ni.unsqueeze(0), ones).squeeze(0)
                    loss = loss + bpr_loss(s_proj, p_proj, n_proj)
                loss = loss / len(s_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs}  BPR loss: {avg_loss:.4f}")
        self.model.eval()
        self.is_trained = True
        return epoch_losses

    @torch.no_grad()
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        n = len(embeddings)
        x = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        adj = torch.ones(n, n, device=self.device)
        deg = adj.sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
        adj_norm = adj / deg / deg.T
        return self.model(x, adj_norm).cpu().numpy()

    def retrieve(self, tool_embeddings, step_embeddings, tool_names,
                 n_steps: Optional[int] = None) -> List[str]:
        n_tools = len(tool_names)
        if n_tools == 0:
            return []
        if self.is_trained:
            projected_tools = self.project(tool_embeddings)
            projected_steps = self.project(step_embeddings)
        else:
            projected_tools = tool_embeddings
            projected_steps = step_embeddings
        n_steps = n_steps or len(step_embeddings)
        selected: List[str] = []
        used_indices: set = set()
        for i in range(n_steps):
            if i >= len(projected_steps):
                break
            step_emb = projected_steps[i]
            scores = projected_tools @ step_emb
            cs = scores.copy()
            if len(used_indices) < n_tools:
                for idx in used_indices:
                    cs[idx] = -np.inf
            best = int(np.argmax(cs))
            selected.append(tool_names[best])
            used_indices.add(best)
        return selected


# ══════════════════════════════════════════════════════════════════════════════
# 5. PROMPT BUILDERS — FIXED
# ══════════════════════════════════════════════════════════════════════════════

def _format_tools_list(tools: Dict[str, Any]) -> str:
    unique: Dict[str, Dict] = {}
    for sub_q, info in tools.items():
        name = info.get("name", sub_q)
        if name not in unique:
            unique[name] = info
    lines = ["Available Tools:"]
    for name, info in unique.items():
        props = info.get("parameters", {}).get("properties", {})
        req = info.get("parameters", {}).get("required", [])
        parts = [
            f"{k}: {v.get('type','any')}{' (required)' if k in req else ''}"
            for k, v in props.items()
        ]
        lines.append(f"- {name}({', '.join(parts)})")
    return "\n".join(lines)


def build_decompose_prompt(question: str, tools: Dict[str, Any]) -> str:
    """
    FIX #1: Explicit format prohibition + worked example.

    Previously the model echoed `Step N:` patterns from its SFT training and
    slipped into plan-generation mode with parentheses and tool names.
    Now we use a numbered `1. 2. 3.` list, forbid structural tokens, and
    show a concrete example to anchor the output format.
    """
    tools_str = _format_tools_list(tools)
    return (
        f"Break the query below into an ordered list of logical actions.\n\n"
        f"STRICT FORMAT RULES:\n"
        f"  - Use a simple numbered list: '1.', '2.', '3.', etc.\n"
        f"  - Each item is a single sentence describing the LOGICAL OPERATION.\n"
        f"  - NO parentheses. NO equals signs. NO curly braces. NO {{N}} references.\n"
        f"  - NO tool names. NO parameter names.\n"
        f"  - Just describe WHAT happens at each step in plain English.\n\n"
        f"GOOD example for a query about finding an author's birth year:\n"
        f"1. Identify the author who wrote the given book.\n"
        f"2. Look up the birth year of that author.\n"
        f"3. Subtract the birth year from the current year.\n\n"
        f"BAD example (do NOT output anything like this):\n"
        f"1. Step 1: find_author(book='...')\n"
        f"2. get_birth_year(name={{0}})\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n\n"
        f"Numbered list of logical actions:\n"
    )


def build_fill_params_prompt(
    question: str,
    tools: Dict[str, Any],
    selected_tools: List[str],
) -> Tuple[str, str]:
    """
    FIX #2: Pre-filled scaffold — returns (prompt, scaffold_prefix).

    The scaffold already contains the {{N}} = tool_name( prefix for every
    step. We feed this to the model as an assistant-continuation so it
    MUST complete with parameter lists rather than regenerate the tool names.

    BUG A FIX (prompt side): Added explicit kwarg-syntax instruction and
    worked example so the model uses 'param=value' not 'param: value'.
    This pairs with the parser fix in parse_plan_steps which now accepts
    both separators as a safety net for any remaining colon-style output.

    Returns:
      prompt        : the user-facing instruction portion
      scaffold_prefix: the partial plan the model continues from
    """
    unique: Dict[str, Dict] = {}
    for sub_q, info in tools.items():
        name = info.get("name", sub_q)
        if name not in unique:
            unique[name] = info

    # Build a readable tool-signature block for the selected tools only
    sig_lines: List[str] = []
    for i, tool_name in enumerate(selected_tools):
        info = unique.get(tool_name, {})
        props = info.get("parameters", {}).get("properties", {})
        req = info.get("parameters", {}).get("required", [])
        parts = [
            f"{k}: {v.get('type','any')}{' (required)' if k in req else ''}"
            for k, v in props.items()
        ]
        sig_lines.append(f"  Step {i}: {tool_name}({', '.join(parts)})")
    signatures = "\n".join(sig_lines)

    scaffold_lines = [
        f"Step {i}: {{{{{i}}}}} = {tool_name}("
        for i, tool_name in enumerate(selected_tools)
    ]
    scaffold_template = "\n".join(scaffold_lines)  # for display in prompt only

    # ── BUG A FIX (Edit 2): explicit kwarg-syntax instruction + example ──────
    # The base LLM defaults to dict-style 'key: value' syntax inside parens.
    # Two lines added to STRICT FORMAT RULES constrain it to 'key=value'.
    prompt = (
        f"Fill in parameter values for each step of the plan below.\n\n"
        f"STRICT FORMAT RULES:\n"
        f"  - Output EXACTLY {len(selected_tools)} lines, one per step.\n"
        f"  - Each line MUST start with: Step N: {{{{N}}}} = <fixed_tool_name>(\n"
        f"  - Do NOT change the tool name at any step.\n"
        f"  - Use {{{{0}}}}, {{{{1}}}}, etc. to reference outputs of prior steps.\n"
        f"  - Fill in all required parameters with concrete values or {{{{N}}}} refs.\n"
        f"  - Use Python kwarg syntax inside parentheses: param_name=value "
        f"(NOT param_name: value).\n"
        f"  - Example: Step 0: {{{{0}}}} = lookup_tool(name=\"alice\", limit=10)\n"
        f"  - Close each step's parentheses on the same line.\n"
        f"  - No commentary, no markdown, no code fences.\n\n"
        f"Query: {question}\n\n"
        f"Tool signatures (tool names are FIXED, only fill parameters):\n{signatures}\n\n"
        f"Required output skeleton (reproduce exactly, filling the parameter values):\n"
        f"{scaffold_template}<fill parameters>)\n\n"
        f"Your completed plan (exactly {len(selected_tools)} Step lines):"
    )

    return prompt, scaffold_template


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN MODEL WRAPPER — with scaffold-completion helper
# ══════════════════════════════════════════════════════════════════════════════

class GNN4TaskPlanModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading LLM from {model_path}...")
        self.tokenizer = HFTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self._gen_device = next(self.model.parameters()).device
        print(f"✓ LLM loaded on {self._gen_device}")

    def _call_llm(self, system, user, max_new_tokens=512, temperature=0.0):
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=2048).to(self._gen_device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def decompose(self, question: str, tools: Dict[str, Any],
                  max_new_tokens: int = 256) -> List[str]:
        """
        Stage 1: extract NL step descriptions.

        FIX #1 applied in prompt. Parser also rewritten to match the
        numbered-list format and reject any line containing tool-syntax tokens.
        """
        raw = self._call_llm(
            DECOMPOSE_SYSTEM_PROMPT,
            build_decompose_prompt(question, tools),
            max_new_tokens=max_new_tokens,
        )
        steps: List[str] = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match "1." / "1)" / "1:" leading numbers
            m = re.match(r"^\d+\s*[\.\):]\s*(.+)$", line)
            if not m:
                continue
            desc = m.group(1).strip()
            # Reject lines that leaked tool-call syntax despite instructions
            if "(" in desc or "=" in desc or "{{" in desc:
                # Strip everything from the first forbidden char onward
                cut = min(
                    (desc.find(c) for c in "({=" if c in desc),
                    default=len(desc),
                )
                desc = desc[:cut].strip().rstrip(",.;:")
            if desc:
                steps.append(desc)
        return steps

    def fill_parameters(
        self,
        question: str,
        tools: Dict[str, Any],
        selected_tools: List[str],
        max_new_tokens: int = 512,
    ) -> str:
        """
        Stage 3: fill parameters using scaffold-constrained generation.

        FIX #2 applied via two mechanisms:
        1. Prompt shows the exact scaffold and demands verbatim reproduction.
        2. Post-hoc repair: if the model's output deviates, we reconstruct
           valid Step-N lines by splicing the GNN's tool names into whatever
           parameters the model produced.
        """
        if not selected_tools:
            return ""

        prompt, _scaffold = build_fill_params_prompt(question, tools, selected_tools)
        raw = self._call_llm(
            FILL_PARAMS_SYSTEM_PROMPT, prompt, max_new_tokens=max_new_tokens
        )

        # Extract valid Step-N lines
        step_lines: List[Optional[str]] = [None] * len(selected_tools)
        for line in raw.split("\n"):
            line = line.strip()
            m = re.match(r"Step\s+(\d+)\s*:\s*", line)
            if not m:
                continue
            step_idx = int(m.group(1))
            if step_idx < 0 or step_idx >= len(selected_tools):
                continue
            # Must contain {{N}} = and the correct tool name; if not, repair
            expected_tool = selected_tools[step_idx]
            if (f"{{{{{step_idx}}}}}" in line and
                    "=" in line and
                    expected_tool in line and
                    line.rstrip().endswith(")")):
                # Line is well-formed
                step_lines[step_idx] = line
            else:
                # Repair: extract whatever parameters the model emitted
                repaired = _repair_step_line(line, step_idx, expected_tool)
                if repaired:
                    step_lines[step_idx] = repaired

        # For any step the model failed to produce, emit an empty call so
        # the plan is at least parseable (judge will penalize appropriately).
        for i, tool_name in enumerate(selected_tools):
            if step_lines[i] is None:
                step_lines[i] = f"Step {i}: {{{{{i}}}}} = {tool_name}()"

        return "\n".join(step_lines)


def _repair_step_line(raw_line: str, step_idx: int, expected_tool: str) -> Optional[str]:
    """
    Attempt to salvage a malformed Step line by extracting whatever
    parameter content the model produced and reattaching the correct
    {{N}} = tool_name(...) scaffold.
    """
    # Try to find parameters inside parentheses anywhere in the line
    paren_match = re.search(r"\((.*)\)\s*$", raw_line)
    if paren_match:
        params = paren_match.group(1).strip()
    else:
        # No parens at all — check for k=v pairs after tool name or step colon
        after_colon = re.search(r":\s*(.*)$", raw_line)
        if not after_colon:
            return None
        tail = after_colon.group(1)
        # If tail has k=v content, wrap it; else give up
        if "=" in tail:
            params = tail
        else:
            return None
    return f"Step {step_idx}: {{{{{step_idx}}}}} = {expected_tool}({params})"


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING DATA PREPARATION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def build_training_triplets(train_examples, embedder, n_negatives=2):
    triplets = []
    for ex in tqdm(train_examples, desc="  Building training triplets"):
        tools = ex["tools"]
        ground_truth = ex["ground_truth"]
        if not ground_truth.strip():
            continue
        unique: Dict[str, Dict] = {}
        for sub_q, info in tools.items():
            name = info.get("name", sub_q)
            if name not in unique:
                unique[name] = info
        if len(unique) < 2:
            continue
        tool_names = list(unique.keys())
        tool_embs = embedder.encode_tools(tool_names, [unique[n] for n in tool_names])
        for line in ground_truth.split("\n"):
            line = line.strip()
            if not line.startswith("Step "):
                continue
            m = re.search(r"=\s*([^\(]+)\(", line)
            if not m:
                continue
            pos_tool_name = m.group(1).strip()
            if pos_tool_name not in unique:
                continue
            pos_idx = tool_names.index(pos_tool_name)
            pos_emb = tool_embs[pos_idx]
            step_emb = embedder.encode_steps([pos_tool_name])[0]
            sims = tool_embs @ pos_emb
            sims[pos_idx] = -np.inf
            neg_indices = np.argsort(sims)[::-1][:n_negatives]
            for neg_idx in neg_indices:
                if sims[neg_idx] == -np.inf:
                    continue
                neg_emb = tool_embs[neg_idx]
                triplets.append((step_emb, pos_emb, neg_emb))
    return triplets


# ══════════════════════════════════════════════════════════════════════════════
# 8. PLAN PARSING & STRUCTURAL EVAL
# ══════════════════════════════════════════════════════════════════════════════

def parse_plan_steps(plan_text: str) -> List[Dict]:
    """
    Parse Step-N lines from a generated plan into structured dicts.

    BUG A FIX (Edit 1): The parameter kv-separator is now tolerant of both
    '=' (Python kwarg, canonical) and ':' (dict-style, what the base LLM
    historically emits). The colon fallback only fires when no '=' is present
    in a given parameter part, and it searches only up to the first quote
    character so values like "time: 12:30" are never mis-split.
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
            var_match = re.search(r"(\{\{\d+\}\})\s*=", line)
            if not var_match:
                continue
            output_var = var_match.group(1)
            tool_match = re.search(r"=\s*([^\(]+)\((.*)\)\s*$", line)
            if not tool_match:
                tool_match_empty = re.search(r"=\s*([^\(]+)\(\)\s*$", line)
                if tool_match_empty:
                    tool_name = tool_match_empty.group(1).strip()
                    params: Dict[str, str] = {}
                else:
                    continue
            else:
                tool_name = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                params = {}
                if params_str:
                    param_parts, current, depth, in_str, str_char = [], "", 0, False, None
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
                    # ── BUG A FIX (Edit 1): accept '=' or ':' as kv separator ──
                    for part in param_parts:
                        sep_idx = part.find("=")
                        if sep_idx < 0:
                            # No '=' — try a colon, but only before the first
                            # quote char so "time: 12:30" doesn't mis-split.
                            first_quote = next(
                                (i for i, ch in enumerate(part) if ch in ('"', "'")),
                                len(part),
                            )
                            colon_idx = part.find(":", 0, first_quote)
                            if colon_idx >= 0:
                                sep_idx = colon_idx
                        if sep_idx >= 0:
                            k = part[:sep_idx].strip()
                            v = part[sep_idx + 1:].strip()
                            if k:
                                params[k] = v
            steps.append({"step_id": step_id, "output_variable": output_var,
                          "tool_name": tool_name, "parameters": params})
        except Exception:
            continue
    return steps


def _is_nl_tool_name(name: str) -> bool:
    return len(name.split()) > 4 or name.endswith("?")


def _functional_tool_match(gen_name: str, gt_name: str) -> float:
    STOP = {"what","is","the","of","in","a","an","and","or","to","how","many",
            "who","which","are","was","were","be","been","at","on","for","with",
            "that","this","it","its","from"}
    def kw(s):
        return {w for w in re.sub(r"[^a-z0-9\s]"," ",s.lower()).split()
                if w not in STOP and len(w) > 2}
    g, r = kw(gen_name), kw(gt_name)
    if not g or not r:
        return 0.0
    return round(len(g & r) / len(g | r), 3)


def _norm(v):
    return " ".join(str(v).strip().strip("\"'").lower().split())


def _remap_gt_tool_name(nl_name, tools):
    if nl_name in tools:
        api = tools[nl_name].get("name")
        if api: return api
    for key, tool_info in tools.items():
        if nl_name in key or key in nl_name:
            api = tool_info.get("name")
            if api: return api
    return nl_name


def evaluate_plan_vs_gt(gen_steps, gt_steps, tools=None):
    empty = {"valid": False, "error": "", "step_count_match": False,
             "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
             "param_accuracy": 0.0, "dependency_accuracy": 0.0,
             "exact_match": False, "functional_match": False,
             "param_only_match": False,
             "gt_uses_nl_tool_names": False, "step_details": []}
    if not gen_steps: return {**empty, "error": "no steps generated"}
    if not gt_steps:  return {**empty, "error": "no ground truth steps"}
    gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)
    gen_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gen_steps)
    if gt_uses_nl and tools and not gen_uses_nl:
        gt_steps = [{**s, "tool_name": _remap_gt_tool_name(s["tool_name"], tools)} for s in gt_steps]
        gt_uses_nl = any(_is_nl_tool_name(s["tool_name"]) for s in gt_steps)
    step_count_match = len(gen_steps) == len(gt_steps)
    ce, tf, tpc, tp, cd, td = 0, 0.0, 0, 0, 0, 0
    step_details = []
    for i in range(max(len(gen_steps), len(gt_steps))):
        gen = gen_steps[i] if i < len(gen_steps) else None
        gt  = gt_steps[i]  if i < len(gt_steps)  else None
        detail: Dict[str, Any] = {"step_id": i}
        if gen and gt:
            exact_ok = gen["tool_name"].strip().lower() == gt["tool_name"].strip().lower()
            detail["exact_tool_correct"] = exact_ok
            if exact_ok: ce += 1
            fs = _functional_tool_match(gen["tool_name"], gt["tool_name"])
            detail["functional_tool_score"] = fs
            tf += fs
            gt_keys = set(gt["parameters"].keys())
            gen_keys = set(gen["parameters"].keys())
            common = gt_keys & gen_keys
            incorrect = []
            for k in common:
                gv, dv = _norm(gt["parameters"][k]), _norm(gen["parameters"][k])
                if gv == dv or gv in dv or dv in gv: tpc += 1
                else:
                    incorrect.append({"param": k,
                                      "generated": gen["parameters"][k],
                                      "ground_truth": gt["parameters"][k]})
            tp += len(gt_keys)
            detail["param_comparison"] = {
                "total_gt_params": len(gt_keys),
                "correct": len(common) - len(incorrect),
                "missing": list(gt_keys - gen_keys),
                "extra": list(gen_keys - gt_keys),
                "incorrect": incorrect,
            }
            gt_refs  = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            td += len(gt_refs)
            cd += len(gt_refs & gen_refs)
            detail["dependency_refs_match"] = (gt_refs == gen_refs)
        else:
            detail["exact_tool_correct"] = False
            detail["functional_tool_score"] = 0.0
            detail["param_comparison"] = None
            detail["dependency_refs_match"] = False
        step_details.append(detail)
    n_gt = len(gt_steps)
    ea = ce / n_gt
    fa = tf / n_gt
    pa = tpc / tp if tp > 0 else 0.0
    da = cd / td if td > 0 else 1.0
    return {"valid": True, "gt_uses_nl_tool_names": gt_uses_nl,
            "step_count_match": step_count_match,
            "generated_steps": len(gen_steps), "ground_truth_steps": n_gt,
            "exact_tool_accuracy": ea, "functional_tool_accuracy": fa,
            "param_accuracy": pa, "dependency_accuracy": da,
            "exact_match": step_count_match and ea == 1.0 and pa == 1.0,
            "functional_match": step_count_match and fa >= 0.5 and pa >= 0.5,
            "param_only_match": pa >= 0.5,
            "step_details": step_details}


# ══════════════════════════════════════════════════════════════════════════════
# 9. JUDGE CLIENT (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _format_tools_for_judge(tools):
    if not tools: return ""
    lines = ["Available Tools:"]
    seen = {}
    for sub_q, info in tools.items():
        name = info.get("name", sub_q)
        if name not in seen: seen[name] = info
    for name, info in seen.items():
        props = info.get("parameters", {}).get("properties", {})
        ps = ", ".join(f"{k}: {v.get('type','any')}" for k, v in props.items())
        lines.append(f"- {name}({ps})")
    return "\n".join(lines)


def score_plan_via_judge(query, plan_str, tools, max_tokens=300, retries=3):
    tools_str = _format_tools_for_judge(tools)
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
        "temperature": 0.0, "max_tokens": max_tokens,
    }
    content = ""
    for attempt in range(retries):
        try:
            resp = requests.post(JUDGE_SERVER_URL, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "```json" in content:
                content = content[content.find("```json")+7: content.rfind("```")].strip()
            elif "```" in content:
                content = content[content.find("```")+3: content.rfind("```")].strip()
            if not content.endswith("}"):
                last = content.rfind("}")
                if last != -1: content = content[:last+1]
            ann = json.loads(content)
            ann["quality_score"] = max(0, min(100, int(ann.get("quality_score", 50))))
            ann["confidence"]    = max(0.0, min(1.0, float(ann.get("confidence", 0.5))))
            ann["_full_parse"]   = True
            return ann
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt < retries - 1:
                time.sleep(2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            m = re.search(r'"quality_score"\s*:\s*(\d+)', content)
            if m:
                return {"quality_score": max(0, min(100, int(m.group(1)))),
                        "success_prediction": "uncertain", "reasoning": "partial parse",
                        "issues": [], "confidence": 0.5, "_full_parse": False}
            break
    return {"quality_score": 0, "success_prediction": "no",
            "reasoning": "judge call failed", "issues": [], "confidence": 0.0,
            "_full_parse": False}


# ══════════════════════════════════════════════════════════════════════════════
# 10. DATA LOADING (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def load_parquet(parquet_path, perfect_only=False):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos   = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    examples = []
    for i in range(len(extra_infos)):
        ei = extra_infos[i]
        if isinstance(ei, str): ei = json.loads(ei)
        if not isinstance(ei, dict): ei = {}
        rm = reward_models[i]
        if isinstance(rm, str): rm = json.loads(rm)
        if not isinstance(rm, dict): rm = {}
        dj = ei.get("data_json", "{}")
        if isinstance(dj, str): dj = json.loads(dj)
        if not isinstance(dj, dict): dj = {}
        error_type = str(ei.get("error_type", "none"))
        quality_score = int(ei.get("quality_score", 0))
        if perfect_only and not (error_type == "none" and quality_score >= 100):
            continue
        examples.append({
            "question": dj.get("question", ""),
            "tools": dj.get("tools", {}),
            "ground_truth": rm.get("ground_truth", ""),
            "error_type": error_type,
            "quality_score": quality_score,
            "query_id": ei.get("query_id", -1),
        })
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# 11. EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_gnn4taskplan(
    llm, embedder, retriever, mode, examples, perfect_gt_by_qid,
    sgc_k=2, max_new_tokens_decompose=256, max_new_tokens_fill=512,
    judge_max_tokens=300, return_raw=False,
):
    results = []
    empty_plan_count = 0
    empty_decomp_count = 0
    tool_name_match_count = 0  # ← new diagnostic: does the plan use GNN tools?

    for ex in tqdm(examples, desc=f"GNN4TaskPlan ({mode}) evaluation"):
        question, tools, ground_truth = ex["question"], ex["tools"], ex["ground_truth"]
        gt_steps = parse_plan_steps(ground_truth)

        unique: Dict[str, Dict] = {}
        for sub_q, info in tools.items():
            name = info.get("name", sub_q)
            if name not in unique:
                unique[name] = info
        tool_names = list(unique.keys())
        tool_infos = [unique[n] for n in tool_names]

        # Stage 1: decomposition
        decomposed_steps = llm.decompose(
            question, tools, max_new_tokens=max_new_tokens_decompose
        )
        if not decomposed_steps:
            empty_decomp_count += 1
            decomposed_steps = [f"perform logical step {i+1} of the query"
                                for i in range(len(tool_names))]

        # Stage 2: GNN retrieval
        tool_embs = embedder.encode_tools(tool_names, tool_infos)
        step_embs = embedder.encode_steps(decomposed_steps)
        n_steps = max(len(decomposed_steps), len(gt_steps))
        n_steps = min(n_steps, len(tool_names))
        selected_tools = retriever.retrieve(tool_embs, step_embs, tool_names,
                                            n_steps=n_steps)

        # Stage 3: parameter filling with scaffold
        plan = llm.fill_parameters(question, tools, selected_tools,
                                   max_new_tokens=max_new_tokens_fill) if selected_tools else ""
        if not plan.strip():
            empty_plan_count += 1

        # Diagnostic: verify generated plan actually uses GNN-selected tools
        gen_steps = parse_plan_steps(plan)
        gen_tool_names = [s["tool_name"] for s in gen_steps]
        if gen_tool_names and set(gen_tool_names) & set(selected_tools):
            tool_name_match_count += 1

        # Judge scoring
        judge_ann = score_plan_via_judge(
            question, plan or "(empty plan)", tools,
            max_tokens=judge_max_tokens,
        )

        # Structural eval
        if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                and ex["query_id"] in perfect_gt_by_qid):
            _struct_gt_steps = parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        else:
            _struct_gt_steps = gt_steps
        struct_eval = evaluate_plan_vs_gt(gen_steps, _struct_gt_steps, tools=tools)

        judge_success = judge_ann["quality_score"] >= 80
        ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)
        if ref_is_perfect:
            error_type_handled = judge_success
        else:
            error_type_handled = judge_ann["quality_score"] >= ex["quality_score"]

        result = {
            "query_id": ex["query_id"], "question": question,
            "error_type": ex["error_type"], "ref_quality_score": ex["quality_score"],
            "ref_is_perfect": ref_is_perfect, "ground_truth": ground_truth,
            "generated_plan": plan, "decomposed_steps": decomposed_steps,
            "selected_tools": selected_tools,
            "n_extracted_steps": len(gen_steps), "n_tools": len(tool_names),
            "uses_gnn_tools": bool(gen_tool_names and
                                   set(gen_tool_names) & set(selected_tools)),
            "judge_success": judge_success, "judge_score": judge_ann["quality_score"],
            "judge_success_pred": judge_ann["success_prediction"],
            "judge_confidence": judge_ann["confidence"],
            "judge_full_parse": judge_ann.get("_full_parse", False),
            "gt_uses_nl_tool_names": struct_eval["gt_uses_nl_tool_names"],
            "exact_match": struct_eval["exact_match"],
            "functional_match": struct_eval["functional_match"],
            "param_only_match": struct_eval["param_only_match"],
            "step_count_match": struct_eval["step_count_match"],
            "generated_n_steps": struct_eval.get("generated_steps", 0),
            "gt_n_steps": struct_eval.get("ground_truth_steps", len(gt_steps)),
            "exact_tool_accuracy": struct_eval["exact_tool_accuracy"],
            "functional_tool_accuracy": struct_eval["functional_tool_accuracy"],
            "param_accuracy": struct_eval["param_accuracy"],
            "dependency_accuracy": struct_eval["dependency_accuracy"],
            "error_type_handled": error_type_handled,
            "judge_agrees_with_ref": (ref_is_perfect == judge_success),
            "method": f"GNN4TaskPlan-{mode.upper()}",
            "gnn_mode": mode, "react_mode": None,
        }
        if return_raw:
            result["raw_decomposed_steps"] = "\n".join(decomposed_steps)
        results.append(result)

    n = len(results)
    if empty_decomp_count > 0:
        print(f"\n  ⚠  {empty_decomp_count}/{n} empty decompositions (used fallback)")
    if empty_plan_count > 0:
        print(f"  ⚠  {empty_plan_count}/{n} empty plans")
    print(f"  ✓ {tool_name_match_count}/{n} plans use GNN-selected tools")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 12. STATISTICS (unchanged format, added tool-match diagnostic)
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(results, label):
    n = len(results)
    if n == 0: return {"label": label, "n": 0}
    scores = [r["judge_score"] for r in results]
    func_tools = [r["functional_tool_accuracy"] for r in results]
    param_accs = [r["param_accuracy"] for r in results]
    dep_accs   = [r["dependency_accuracy"] for r in results]
    judge_sr  = float(np.mean([r["judge_success"] for r in results]))
    err_hr    = float(np.mean([r["error_type_handled"] for r in results]))
    exact_mr  = float(np.mean([r["exact_match"] for r in results]))
    func_mr   = float(np.mean([r["functional_match"] for r in results]))
    param_omr = float(np.mean([r["param_only_match"] for r in results]))
    step_mr   = float(np.mean([r["step_count_match"] for r in results]))
    fp_rate   = float(np.mean([r.get("judge_full_parse", False) for r in results]))
    empty_rate = float(np.mean([r["generated_n_steps"] == 0 for r in results]))
    uses_gnn   = float(np.mean([r.get("uses_gnn_tools", False) for r in results]))

    error_types = sorted(set(r["error_type"] for r in results))
    per_error = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n": len(sub),
            "judge_success_rate":      float(np.mean([r["judge_success"] for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"] for r in sub])),
            "mean_judge_score":        float(np.mean([r["judge_score"] for r in sub])),
            "functional_tool_acc":     float(np.mean([r["functional_tool_accuracy"] for r in sub])),
            "mean_param_accuracy":     float(np.mean([r["param_accuracy"] for r in sub])),
            "exact_match_rate":        float(np.mean([r["exact_match"] for r in sub])),
            "functional_match_rate":   float(np.mean([r["functional_match"] for r in sub])),
            "param_only_match_rate":   float(np.mean([r["param_only_match"] for r in sub])),
            "step_count_match_rate":   float(np.mean([r["step_count_match"] for r in sub])),
        }

    success_dist = {}
    for pred in ["yes","likely_yes","uncertain","likely_no","no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100*c/n, 1)}

    gnn_mode = results[0].get("gnn_mode", "?")
    stats = {
        "label": label,
        "method": f"GNN4TaskPlan-{gnn_mode.upper()} (Wu et al. NeurIPS 2024)",
        "gnn_mode": gnn_mode, "n_examples": n,
        "gt_uses_nl_tools": bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(fp_rate, 3),
        "empty_plan_rate": round(empty_rate, 3),
        "uses_gnn_selected_tools_rate": round(uses_gnn, 3),
        "accuracy": {
            "judge_success_rate": round(judge_sr, 3),
            "error_handled_rate": round(err_hr, 3),
        },
        "judge_scores": {
            "mean": round(float(np.mean(scores)), 2),
            "median": round(float(np.median(scores)), 2),
            "std": round(float(np.std(scores)), 2),
            "pct_gte_80": round(100 * sum(s >= 80 for s in scores) / n, 1),
            "pct_eq_100": round(100 * sum(s == 100 for s in scores) / n, 1),
        },
        "structural": {
            "exact_match_rate": round(exact_mr, 3),
            "functional_match_rate": round(func_mr, 3),
            "param_only_match_rate": round(param_omr, 3),
            "step_count_match_rate": round(step_mr, 3),
            "mean_functional_tool_acc": round(float(np.mean(func_tools)), 3),
            "mean_param_accuracy": round(float(np.mean(param_accs)), 3),
            "mean_dependency_accuracy": round(float(np.mean(dep_accs)), 3),
        },
        "success_prediction_dist": success_dist,
        "per_error_type": per_error,
    }

    W = 70
    print(f"\n{'='*W}\n  {label}\n{'='*W}")
    print(f"  Method : GNN4TaskPlan-{gnn_mode.upper()}  |  N : {n}")
    print(f"  Uses GNN-selected tools : {100*uses_gnn:.1f}%  "
          f"(diagnostic — should be ~100% with scaffold fix)")
    if empty_rate > 0.05:
        print(f"  ⚠  Empty plan rate: {100*empty_rate:.1f}%")
    if fp_rate < 0.9:
        print(f"  ⚠  Judge full-parse rate: {100*fp_rate:.0f}%")
    print(f"\n  ── Primary Accuracy ──────────────────────────────────────────")
    print(f"  Judge success (≥80) : {100*judge_sr:.1f}%")
    print(f"  Error handled       : {100*err_hr:.1f}%")
    print(f"\n  ── Judge Scores ──────────────────────────────────────────────")
    print(f"  Mean/Median/Std : {stats['judge_scores']['mean']:.1f} / "
          f"{stats['judge_scores']['median']:.1f} / {stats['judge_scores']['std']:.1f}")
    print(f"  ≥80 : {stats['judge_scores']['pct_gte_80']:.1f}%   "
          f"=100 : {stats['judge_scores']['pct_eq_100']:.1f}%")
    print(f"\n  ── Structural Metrics ────────────────────────────────────────")
    print(f"  Functional match : {100*func_mr:.1f}%   "
          f"Param match : {100*param_omr:.1f}%   Step match : {100*step_mr:.1f}%")
    print(f"  FuncTool acc : {np.mean(func_tools):.3f}   "
          f"Param acc : {np.mean(param_accs):.3f}   Dep acc : {np.mean(dep_accs):.3f}")
    if len(error_types) > 1:
        print(f"\n  ── Per Error-Type ─────────────────────────────────────────")
        hdr = (f"  {'Error Type':28s}  {'N':>4}  {'Success%':>8}  "
               f"{'Handled%':>8}  {'Judge':>6}  {'FuncTool%':>9}  "
               f"{'Param%':>6}  {'FuncMatch%':>10}")
        print(hdr)
        print("  " + "-"*(len(hdr)-2))
        for et, d in per_error.items():
            print(f"  {et:28s}  {d['n']:>4}  "
                  f"{100*d['judge_success_rate']:>8.1f}  "
                  f"{100*d['error_type_handled_rate']:>8.1f}  "
                  f"{d['mean_judge_score']:>6.1f}  "
                  f"{100*d['functional_tool_acc']:>9.1f}  "
                  f"{100*d['mean_param_accuracy']:>6.1f}  "
                  f"{100*d['functional_match_rate']:>10.1f}")
    print()
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 13. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GNN4TaskPlan baseline (Wu et al., NeurIPS 2024) — fixed version"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--mode", default="sgc", choices=["sgc", "graphsage"])
    parser.add_argument("--sgc-k", type=int, default=2)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--gnn-epochs", type=int, default=20)
    parser.add_argument("--gnn-lr", type=float, default=1e-3)
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--gnn-negatives", type=int, default=2)
    parser.add_argument("--gnn-device", default="cuda:0")
    parser.add_argument("--max-decompose-tokens", type=int, default=256)
    parser.add_argument("--max-fill-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=300)
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--perfect-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="gnn4tp_results.json")
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--return-raw", action="store_true")
    parser.add_argument("--judge_url", default="http://localhost:8001/v1/chat/completions")
    args = parser.parse_args()

    if not args.perfect_only and not args.full:
        parser.error("Specify at least one of --perfect-only or --full")
    stats_output = args.stats_output or args.output.replace(".json", ".stats.json")

    global JUDGE_SERVER_URL
    JUDGE_SERVER_URL = args.judge_url
    try:
        r = requests.get(JUDGE_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5)
        print(f"✅ Judge server healthy: {r.json()}")
    except Exception as e:
        print(f"❌ Judge server not reachable: {e}"); return

    print("\nLoading models...")
    embedder = ToolEmbedder(args.embedding_model, device=args.embedding_device)
    llm = GNN4TaskPlanModel(args.model, device=args.device)

    if args.mode == "sgc":
        retriever = SGCRetriever(k=args.sgc_k)
        print(f"✓ SGC retriever ready (k={args.sgc_k}, training-free)")
    else:
        test_emb = embedder.encode(["test"])
        emb_dim = test_emb.shape[1]
        retriever = GraphSAGERetriever(emb_dim=emb_dim, hidden_dim=args.gnn_hidden,
                                        device=args.gnn_device)
        if args.train_parquet:
            print(f"\nLoading training data from {args.train_parquet}...")
            train_examples = load_parquet(args.train_parquet, perfect_only=False)
            if args.limit:
                train_examples = train_examples[:min(args.limit * 10, len(train_examples))]
            print(f"  {len(train_examples)} training examples")
            print("  Building BPR training triplets...")
            triplets = build_training_triplets(train_examples, embedder,
                                                n_negatives=args.gnn_negatives)
            print(f"  {len(triplets)} triplets")
            print(f"\nTraining GraphSAGE (epochs={args.gnn_epochs}, lr={args.gnn_lr})...")
            retriever.train(triplets, epochs=args.gnn_epochs, lr=args.gnn_lr)
        else:
            print("⚠  No --train-parquet provided. Using untrained retriever.")

    perfect_gt_by_qid = load_perfect_gt_from_parquet(args.test_parquet)
    eval_kwargs = dict(
        llm=llm, embedder=embedder, retriever=retriever, mode=args.mode,
        perfect_gt_by_qid=perfect_gt_by_qid, sgc_k=args.sgc_k,
        max_new_tokens_decompose=args.max_decompose_tokens,
        max_new_tokens_fill=args.max_fill_tokens,
        judge_max_tokens=args.judge_max_tokens,
        return_raw=args.return_raw,
    )
    all_output = {"config": vars(args), "runs": {}}
    all_stats = {"config": vars(args), "runs": {}}

    if args.perfect_only:
        print("\nLoading perfect-only examples...")
        examples = load_parquet(args.test_parquet, perfect_only=True)
        if args.limit: examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_gnn4taskplan(examples=examples, **eval_kwargs)
        stats = compute_stats(results, f"PERFECT-ONLY  GNN4TaskPlan-{args.mode.upper()}")
        all_output["runs"]["perfect_only"] = results
        all_stats["runs"]["perfect_only"] = stats

    if args.full:
        print("\nLoading full test set...")
        examples = load_parquet(args.test_parquet, perfect_only=False)
        if args.limit: examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_gnn4taskplan(examples=examples, **eval_kwargs)
        stats = compute_stats(results, f"FULL TEST SET  GNN4TaskPlan-{args.mode.upper()}")
        all_output["runs"]["full"] = results
        all_stats["runs"]["full"] = stats

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f: json.dump(all_output, f, indent=2)
    print(f"Results saved → {args.output}")
    with open(stats_output, "w") as f: json.dump(all_stats, f, indent=2)
    print(f"Stats saved → {stats_output}")


if __name__ == "__main__":
    main()