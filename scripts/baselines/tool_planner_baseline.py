#!/usr/bin/env python3
"""
Tool-Planner Baseline for ToolHop Tool Planning

Implements the core ideas of Tool-Planner (Liu et al., ICLR 2025) as a
prompting-only baseline for comparison against the SFT+PPO RL planner.

Key components (matching paper Section 3):
  1. Tool Clustering   : Per-query grouping of tools into toolkits using
                         SimCSE/RoBERTa sentence embeddings + k-means++.
                         k = max(1, n_unique_tools // 3) by default.
  2. Toolkit Planning  : LLM generates a Step-N plan with toolkit-level
                         context (aware of which tools are functionally similar).
  3. Offline Revision  : If the initial plan scores <60 via the judge, a
                         single LLM revision pass is triggered — simulating
                         Tool-Planner's cross-toolkit error recovery without a
                         real execution environment (ToolHop is offline-only).

Adaptation notes (ToolHop vs ToolBench):
  - ToolBench has 16K APIs across thousands of queries; ToolHop has 3–8
    tools per query.  Clustering is therefore per-query (not dataset-wide).
  - GT tool names are NL sub-questions (ToolHop artifact).
    exact_tool_accuracy will be ~0; use functional_tool_accuracy instead.
  - "Within-toolkit fallback" cannot be tested without execution;
    we approximate it via the revision pass.

Output format mirrors best_of_n_selection.py / react_baseline.py exactly
for direct side-by-side comparison in papers.

Dependencies:
    pip install sentence-transformers scikit-learn
    (or just transformers + sklearn — sentence-transformers is optional)

Usage:
    # Start judge server first (GPU 7):
    CUDA_VISIBLE_DEVICES=7 python judge_server.py \
        --model ${FORTE_ROOT}/judge_finetuning/models/judge/merged --port 8002

    # Run Tool-Planner baseline:
    python tool_planner_baseline.py \
        --model Qwen/Llama-7B-Instruct \
        --test-parquet data/verl_rl_full_clean/test.parquet \
        --full \
        --output tool_planner_results.json \
        --stats-output tool_planner_stats.json \
        --judge_url http://localhost:8004/v1/chat/completions

    # With explicit SimCSE embedding model (matches paper):
    CUDA_VISIBLE_DEVICES=2 python tool_planner_baseline.py \
        --model ${FORTE_ROOT}/planner_finetuning/checkpoints_planner_sft-llama-3b/global_step_147 \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
        --full \
        --output tool_planner_Llama-3B-results.json \
        --judge_url http://localhost:8001/v1/chat/completions

    # Disable revision pass (ablate error recovery):
    python tool_planner_baseline.py ... --no-revision

    # Fix k=2 toolkits per query regardless of tool count:
    python tool_planner_baseline.py ... --n-clusters 2

KNOWN DATASET ARTIFACT:
    Ground truth tool names are NL sub-questions (e.g. "What is the ...?").
    Trained model and prompting baselines generate abstracted API-style names.
    exact_tool_accuracy will therefore be ~0 on this dataset.
    Report functional_tool_accuracy (keyword Jaccard) and param_accuracy instead.
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

JUDGE_SERVER_URL = "http://localhost:8004/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an expert at creating multi-step tool execution plans. "
    "Given a query and available tools, generate a correct sequence of "
    "tool calls to answer the query."
)

TOOL_PLANNER_SYSTEM_PROMPT = (
    "You are an expert at task planning using clustered tool groups (toolkits). "
    "Given a query and toolkits — groups of functionally similar tools — you "
    "reason about which toolkits are needed, in what order, and pick the best "
    "tool from each toolkit to generate a concrete execution plan."
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
# 1. TOOL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

class ToolClusterer:
    def __init__(self, model_name: str = "princeton-nlp/sup-simcse-roberta-base", device: str = "cpu"):
        print(f"  Loading embedding model: {model_name} on {device}...")
        self._tok = HFTokenizer.from_pretrained(model_name)
        self._enc = AutoModel.from_pretrained(model_name).to(device)
        self._enc.eval()
        self._device = device
        print(f"  ✓ Embedding model ready")

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> np.ndarray:
        enc = self._tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self._device)
        out = self._enc(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
        embs = embs.cpu().float().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
        return embs / norms

    def cluster(self, tool_names: List[str], tool_descs: List[str], n_clusters: Optional[int] = None) -> List[List[int]]:
        n = len(tool_names)
        if n == 0: return []
        k = n_clusters if n_clusters else max(1, n // 3)
        k = min(k, n)
        if k == 1 or n == 1: return [list(range(n))]
        embs = self._embed(tool_descs)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            labels = km.fit_predict(embs)
        except ImportError:
            labels = np.array([i % k for i in range(n)])
        clusters: List[List[int]] = [[] for _ in range(k)]
        for idx, lab in enumerate(labels.tolist()):
            clusters[lab].append(idx)
        return [c for c in clusters if c]


def load_perfect_gt_from_parquet(parquet_path: str) -> dict:
    import json as _json
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    perfect_gt: dict = {}
    for ei, rm in zip(extra_infos, reward_models):
        if isinstance(ei, str): ei = _json.loads(ei)
        if isinstance(rm, str): rm = _json.loads(rm)
        if not isinstance(ei, dict) or not isinstance(rm, dict): continue
        if (str(ei.get("error_type", "")) == "none" and int(ei.get("quality_score", 0)) >= 100):
            qid = ei.get("query_id", -1)
            gt_str = rm.get("ground_truth", "")
            if gt_str and qid not in perfect_gt:
                perfect_gt[qid] = gt_str
    print(f"  Loaded perfect GT for {len(perfect_gt)} query_ids.")
    return perfect_gt


def _tool_desc(tool_name: str, tool_info: Dict) -> str:
    props = tool_info.get("parameters", {}).get("properties", {})
    req = tool_info.get("parameters", {}).get("required", [])
    param_str = ", ".join(f"{k}({v.get('type','any')}{'*' if k in req else ''})" for k, v in props.items())
    return f"{tool_name}: {param_str}"


def _format_toolkit(toolkit: List[Tuple[str, Dict]], idx: int) -> str:
    lines = [f"Toolkit {idx} ({len(toolkit)} tool{'s' if len(toolkit)>1 else ''}):"]
    for name, info in toolkit:
        props = info.get("parameters", {}).get("properties", {})
        req = info.get("parameters", {}).get("required", [])
        parts = [f"{k}: {v.get('type','any')}{' (required)' if k in req else ''}" for k, v in props.items()]
        lines.append(f"  - {name}({', '.join(parts)})")
    return "\n".join(lines)


def _format_tools_flat(tools: Dict[str, Any]) -> str:
    unique: Dict[str, Dict] = {}
    for sub_q, info in tools.items():
        name = info.get("name", sub_q)
        if name not in unique: unique[name] = info
    lines = ["Available Tools:"]
    for name, info in unique.items():
        props = info.get("parameters", {}).get("properties", {})
        req = info.get("parameters", {}).get("required", [])
        parts = [f"{k}: {v.get('type','any')}{' (required)' if k in req else ''}" for k, v in props.items()]
        lines.append(f"- {name}({', '.join(parts)})")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_toolkit_planning_prompt(question: str, toolkits: List[List[Tuple[str, Dict]]]) -> str:
    toolkit_blocks = "\n\n".join(_format_toolkit(tk, i) for i, tk in enumerate(toolkits))
    return (
        f"Your task: create a tool execution plan for the query below.\n\n"
        f"Query: {question}\n\n"
        f"Available Toolkits (each is a group of functionally similar tools):\n\n"
        f"{toolkit_blocks}\n\n"
        f"Instructions:\n"
        f"1. Identify which toolkits are needed and in what order.\n"
        f"2. From each needed toolkit, pick the SINGLE best-fitting tool.\n"
        f"3. Generate a step-by-step plan where each step calls exactly one tool.\n\n"
        f"Format rules:\n"
        f"  - Each step: Step N: {{{{N}}}} = tool_name(param1=value1, ...)\n"
        f"  - Use {{{{N}}}} variables for intermediate results.\n"
        f"  - Reference prior outputs with {{{{N}}}} in parameter values.\n\n"
        f"Plan:"
    )


def build_revision_prompt(question: str, tools: Dict[str, Any], original_plan: str, issue_hint: str) -> str:
    tools_str = _format_tools_flat(tools)
    return (
        f"The plan below has an issue. Please generate a corrected plan.\n\n"
        f"Query: {question}\n\n"
        f"{tools_str}\n\n"
        f"Original plan:\n{original_plan}\n\n"
        f"Detected issue: {issue_hint}\n\n"
        f"Generate a revised, complete plan. Rules:\n"
        f"  - Each step: Step N: {{{{N}}}} = tool_name(param1=value1, ...)\n"
        f"  - Use {{{{N}}}} for intermediate results.\n"
        f"  - Reference prior outputs with {{{{N}}}} in parameters.\n\n"
        f"Revised plan:"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLAN PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_plan_steps(plan_text: str) -> List[Dict]:
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("Step "): continue
        try:
            step_match = re.match(r"Step (\d+):", line)
            if not step_match: continue
            step_id = int(step_match.group(1))
            var_match = re.search(r"(\{\{\d+\}\})\s*=", line)
            if not var_match: continue
            output_var = var_match.group(1)
            tool_match = re.search(r"=\s*([^\(]+)\((.*)\)\s*$", line)
            if not tool_match:
                tool_match_empty = re.search(r"=\s*([^\(]+)\(\)\s*$", line)
                if tool_match_empty:
                    tool_name = tool_match_empty.group(1).strip()
                    params: Dict[str, str] = {}
                else: continue
            else:
                tool_name = tool_match.group(1).strip()
                params_str = tool_match.group(2).strip()
                params = {}
                if params_str:
                    param_parts: List[str] = []
                    current = ""
                    depth = 0
                    in_str = False
                    str_char: Optional[str] = None
                    for ch in params_str:
                        if ch in ('"', "'") and (not in_str or ch == str_char):
                            in_str = not in_str
                            str_char = ch if in_str else None
                        if not in_str:
                            if ch in "([{": depth += 1
                            elif ch in ")]}": depth -= 1
                            elif ch == "," and depth == 0:
                                param_parts.append(current.strip())
                                current = ""
                                continue
                        current += ch
                    if current.strip(): param_parts.append(current.strip())
                    for part in param_parts:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            params[k.strip()] = v.strip()
            steps.append({"step_id": step_id, "output_variable": output_var, "tool_name": tool_name, "parameters": params})
        except Exception: continue
    return steps


def _extract_step_lines(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.split("\n") if re.match(r"\s*Step \d+:", line)]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 4. STRUCTURAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_nl_tool_name(name: str) -> bool:
    return len(name.split()) > 4 or name.endswith("?")


def _functional_tool_match(gen_name: str, gt_name: str) -> float:
    STOP = {"what","is","the","of","in","a","an","and","or","to","how","many",
            "who","which","are","was","were","be","been","at","on","for","with",
            "that","this","it","its","from"}
    def kw(s: str) -> set:
        return {w for w in re.sub(r"[^a-z0-9\s]"," ",s.lower()).split() if w not in STOP and len(w) > 2}
    g, r = kw(gen_name), kw(gt_name)
    if not g or not r: return 0.0
    return round(len(g & r) / len(g | r), 3)


def _norm(v: str) -> str:
    return " ".join(str(v).strip().strip("\"'").lower().split())


def _remap_gt_tool_name(nl_name: str, tools: Dict[str, Any]) -> str:
    if nl_name in tools:
        api_name = tools[nl_name].get("name")
        if api_name: return api_name
    for key, tool_info in tools.items():
        if nl_name in key or key in nl_name:
            api_name = tool_info.get("name")
            if api_name: return api_name
    return nl_name


def evaluate_plan_vs_gt(gen_steps: List[Dict], gt_steps: List[Dict], tools: Dict[str, Any] = None) -> Dict:
    empty = {"valid": False, "error": "", "step_count_match": False,
             "exact_tool_accuracy": 0.0, "functional_tool_accuracy": 0.0,
             "param_accuracy": 0.0, "dependency_accuracy": 0.0,
             "exact_match": False, "functional_match": False, "param_only_match": False,
             "gt_uses_nl_tool_names": False, "step_details": []}
    if not gen_steps: return {**empty, "error": "no steps generated"}
    if not gt_steps: return {**empty, "error": "no ground truth steps"}

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
        gt = gt_steps[i] if i < len(gt_steps) else None
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
                gv = _norm(gt["parameters"][k])
                dv = _norm(gen["parameters"][k])
                if gv == dv or gv in dv or dv in gv: tpc += 1
                else: incorrect.append({"param": k, "generated": gen["parameters"][k], "ground_truth": gt["parameters"][k]})
            tp += len(gt_keys)
            detail["param_comparison"] = {"total_gt_params": len(gt_keys), "correct": len(common) - len(incorrect),
                                          "missing": list(gt_keys - gen_keys), "extra": list(gen_keys - gt_keys), "incorrect": incorrect}
            gt_refs = set(re.findall(r"\{\{\d+\}\}", str(gt["parameters"])))
            gen_refs = set(re.findall(r"\{\{\d+\}\}", str(gen["parameters"])))
            td += len(gt_refs); cd += len(gt_refs & gen_refs)
            detail["dependency_refs_match"] = (gt_refs == gen_refs)
        else:
            detail["exact_tool_correct"] = False; detail["functional_tool_score"] = 0.0
            detail["param_comparison"] = None; detail["dependency_refs_match"] = False
        step_details.append(detail)

    n_gt = len(gt_steps)
    ea = ce / n_gt; fa = tf / n_gt
    pa = tpc / tp if tp > 0 else 0.0; da = cd / td if td > 0 else 1.0
    return {"valid": True, "gt_uses_nl_tool_names": gt_uses_nl, "step_count_match": step_count_match,
            "generated_steps": len(gen_steps), "ground_truth_steps": n_gt,
            "exact_tool_accuracy": ea, "functional_tool_accuracy": fa,
            "param_accuracy": pa, "dependency_accuracy": da,
            "exact_match": step_count_match and ea == 1.0 and pa == 1.0,
            "functional_match": step_count_match and fa >= 0.5 and pa >= 0.5,
            "param_only_match": pa >= 0.5, "step_details": step_details}


# ══════════════════════════════════════════════════════════════════════════════
# 5. JUDGE CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def _format_tools_for_judge(tools: Dict[str, Any]) -> str:
    if not tools: return ""
    lines = ["Available Tools:"]
    seen: Dict[str, Dict] = {}
    for sub_q, info in tools.items():
        name = info.get("name", sub_q)
        if name not in seen: seen[name] = info
    for name, info in seen.items():
        props = info.get("parameters", {}).get("properties", {})
        ps = ", ".join(f"{k}: {v.get('type','any')}" for k, v in props.items())
        lines.append(f"- {name}({ps})")
    return "\n".join(lines)


def score_plan_via_judge(query: str, plan_str: str, tools: Dict, max_tokens: int = 300, retries: int = 3) -> Dict[str, Any]:
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
    payload = {"model": "judge", "messages": [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_content}], "temperature": 0.0, "max_tokens": max_tokens}
    content = ""
    for attempt in range(retries):
        try:
            resp = requests.post(JUDGE_SERVER_URL, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "```json" in content: content = content[content.find("```json") + 7: content.rfind("```")].strip()
            elif "```" in content: content = content[content.find("```") + 3: content.rfind("```")].strip()
            if not content.endswith("}"):
                last = content.rfind("}")
                if last != -1: content = content[:last + 1]
            ann = json.loads(content)
            ann["quality_score"] = max(0, min(100, int(ann.get("quality_score", 50))))
            ann["confidence"] = max(0.0, min(1.0, float(ann.get("confidence", 0.5))))
            ann["_full_parse"] = True
            return ann
        except (requests.ConnectionError, requests.Timeout):
            if attempt < retries - 1: time.sleep(2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            try:
                m = re.search(r'"quality_score"\s*:\s*(\d+)', content)
                if m: return {"quality_score": max(0, min(100, int(m.group(1)))), "success_prediction": "uncertain", "reasoning": "partial parse", "issues": [], "confidence": 0.5, "_full_parse": False}
            except Exception: pass
            break
    return {"quality_score": 0, "success_prediction": "no", "reasoning": "judge call failed", "issues": [], "confidence": 0.0, "_full_parse": False}


# ══════════════════════════════════════════════════════════════════════════════
# 6. TOOL-PLANNER MODEL WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class ToolPlannerModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading planner model from {model_path}...")
        self.tokenizer = HFTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self._gen_device = next(self.model.parameters()).device
        print(f"✓ Planner loaded on {self._gen_device}")

    def _call(self, messages: List[Dict], max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self._gen_device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature if temperature > 0 else 1.0, do_sample=temperature > 0, top_p=0.9 if temperature > 0 else None, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def plan_with_toolkits(self, question: str, toolkits: List[List[Tuple[str, Dict]]], max_new_tokens: int = 512) -> str:
        messages = [{"role": "system", "content": TOOL_PLANNER_SYSTEM_PROMPT}, {"role": "user", "content": build_toolkit_planning_prompt(question, toolkits)}]
        return self._call(messages, max_new_tokens=max_new_tokens)

    def revise(self, question: str, tools: Dict[str, Any], original_plan: str, hint: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "system", "content": TOOL_PLANNER_SYSTEM_PROMPT}, {"role": "user", "content": build_revision_prompt(question, tools, original_plan, hint)}]
        return self._call(messages, max_new_tokens=max_new_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 7. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_test_parquet(parquet_path: str, perfect_only: bool = False) -> List[Dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    extra_infos = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()
    examples: List[Dict] = []
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
        if perfect_only and not (error_type == "none" and quality_score >= 100): continue
        examples.append({"question": dj.get("question", ""), "tools": dj.get("tools", {}), "ground_truth": rm.get("ground_truth", ""), "error_type": error_type, "quality_score": quality_score, "query_id": ei.get("query_id", -1)})
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_tool_planner(
    model: ToolPlannerModel, clusterer: ToolClusterer, examples: List[Dict],
    perfect_gt_by_qid: Dict[int, str],
    n_clusters: Optional[int] = None, max_new_tokens: int = 512,
    judge_max_tokens: int = 300, revision_threshold: int = 60,
    do_revision: bool = True, return_raw: bool = False,
) -> List[Dict]:
    results = []
    empty_plan_count = 0

    for ex in tqdm(examples, desc="Tool-Planner evaluation"):
        question = ex["question"]; tools = ex["tools"]; ground_truth = ex["ground_truth"]
        gt_steps = parse_plan_steps(ground_truth)

        unique_tools: Dict[str, Dict] = {}
        for sub_q, info in tools.items():
            name = info.get("name", sub_q)
            if name not in unique_tools: unique_tools[name] = info
        tool_names = list(unique_tools.keys())
        tool_descs = [_tool_desc(n, unique_tools[n]) for n in tool_names]

        k = n_clusters if n_clusters else max(1, len(tool_names) // 3)
        cluster_idxs = clusterer.cluster(tool_names, tool_descs, n_clusters=k)
        toolkits: List[List[Tuple[str, Dict]]] = [[(tool_names[i], unique_tools[tool_names[i]]) for i in ci] for ci in cluster_idxs]

        raw_phase1 = model.plan_with_toolkits(question, toolkits, max_new_tokens=max_new_tokens)
        initial_plan = _extract_step_lines(raw_phase1)

        revision_triggered = False
        initial_judge_score: Optional[int] = None
        final_plan = initial_plan

        if do_revision and initial_plan.strip():
            fast_ann = score_plan_via_judge(question, initial_plan or "(empty plan)", tools, max_tokens=64)
            initial_judge_score = fast_ann["quality_score"]
            if initial_judge_score < revision_threshold:
                revision_triggered = True
                hint = f"Quality score {initial_judge_score}/100. Issues may include: wrong tool selection, missing parameters, or incorrect dependency references."
                raw_revised = model.revise(question, tools, initial_plan, hint, max_new_tokens=max_new_tokens)
                revised_plan = _extract_step_lines(raw_revised)
                if revised_plan.strip(): final_plan = revised_plan

        if not final_plan.strip(): empty_plan_count += 1

        judge_ann = score_plan_via_judge(question, final_plan or "(empty plan)", tools, max_tokens=judge_max_tokens)

        gen_steps = parse_plan_steps(final_plan)
        if (ex["error_type"] in ARTIFACT_ERROR_TYPES and ex["query_id"] in perfect_gt_by_qid):
            _struct_gt_steps = parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        else:
            _struct_gt_steps = gt_steps
        struct_eval = evaluate_plan_vs_gt(gen_steps, _struct_gt_steps, tools=tools)

        judge_success = judge_ann["quality_score"] >= 80
        ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)
        error_type_handled = judge_success if ref_is_perfect else judge_ann["quality_score"] >= ex["quality_score"]

        result: Dict[str, Any] = {
            "query_id": ex["query_id"], "question": question, "error_type": ex["error_type"],
            "ref_quality_score": ex["quality_score"], "ref_is_perfect": ref_is_perfect,
            "ground_truth": ground_truth, "generated_plan": final_plan, "initial_plan": initial_plan,
            "n_extracted_steps": len(gen_steps), "n_tools": len(tool_names), "n_toolkits": len(toolkits),
            "toolkit_sizes": [len(tk) for tk in toolkits],
            "judge_success": judge_success, "judge_score": judge_ann["quality_score"],
            "judge_success_pred": judge_ann["success_prediction"], "judge_confidence": judge_ann["confidence"],
            "judge_full_parse": judge_ann.get("_full_parse", False),
            "initial_judge_score": initial_judge_score, "revision_triggered": revision_triggered,
            "gt_uses_nl_tool_names": struct_eval["gt_uses_nl_tool_names"],
            "exact_match": struct_eval["exact_match"], "functional_match": struct_eval["functional_match"],
            "param_only_match": struct_eval["param_only_match"], "step_count_match": struct_eval["step_count_match"],
            "generated_n_steps": struct_eval.get("generated_steps", 0),
            "gt_n_steps": struct_eval.get("ground_truth_steps", len(gt_steps)),
            "exact_tool_accuracy": struct_eval["exact_tool_accuracy"],
            "functional_tool_accuracy": struct_eval["functional_tool_accuracy"],
            "param_accuracy": struct_eval["param_accuracy"], "dependency_accuracy": struct_eval["dependency_accuracy"],
            "error_type_handled": error_type_handled, "judge_agrees_with_ref": (ref_is_perfect == judge_success),
            "method": "Tool-Planner", "react_mode": None,
        }
        if return_raw: result["raw_phase1_output"] = raw_phase1
        results.append(result)

    if empty_plan_count > 0:
        print(f"\n  ⚠  {empty_plan_count}/{len(results)} examples produced no parseable plan steps.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 9. STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(results: List[Dict], label: str) -> Dict:
    n = len(results)
    if n == 0: return {"label": label, "n": 0}

    scores = [r["judge_score"] for r in results]
    func_tools = [r["functional_tool_accuracy"] for r in results]
    param_accs = [r["param_accuracy"] for r in results]
    dep_accs = [r["dependency_accuracy"] for r in results]

    judge_sr = float(np.mean([r["judge_success"] for r in results]))
    err_hr = float(np.mean([r["error_type_handled"] for r in results]))
    exact_mr = float(np.mean([r["exact_match"] for r in results]))
    func_mr = float(np.mean([r["functional_match"] for r in results]))
    param_omr = float(np.mean([r["param_only_match"] for r in results]))
    step_mr = float(np.mean([r["step_count_match"] for r in results]))
    fp_rate = float(np.mean([r.get("judge_full_parse", False) for r in results]))
    empty_rate = float(np.mean([r["generated_n_steps"] == 0 for r in results]))
    revision_rate = float(np.mean([r.get("revision_triggered", False) for r in results]))
    mean_n_toolkits = float(np.mean([r["n_toolkits"] for r in results]))
    mean_n_tools = float(np.mean([r["n_tools"] for r in results]))

    error_types = sorted(set(r["error_type"] for r in results))
    per_error: Dict[str, Dict] = {}
    for et in error_types:
        sub = [r for r in results if r["error_type"] == et]
        per_error[et] = {
            "n": len(sub),
            "judge_success_rate": float(np.mean([r["judge_success"] for r in sub])),
            "error_type_handled_rate": float(np.mean([r["error_type_handled"] for r in sub])),
            "mean_judge_score": float(np.mean([r["judge_score"] for r in sub])),
            "functional_tool_acc": float(np.mean([r["functional_tool_accuracy"] for r in sub])),
            "mean_param_accuracy": float(np.mean([r["param_accuracy"] for r in sub])),
            "exact_match_rate": float(np.mean([r["exact_match"] for r in sub])),
            "functional_match_rate": float(np.mean([r["functional_match"] for r in sub])),
            "param_only_match_rate": float(np.mean([r["param_only_match"] for r in sub])),
            "step_count_match_rate": float(np.mean([r["step_count_match"] for r in sub])),
        }

    success_dist: Dict[str, Dict] = {}
    for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
        c = sum(r["judge_success_pred"] == pred for r in results)
        success_dist[pred] = {"count": c, "pct": round(100 * c / n, 1)}

    stats: Dict[str, Any] = {
        "label": label, "method": "Tool-Planner (offline)", "n_examples": n,
        "gt_uses_nl_tools": bool(results[0].get("gt_uses_nl_tool_names", False)),
        "judge_full_parse_rate": round(fp_rate, 3), "empty_plan_rate": round(empty_rate, 3),
        "revision_rate": round(revision_rate, 3),
        "toolkit_stats": {"mean_tools_per_query": round(mean_n_tools, 2), "mean_toolkits_per_query": round(mean_n_toolkits, 2)},
        "accuracy": {"judge_success_rate": round(judge_sr, 3), "error_handled_rate": round(err_hr, 3)},
        "judge_scores": {"mean": round(float(np.mean(scores)), 2), "median": round(float(np.median(scores)), 2), "std": round(float(np.std(scores)), 2), "pct_gte_80": round(100 * sum(s >= 80 for s in scores) / n, 1), "pct_eq_100": round(100 * sum(s == 100 for s in scores) / n, 1)},
        "structural": {
            "exact_match_rate": round(exact_mr, 3), "functional_match_rate": round(func_mr, 3),
            "param_only_match_rate": round(param_omr, 3), "step_count_match_rate": round(step_mr, 3),
            "mean_functional_tool_acc": round(float(np.mean(func_tools)), 3),
            "mean_param_accuracy": round(float(np.mean(param_accs)), 3),
            "mean_dependency_accuracy": round(float(np.mean(dep_accs)), 3),
        },
        "success_prediction_dist": success_dist, "per_error_type": per_error,
    }

    W = 70
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  Method : Tool-Planner (offline, Liu et al. ICLR 2025)")
    print(f"  N      : {n}  |  avg {mean_n_tools:.1f} tools → {mean_n_toolkits:.1f} toolkits / query")
    if stats["gt_uses_nl_tools"]: print(f"  ⚠  GT uses NL tool names — exact_tool_accuracy ~0")
    if empty_rate > 0.05: print(f"  ⚠  Empty plan rate: {100*empty_rate:.1f}%")
    if fp_rate < 0.9: print(f"  ⚠  Judge full-parse rate: {100*fp_rate:.0f}%")
    print(f"  Revision triggered: {100*revision_rate:.1f}%")
    print(f"\n  ── Primary Accuracy ──────────────────────────────────────────")
    print(f"  Judge success (score≥80) : {100*judge_sr:.1f}%")
    print(f"  Error type handled       : {100*err_hr:.1f}%")
    print(f"\n  ── Judge Scores ──────────────────────────────────────────────")
    print(f"  Mean / Median / Std : {stats['judge_scores']['mean']:.1f} / {stats['judge_scores']['median']:.1f} / {stats['judge_scores']['std']:.1f}")
    print(f"  ≥80: {stats['judge_scores']['pct_gte_80']:.1f}%  |  =100: {stats['judge_scores']['pct_eq_100']:.1f}%")
    print(f"\n  ── Structural Metrics ─────────────────────────────────────────")
    print(f"  Exact match: {100*exact_mr:.1f}%  Functional match: {100*func_mr:.1f}%  Param-only: {100*param_omr:.1f}%")
    print(f"  Step count match: {100*step_mr:.1f}%  Func tool acc: {np.mean(func_tools):.3f}  Param acc: {np.mean(param_accs):.3f}")
    if len(error_types) > 1:
        print(f"\n  ── Per Error-Type ─────────────────────────────────────────────")
        for et, d in per_error.items():
            print(f"  {et:28s}  N={d['n']:>4}  Success={100*d['judge_success_rate']:.1f}%  Judge={d['mean_judge_score']:.1f}  FuncMatch={100*d['functional_match_rate']:.1f}%")
    print()
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-Planner baseline (offline) for ToolHop")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-model", default="princeton-nlp/sup-simcse-roberta-base")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=300)
    parser.add_argument("--revision-threshold", type=int, default=60)
    parser.add_argument("--no-revision", action="store_true")
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--perfect-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="tool_planner_results.json")
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
    clusterer = ToolClusterer(args.embedding_model, device=args.embedding_device)
    model = ToolPlannerModel(args.model, device=args.device)

    # Load perfect GT lookup
    perfect_gt_by_qid = load_perfect_gt_from_parquet(args.test_parquet)

    all_output: Dict[str, Any] = {"config": vars(args), "runs": {}}
    all_stats: Dict[str, Any] = {"config": vars(args), "runs": {}}

    run_kwargs = dict(n_clusters=args.n_clusters, max_new_tokens=args.max_new_tokens,
                      judge_max_tokens=args.judge_max_tokens, revision_threshold=args.revision_threshold,
                      do_revision=not args.no_revision, return_raw=args.return_raw,
                      perfect_gt_by_qid=perfect_gt_by_qid)

    if args.perfect_only:
        print("\nLoading perfect-only test examples...")
        examples = load_test_parquet(args.test_parquet, perfect_only=True)
        if args.limit: examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_tool_planner(model, clusterer, examples, **run_kwargs)
        stats = compute_stats(results, "PERFECT-ONLY  Tool-Planner")
        all_output["runs"]["perfect_only"] = results; all_stats["runs"]["perfect_only"] = stats

    if args.full:
        print("\nLoading full test set...")
        examples = load_test_parquet(args.test_parquet, perfect_only=False)
        if args.limit: examples = examples[:args.limit]
        print(f"  {len(examples)} examples")
        results = evaluate_tool_planner(model, clusterer, examples, **run_kwargs)
        stats = compute_stats(results, "FULL TEST SET  Tool-Planner")
        all_output["runs"]["full"] = results; all_stats["runs"]["full"] = stats

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w") as f: json.dump(all_output, f, indent=2)
    print(f"Results saved → {args.output}")
    with open(stats_output, "w") as f: json.dump(all_stats, f, indent=2)
    print(f"Stats   saved → {stats_output}")


if __name__ == "__main__":
    main()