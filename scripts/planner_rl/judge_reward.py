"""
Key speed fix: persistent requests.Session with connection keep-alive.
Each compute_score call currently opens a new TCP connection to Flask —
~50-100ms overhead per call on top of inference time.
A single persistent session reuses the connection, dropping overhead to ~1ms.

Combined with the Flask server's dynamic batching, this is the fastest
achievable without changing verl internals.
"""

# import json
# import re
# import time
# import threading
# import requests
# from typing import Dict, Any
# from tqdm import tqdm

# # ── Server config ─────────────────────────────────────────────────────────────
# JUDGE_SERVER_URL = "http://localhost:8001/v1/chat/completions"
# JUDGE_MODEL_NAME = "judge"
# REQUEST_TIMEOUT  = 120
# MAX_RETRIES      = 3
# RETRY_DELAY      = 2.0

# # ── Persistent session — one per thread, reuses TCP connection ────────────────
# _session_local = threading.local()

# def _get_session() -> requests.Session:
#     """Return a thread-local persistent session with keep-alive."""
#     if not hasattr(_session_local, "session"):
#         s = requests.Session()
#         # Keep up to 8 connections alive to the judge server
#         adapter = requests.adapters.HTTPAdapter(
#             pool_connections=8,
#             pool_maxsize=8,
#             max_retries=0,
#         )
#         s.mount("http://", adapter)
#         s.headers.update({"Connection": "keep-alive"})
#         _session_local.session = s
#     return _session_local.session

# # ── Batch progress tracking ───────────────────────────────────────────────────
# _progress_bar   = None
# _call_count     = 0
# _last_call_time = 0.0
# BATCH_RESET_SECS = 120.0
# _bar_lock = threading.Lock()

# def _get_progress_bar():
#     global _progress_bar, _call_count, _last_call_time
#     now = time.time()
#     if _progress_bar is None or (now - _last_call_time) > BATCH_RESET_SECS:
#         if _progress_bar is not None:
#             _progress_bar.close()
#         _call_count = 0
#         _progress_bar = tqdm(
#             desc="  ⚖  Scoring responses",
#             unit=" responses",
#             dynamic_ncols=True,
#             colour="cyan",
#             leave=True,
#         )
#     _last_call_time = now
#     return _progress_bar

# # ── Judge prompt ──────────────────────────────────────────────────────────────
# JUDGE_SYSTEM_PROMPT = """You are an expert judge for evaluating tool execution plans. Your task is to:
# 1. Analyze the plan's correctness and efficiency
# 2. Assign a quality score (0-100)
# 3. Predict success likelihood (yes/likely_yes/uncertain/likely_no/no)
# 4. Identify specific issues with severity levels
# 5. Provide detailed reasoning

# Scoring guidelines:
# - 100: Perfect execution, no errors
# - 80-99: Minor issues, likely to succeed
# - 60-79: Moderate issues, uncertain outcome
# - 40-59: Major issues, likely to fail
# - 0-39: Critical errors, will fail"""

# def _format_tools(tools: Dict[str, Any]) -> str:
#     if not tools:
#         return ""
#     lines = ["Available Tools:"]
#     unique_tools = {}
#     for sub_q, tool_info in tools.items():
#         name = tool_info.get("name", sub_q)
#         if name not in unique_tools:
#             unique_tools[name] = tool_info
#     for tool_name, tool_info in unique_tools.items():
#         props = tool_info.get("parameters", {}).get("properties", {})
#         params_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
#         lines.append(f"- {tool_name}({params_str})")
#     return "\n".join(lines)

# def _build_judge_prompt(query: str, plan_str: str, tools: Dict) -> str:
#     tools_str = _format_tools(tools)
#     return f"""Query: {query}

# {tools_str}

# Plan to Evaluate:
# {plan_str}

# Please evaluate this plan and provide:
# 1. Quality score (0-100)
# 2. Success prediction (yes/likely_yes/uncertain/likely_no/no)
# 3. Detailed reasoning
# 4. List of issues (if any)
# 5. Confidence (0.0-1.0)

# Format your response as JSON:
# {{
#   "quality_score": <int>,
#   "success_prediction": "<string>",
#   "reasoning": "<string>",
#   "issues": [...],
#   "confidence": <float>
# }}"""

# def _call_judge_server(query: str, plan_str: str, tools: Dict) -> Dict[str, Any]:
#     """HTTP call using persistent session (keep-alive, no reconnect overhead)."""
#     payload = {
#         "model": JUDGE_MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
#             {"role": "user",   "content": _build_judge_prompt(query, plan_str, tools)},
#         ],
#         "temperature": 0.0,
#         "max_tokens": 32,
#     }
#     session = _get_session()
#     last_error = None
#     for attempt in range(MAX_RETRIES):
#         try:
#             resp = session.post(JUDGE_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
#             resp.raise_for_status()
#             content = resp.json()["choices"][0]["message"]["content"].strip()

#             if "```json" in content:
#                 content = content[content.find("```json") + 7 : content.rfind("```")].strip()
#             elif "```" in content:
#                 content = content[content.find("```") + 3 : content.rfind("```")].strip()
#             if not content.endswith("}"):
#                 last_brace = content.rfind("}")
#                 if last_brace != -1:
#                     content = content[:last_brace + 1]

#             annotation = json.loads(content)
#             annotation["quality_score"] = max(0, min(100, int(annotation.get("quality_score", 50))))
#             annotation["confidence"]    = max(0.0, min(1.0, float(annotation.get("confidence", 0.5))))
#             return annotation

#         except (requests.ConnectionError, requests.Timeout) as e:
#             last_error = e
#             # Recreate session on connection error
#             _session_local.session = None
#             if attempt < MAX_RETRIES - 1:
#                 time.sleep(RETRY_DELAY)
#         except (json.JSONDecodeError, KeyError, ValueError):
#             match = re.search(r'"quality_score"\s*:\s*(\d+)', content)
#             if match:
#                 return {"quality_score": max(0, min(100, int(match.group(1)))),
#                         "success_prediction": "uncertain", "reasoning": "partial parse",
#                         "issues": [], "confidence": 0.5}
#             break

#     return {"quality_score": 0, "success_prediction": "uncertain",
#             "reasoning": "judge call failed", "issues": [], "confidence": 0.0}

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def _parse_extra_info(extra_info) -> Dict:
#     if extra_info is None: return {}
#     if isinstance(extra_info, str):
#         try: return json.loads(extra_info)
#         except: return {}
#     if isinstance(extra_info, dict): return extra_info
#     return {}

# def _extract_query_and_tools(ei: Dict):
#     data_json = ei.get("data_json")
#     if data_json:
#         try:
#             data = json.loads(data_json)
#             query = data.get("question", data.get("query", ""))
#             tools = data.get("tools", {})
#             if query: return query, tools
#         except: pass
#     return ei.get("question", ei.get("query", "")), ei.get("tools", {})

# def _parse_ground_truth(ground_truth) -> str:
#     if isinstance(ground_truth, dict):
#         return ground_truth.get("ground_truth", str(ground_truth))
#     if isinstance(ground_truth, str):
#         try:
#             obj = json.loads(ground_truth)
#             if isinstance(obj, dict) and "ground_truth" in obj:
#                 return obj["ground_truth"]
#         except: pass
#         return ground_truth
#     return str(ground_truth)

# def _count_plan_steps(plan_text: str) -> int:
#     return sum(1 for line in plan_text.split("\n")
#                if re.match(r"Step \d+:", line.strip()) and "=" in line)

# # ── verl entry points ─────────────────────────────────────────────────────────

# def compute_score(data_source, solution_str, ground_truth, extra_info=None):
#     ei = _parse_extra_info(extra_info)
#     gt = _parse_ground_truth(ground_truth)
#     query, tools = _extract_query_and_tools(ei)
#     if not query: query = gt
#     if not solution_str or not solution_str.strip(): return 0.0

#     annotation = _call_judge_server(query, solution_str.strip(), tools)
#     reward = annotation["quality_score"] / 100.0

#     with _bar_lock:
#         bar = _get_progress_bar()
#         global _call_count
#         _call_count += 1
#         bar.update(1)
#         bar.set_postfix({"score": f"{reward:.2f}", "n": _call_count}, refresh=True)
#     return reward


# def compute_score_v2(data_source, solution_str, ground_truth, extra_info=None):
#     """
#     V2: judge score + step-count penalty + wrong_tool penalty.
#     Uses persistent HTTP session for ~5-10x lower per-call latency.
#     """
#     ei = _parse_extra_info(extra_info)
#     gt = _parse_ground_truth(ground_truth)
#     query, tools = _extract_query_and_tools(ei)
#     if not query: query = gt
#     if not solution_str or not solution_str.strip(): return 0.0

#     annotation = _call_judge_server(query, solution_str.strip(), tools)
#     base = annotation["quality_score"] / 100.0

#     n_gen = _count_plan_steps(solution_str)
#     n_gt  = _count_plan_steps(gt) if gt else 0
#     if n_gt > 0:
#         delta = abs(n_gen - n_gt)
#         step_bonus = 0.0 if delta <= 1 else max(-0.15, -0.05 * (delta - 1))
#     else:
#         step_bonus = 0.0

#     error_type = ei.get("error_type", "none")
#     wrong_tool_penalty = -0.10 if (error_type == "wrong_tool" and base < 0.8) else 0.0

#     reward = max(0.0, min(1.0, base + step_bonus + wrong_tool_penalty))

#     with _bar_lock:
#         bar = _get_progress_bar()
#         global _call_count
#         _call_count += 1
#         bar.update(1)
#         bar.set_postfix({
#             "score":  f"{base:.2f}",
#             "step_d": f"{step_bonus:+.2f}",
#             "wt_pen": f"{wrong_tool_penalty:+.2f}",
#             "reward": f"{reward:.2f}",
#             "n":      _call_count,
#         }, refresh=True)
#     return reward


# # ── Batch entry point for verl BatchRewardManager ────────────────────────────
# # Called once per step with ALL responses simultaneously.
# # Uses ThreadPoolExecutor to fire all judge HTTP calls in parallel —
# # the Flask server's dynamic batcher sees them arrive together and
# # processes them in a single GPU forward pass.
# #
# # To use: add to PPO script:
# #   reward_manager=batch
# #   custom_reward_function.name=compute_score_v2_batched

# from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed

# _batch_executor = _TPE(max_workers=256)


# def compute_score_v2_batched(
#     data_sources,
#     solution_strs,
#     ground_truths,
#     extra_infos,
# ) -> list:
#     """
#     Batch reward function for verl's BatchRewardManager.

#     Receives all responses for a step at once and scores them in parallel.
#     Set in PPO script:
#         reward_manager=batch
#         custom_reward_function.name=compute_score_v2_batched
#     """
#     def _score_one(args):
#         idx, solution_str, ground_truth, extra_info = args
#         ei    = _parse_extra_info(extra_info)
#         gt    = _parse_ground_truth(ground_truth)
#         query, tools = _extract_query_and_tools(ei)
#         if not query:
#             query = gt
#         if not solution_str or not solution_str.strip():
#             return idx, 0.0

#         annotation = _call_judge_server(query, solution_str.strip(), tools)
#         base = annotation["quality_score"] / 100.0

#         n_gen = _count_plan_steps(solution_str)
#         n_gt  = _count_plan_steps(gt) if gt else 0
#         if n_gt > 0:
#             delta = abs(n_gen - n_gt)
#             step_bonus = 0.0 if delta <= 1 else max(-0.15, -0.05 * (delta - 1))
#         else:
#             step_bonus = 0.0

#         error_type = ei.get("error_type", "none")
#         wrong_tool_penalty = -0.10 if (error_type == "wrong_tool" and base < 0.8) else 0.0

#         reward = max(0.0, min(1.0, base + step_bonus + wrong_tool_penalty))
#         return idx, reward

#     # Submit all calls simultaneously — Flask server batches them together
#     args_list = list(enumerate(zip(solution_strs, ground_truths, extra_infos)))
#     args_flat  = [(i, s, g, e) for i, (s, g, e) in args_list]

#     results = [None] * len(args_flat)
#     futures = {_batch_executor.submit(_score_one, a): a[0] for a in args_flat}

#     completed = 0
#     for future in _as_completed(futures):
#         idx, reward = future.result()
#         results[idx] = reward
#         completed += 1

#     # Progress bar
#     with _bar_lock:
#         bar = _get_progress_bar()
#         global _call_count
#         _call_count += len(results)
#         bar.update(len(results))
#         scores = [r for r in results if r is not None]
#         bar.set_postfix({
#             "batch":    len(results),
#             "mean":     f"{sum(scores)/len(scores):.2f}" if scores else "0.00",
#             "n_total":  _call_count,
#         }, refresh=True)

#     return results


# # ── Validation-compatible wrapper ─────────────────────────────────────────────
# # verl's val_reward_fn uses the naive manager regardless of +reward_manager=batch.
# # It calls the function with single-item kwargs: (data_source, solution_str,
# # ground_truth, extra_info). We need compute_score_v2_batched to also handle
# # this call signature gracefully for validation.

# _original_batched = compute_score_v2_batched

# def compute_score_v2_batched(
#     data_sources=None,
#     solution_strs=None,
#     ground_truths=None,
#     extra_infos=None,
#     # single-item kwargs from naive manager / val_reward_fn
#     data_source=None,
#     solution_str=None,
#     ground_truth=None,
#     extra_info=None,
#     **kwargs,
# ):
#     """
#     Dual-mode reward function:
#     - Batch mode (train): called by BatchRewardManager with lists
#     - Single mode (val):  called by naive manager with scalar kwargs
#     """
#     print(f"[judge_reward] BATCH CALL: {len(solution_strs) if solution_strs else 'SINGLE'} items")
#     # Single-item call from val_reward_fn
#     if solution_str is not None:
#         return compute_score_v2(
#             data_source=data_source or "",
#             solution_str=solution_str,
#             ground_truth=ground_truth,
#             extra_info=extra_info,
#         )
#     # Batch call from BatchRewardManager
#     return _original_batched(
#         data_sources=data_sources,
#         solution_strs=solution_strs,
#         ground_truths=ground_truths,
#         extra_infos=extra_infos,
#     )
"""
Judge Reward Function — async version for verl PPO.

verl auto-detects async reward functions and runs them concurrently via
asyncio.gather, which is the correct way to parallelize HTTP-based rewards.
This gives true parallelism: all 224 requests fire simultaneously, the Flask
judge server batches them into ceil(224/64)=4 GPU forward passes.

Setup:
    pip install aiohttp
    CUDA_VISIBLE_DEVICES=7 python3 judge_server.py \
        --model /path/to/judge/merged --port 8001 \
        --batch-size 64 --batch-timeout-ms 200

PPO script:
    custom_reward_function.name=compute_score_v2
    (no +reward_manager=batch needed — async is handled automatically)
"""

import json
import re
import time
import threading
import asyncio
import aiohttp
from typing import Dict, Any
from tqdm import tqdm

# ── Server config ─────────────────────────────────────────────────────────────
JUDGE_SERVER_URL = "http://localhost:8002/v1/chat/completions"
JUDGE_MODEL_NAME = "judge"
REQUEST_TIMEOUT  = 120
MAX_RETRIES      = 3

# ── Progress bar ──────────────────────────────────────────────────────────────
_progress_bar   = None
_call_count     = 0
_last_call_time = 0.0
BATCH_RESET_SECS = 120.0
_bar_lock = threading.Lock()

def _get_progress_bar():
    global _progress_bar, _call_count, _last_call_time
    now = time.time()
    if _progress_bar is None or (now - _last_call_time) > BATCH_RESET_SECS:
        if _progress_bar is not None:
            _progress_bar.close()
        _call_count = 0
        _progress_bar = tqdm(
            desc="  ⚖  Scoring responses",
            unit=" responses",
            dynamic_ncols=True,
            colour="cyan",
            leave=True,
        )
    _last_call_time = now
    return _progress_bar

# ── Judge prompt ──────────────────────────────────────────────────────────────
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

def _format_tools(tools: Dict[str, Any]) -> str:
    if not tools:
        return ""
    lines = ["Available Tools:"]
    unique_tools = {}
    for sub_q, tool_info in tools.items():
        name = tool_info.get("name", sub_q)
        if name not in unique_tools:
            unique_tools[name] = tool_info
    for tool_name, tool_info in unique_tools.items():
        props = tool_info.get("parameters", {}).get("properties", {})
        params_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {tool_name}({params_str})")
    return "\n".join(lines)

def _build_judge_prompt(query: str, plan_str: str, tools: Dict) -> str:
    tools_str = _format_tools(tools)
    return f"""Query: {query}

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

# ── Async judge call ──────────────────────────────────────────────────────────

async def _async_call_judge(session: aiohttp.ClientSession,
                             query: str, plan_str: str, tools: Dict) -> Dict[str, Any]:
    """Single async HTTP call. All calls in a batch run concurrently."""
    payload = {
        "model": JUDGE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": _build_judge_prompt(query, plan_str, tools)},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }

    content = ""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(JUDGE_SERVER_URL, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                resp.raise_for_status()
                # Use text() + manual parse to avoid aiohttp content-type strictness
                raw = await resp.text()
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"].strip()

                if "```json" in content:
                    content = content[content.find("```json") + 7 : content.rfind("```")].strip()
                elif "```" in content:
                    content = content[content.find("```") + 3 : content.rfind("```")].strip()
                if not content.endswith("}"):
                    last = content.rfind("}")
                    if last != -1:
                        content = content[:last + 1]

                # Regex-first: works even when JSON is truncated at max_tokens=32
                match = re.search(r'"quality_score"\s*:\s*(\d+)', content)
                if match:
                    return {
                        "quality_score": max(0, min(100, int(match.group(1)))),
                        "success_prediction": "uncertain",
                        "issues": [],
                        "confidence": 0.5,
                    }
                raise ValueError(f"quality_score not found in: {content[:100]}")

        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2.0)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2.0)

    return {"quality_score": 0, "success_prediction": "uncertain",
            "issues": [], "confidence": 0.0}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_extra_info(extra_info) -> Dict:
    if extra_info is None: return {}
    if isinstance(extra_info, str):
        try: return json.loads(extra_info)
        except: return {}
    if isinstance(extra_info, dict): return extra_info
    return {}

def _extract_query_and_tools(ei: Dict):
    data_json = ei.get("data_json")
    if data_json:
        try:
            data = json.loads(data_json)
            query = data.get("question", data.get("query", ""))
            tools = data.get("tools", {})
            if query: return query, tools
        except: pass
    return ei.get("question", ei.get("query", "")), ei.get("tools", {})

def _parse_ground_truth(ground_truth) -> str:
    if isinstance(ground_truth, dict):
        return ground_truth.get("ground_truth", str(ground_truth))
    if isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
            if isinstance(obj, dict) and "ground_truth" in obj:
                return obj["ground_truth"]
        except: pass
        return ground_truth
    return str(ground_truth)

def _count_plan_steps(plan_text: str) -> int:
    return sum(1 for line in plan_text.split("\n")
               if re.match(r"Step \d+:", line.strip()) and "=" in line)

# ── verl entry points ─────────────────────────────────────────────────────────

async def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Async base reward. verl detects async and runs concurrently."""
    ei = _parse_extra_info(extra_info)
    gt = _parse_ground_truth(ground_truth)
    query, tools = _extract_query_and_tools(ei)
    if not query: query = gt
    if not solution_str or not solution_str.strip(): return 0.0

    async with aiohttp.ClientSession() as session:
        annotation = await _async_call_judge(session, query, solution_str.strip(), tools)
    reward = annotation["quality_score"] / 100.0

    with _bar_lock:
        bar = _get_progress_bar()
        global _call_count
        _call_count += 1
        bar.update(1)
        bar.set_postfix({"score": f"{reward:.2f}", "n": _call_count}, refresh=True)
    return reward


async def compute_score_v2(data_source, solution_str, ground_truth, extra_info=None):
    """
    Async V2 reward: judge score + step-count penalty + wrong_tool penalty.

    verl auto-detects this as async and runs all batch calls concurrently via
    asyncio.gather — no BatchRewardManager or +reward_manager=batch needed.
    All 224 requests fire simultaneously; Flask batches them into 4 GPU passes.
    """
    ei = _parse_extra_info(extra_info)
    gt = _parse_ground_truth(ground_truth)
    query, tools = _extract_query_and_tools(ei)
    if not query: query = gt
    if not solution_str or not solution_str.strip(): return 0.0

    async with aiohttp.ClientSession() as session:
        annotation = await _async_call_judge(session, query, solution_str.strip(), tools)

    base = annotation["quality_score"] / 100.0

    n_gen = _count_plan_steps(solution_str)
    n_gt  = _count_plan_steps(gt) if gt else 0
    if n_gt > 0:
        delta = abs(n_gen - n_gt)
        step_bonus = 0.0 if delta <= 1 else max(-0.15, -0.05 * (delta - 1))
    else:
        step_bonus = 0.0

    error_type = ei.get("error_type", "none")
    wrong_tool_penalty = -0.10 if (error_type == "wrong_tool" and base < 0.8) else 0.0

    reward = max(0.0, min(1.0, base + step_bonus + wrong_tool_penalty))

    with _bar_lock:
        bar = _get_progress_bar()
        global _call_count
        _call_count += 1
        bar.update(1)
        bar.set_postfix({
            "score":  f"{base:.2f}",
            "step_d": f"{step_bonus:+.2f}",
            "wt_pen": f"{wrong_tool_penalty:+.2f}",
            "reward": f"{reward:.2f}",
            "n":      _call_count,
        }, refresh=True)
    return reward