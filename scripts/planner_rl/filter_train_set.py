"""
Filter the GRPO training parquet to remove saturated prompts.

Why this script exists:
    Your GRPO run shows val reward stuck at 0.96-0.97 for 144 steps with collapsing
    entropy. Root cause: the SFT model already nails ~96% of training prompts at
    temperature=1.0, so within a group of 8 rollouts most of them score 0.95-1.0.
    Group-normalized advantages collapse to ≈ 0 → no gradient signal → no learning.

    This filter removes those saturated prompts. By keeping only prompts where the
    SFT model's mean rollout reward is in [0.5, 0.9] (the "Goldilocks zone"), every
    surviving prompt has both:
      - room to improve (mean < 1.0)
      - genuine signal (model isn't completely failing)
    AND, crucially, within-group reward variance — which is what GRPO actually
    needs to learn.

What it does:
    1. Load train.parquet
    2. For each prompt, generate N rollouts with the SFT planner (vLLM, batched)
       — vLLM's `n=N` shares prefix KV cache across rollouts, so this is much
       cheaper than naive batching
    3. Score every rollout via your existing async compute_score_v2 (so scores are
       *identical* to what GRPO sees during training — same code path, same judge
       server, same prompt template, same step-count / wrong-tool penalties)
    4. Filter by per-prompt mean reward
    5. Save filtered parquet ready to drop into your bash script

Architecture notes:
    - compute_score_v2 is `async def`. We drive it with asyncio.gather so all
      scoring calls fire concurrently. Same parallelism trick verl uses during
      training, so judge throughput should match what you see in GRPO logs.
    - Concurrency capped via asyncio.Semaphore (default 128) — matches 2× your
      judge server's --batch-size 64. Keeps the GPU queue saturated without
      starving the socket layer.

Usage:
    # Make sure judge server on :8001 is up first
    bash start_judge_server.sh ${FORTE_ROOT}/judge_finetuning/models/judge-new/merged 7

        CUDA_VISIBLE_DEVICES=2 python filter_train_set.py \
        --input-parquet  ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/train.parquet \
        --output-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/train_filtered_qwen-7b.parquet \
        --sft-model-path ${FORTE_ROOT}/planner_finetuning/checkpoints_planner_sft-qwen-7b/global_step_196 \
        --reward-fn-path ${FORTE_ROOT}/planner_rl/judge_reward.py \
        --n-rollouts 4 \
        --score-low 0.4 --score-high 0.95 \
        --save-scored tmp/train_scored.parquet
"""

import sys
import json
import asyncio
import argparse
import importlib.util
from pathlib import Path
from typing import Callable, List, Tuple
import os
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vllm not installed. `pip install vllm>=0.6.0`", file=sys.stderr)
    raise
 
from transformers import AutoTokenizer
 
 
def load_compute_score(reward_fn_path: str, fn_name: str = "compute_score_v2") -> Callable:
    """Load compute_score_v2 from the user's judge_reward.py — same path GRPO uses."""
    path = Path(reward_fn_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reward function file not found: {path}")
    spec = importlib.util.spec_from_file_location("judge_reward_module", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    if not hasattr(module, fn_name):
        raise AttributeError(f"{path} has no function named '{fn_name}'")
    return getattr(module, fn_name)
 
 
def render_prompt(messages, tokenizer) -> str:
    """Convert verl's prompt (list of message dicts) into a chat-templated string."""
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            normalized.append({"role": m["role"], "content": m["content"]})
        else:
            normalized.append({"role": m['role'], "content": m['content']})
    return tokenizer.apply_chat_template(
        normalized, tokenize=False, add_generation_prompt=True
    )
 
 
def extract_call_kwargs(row: pd.Series, response_text: str) -> dict:
    """Build the kwargs for compute_score_v2 from a parquet row."""
    rm = row.get('reward_model', {})
    if isinstance(rm, dict):
        ground_truth = rm.get('ground_truth')
    elif hasattr(rm, 'item'):
        rm_unwrapped = rm.item() if rm is not None else {}
        ground_truth = rm_unwrapped.get('ground_truth') if isinstance(rm_unwrapped, dict) else None
    else:
        ground_truth = None
 
    extra_info = row.get('extra_info', None)
    if hasattr(extra_info, 'item') and not isinstance(extra_info, (str, dict, list)):
        try:
            extra_info = extra_info.item()
        except Exception:
            pass
 
    return {
        "data_source": row.get('data_source', 'toolhop_planner'),
        "solution_str": response_text,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
    }
 
 
# ─────────────────────────────────────────────────────────────────────
# Async scoring driver
# ─────────────────────────────────────────────────────────────────────
async def score_all_async(
    compute_score: Callable,
    flat_args: List[Tuple[int, pd.Series, str]],
    max_concurrent: int = 128,
) -> List[float]:
    """
    Drive the user's async compute_score_v2 with bounded concurrency.
 
    flat_args: list of (idx, row, response_text). idx preserves output ordering.
    Returns: list of scores in flat_args order.
    """
    sem = asyncio.Semaphore(max_concurrent)
    results: List[float] = [0.0] * len(flat_args)
    pbar = tqdm(total=len(flat_args), desc="Scoring rollouts", unit="resp")
 
    async def score_one(idx: int, row: pd.Series, text: str):
        async with sem:
            try:
                kwargs = extract_call_kwargs(row, text)
                result = await compute_score(**kwargs)
                if isinstance(result, dict):
                    for key in ("score", "reward", "value"):
                        if key in result:
                            result = result[key]
                            break
                    else:
                        result = 0.0
                results[idx] = float(result)
            except Exception as e:
                print(f"\n  ⚠ scoring failed at idx={idx}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                results[idx] = 0.0
            finally:
                pbar.update(1)
 
    await asyncio.gather(*[score_one(i, r, t) for (i, r, t) in flat_args])
    pbar.close()
    return results
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-parquet', required=True)
    parser.add_argument('--output-parquet', required=True)
    parser.add_argument('--sft-model-path', required=True,
                        help='SFT planner — same as actor_rollout_ref.model.path in GRPO config')
    parser.add_argument('--reward-fn-path', required=True,
                        help='Path to judge_reward.py (the one in custom_reward_function.path)')
    parser.add_argument('--reward-fn-name', default='compute_score_v2')
 
    # Rollout config
    parser.add_argument('--n-rollouts', type=int, default=4,
                        help='Rollouts per prompt for variance estimation')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Match GRPO training rollout temperature (verl default: 1.0)')
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Match data.max_response_length in GRPO config')
 
    # Filter
    parser.add_argument('--filter-by', choices=['mean', 'std'], default='mean')
    parser.add_argument('--score-low', type=float, default=0.5)
    parser.add_argument('--score-high', type=float, default=0.9)
 
    # vLLM
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--max-model-len', type=int, default=2048)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
 
    # Scoring concurrency — judge server has --batch-size 64, so 128 keeps it 2x saturated
    parser.add_argument('--max-concurrent-scoring', type=int, default=128,
                        help='Max parallel async judge calls. Set to ~2x judge --batch-size.')
 
    # Outputs
    parser.add_argument('--save-scored', type=str, default=None,
                        help='Save full df with rollout_scores/mean/std for inspection')
    parser.add_argument('--limit', type=int, default=None,
                        help='[debug] only process first N prompts')
 
    args = parser.parse_args()
 
    # ─── Load data ────────────────────────────────────────────────────
    print(f"Loading {args.input_parquet}...")
    df = pd.read_parquet(args.input_parquet)
    print(f"  {len(df)} prompts")
    print(f"  columns: {list(df.columns)}")
 
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"  [debug] limited to first {len(df)}")
 
    # ─── Load reward function ────────────────────────────────────────
    print(f"\nLoading {args.reward_fn_name} from {args.reward_fn_path}...")
    compute_score = load_compute_score(args.reward_fn_path, args.reward_fn_name)
    is_async = asyncio.iscoroutinefunction(compute_score)
    print(f"  ✓ loaded ({'async' if is_async else 'sync'})")
    if not is_async:
        print(f"  ⚠ Expected an async function but got sync. Will call synchronously.")
 
    # ─── Render all prompts ───────────────────────────────────────────
    print(f"\nRendering prompts with chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ct_path = os.path.join(args.sft_model_path, 'chat_template.jinja')
    if os.path.exists(ct_path) and not getattr(tokenizer, 'chat_template', None):
        with open(ct_path) as f:
            tokenizer.chat_template = f.read()
        print(f"  Loaded chat_template from {ct_path}")
 
    rendered_prompts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rendered_prompts.append(render_prompt(row['prompt'], tokenizer))
 
    print(f"\nFirst rendered prompt (first 400 chars):")
    print(rendered_prompts[0][:400] + "...")
 
    # ─── Spin up vLLM with SFT planner ───────────────────────────────
    print(f"\nLoading SFT planner via vLLM from {args.sft_model_path}...")
    llm = LLM(
        model=args.sft_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )
 
    # ─── Generate N rollouts per prompt ───────────────────────────────
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n_rollouts,
    )
    print(f"\nGenerating {args.n_rollouts} rollouts × {len(rendered_prompts)} prompts "
          f"= {args.n_rollouts * len(rendered_prompts)} responses")
    outputs = llm.generate(rendered_prompts, sampling_params)
 
    # ─── Flatten rollouts and score concurrently ─────────────────────
    flat_args: List[Tuple[int, pd.Series, str]] = []
    rollout_index_map: List[List[int]] = []  # per-prompt list of flat indices
    flat_idx = 0
    for prompt_idx, out in enumerate(outputs):
        row = df.iloc[prompt_idx]
        prompt_indices = []
        for completion in out.outputs:
            flat_args.append((flat_idx, row, completion.text))
            prompt_indices.append(flat_idx)
            flat_idx += 1
        rollout_index_map.append(prompt_indices)
 
    print(f"\nScoring {len(flat_args)} responses via async compute_score_v2 "
          f"(max_concurrent={args.max_concurrent_scoring})...")
    print(f"  Note: your compute_score_v2's internal progress bar will also tick.")
    print(f"  Their bar = per-call updates. My bar = overall progress.\n")
 
    if is_async:
        flat_scores = asyncio.run(
            score_all_async(compute_score, flat_args,
                            max_concurrent=args.max_concurrent_scoring)
        )
    else:
        # Sync fallback (shouldn't trigger with current judge_reward.py)
        flat_scores = []
        for (idx, row, text) in tqdm(flat_args, desc="Scoring (sync)"):
            try:
                kwargs = extract_call_kwargs(row, text)
                result = compute_score(**kwargs)
                if isinstance(result, dict):
                    result = result.get('score', result.get('reward', 0.0))
                flat_scores.append(float(result))
            except Exception as e:
                print(f"  ⚠ scoring failed at idx={idx}: {e}", file=sys.stderr)
                flat_scores.append(0.0)
 
    # ─── Reshape into per-prompt scores ──────────────────────────────
    per_prompt_scores = [
        [flat_scores[i] for i in indices]
        for indices in rollout_index_map
    ]
 
    # ─── Compute statistics ──────────────────────────────────────────
    means = np.array([np.mean(s) for s in per_prompt_scores])
    stds = np.array([np.std(s) for s in per_prompt_scores])
    mins = np.array([np.min(s) for s in per_prompt_scores])
    maxs = np.array([np.max(s) for s in per_prompt_scores])
 
    df = df.copy()
    df['rollout_scores'] = per_prompt_scores
    df['rollout_mean'] = means
    df['rollout_std'] = stds
    df['rollout_min'] = mins
    df['rollout_max'] = maxs
 
    print(f"\n" + "=" * 70)
    print("REWARD DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"  Per-prompt mean reward:")
    print(f"    overall mean:     {means.mean():.3f}")
    print(f"    median:           {np.median(means):.3f}")
    print(f"    quartiles (25/50/75): {np.percentile(means, [25, 50, 75])}")
    print(f"  Per-prompt std (within-group variance — what GRPO needs):")
    print(f"    mean of stds:     {stds.mean():.3f}")
    print(f"    pct with std=0:   {(stds == 0).mean() * 100:.1f}%   ← zero gradient contribution")
    print(f"    pct with std>0.05: {(stds > 0.05).mean() * 100:.1f}%   ← useful for GRPO")
    print(f"  Saturation breakdown:")
    print(f"    pct mean ≥ 0.95:        {(means >= 0.95).mean() * 100:.1f}%   ← saturated, drop")
    print(f"    pct mean in [0.5, 0.9]: {((means >= 0.5) & (means <= 0.9)).mean() * 100:.1f}%   ← Goldilocks")
    print(f"    pct mean < 0.5:         {(means < 0.5).mean() * 100:.1f}%   ← too hard")
 
    if args.save_scored:
        df.to_parquet(args.save_scored)
        print(f"\nFull scored df saved to {args.save_scored}")
 
    # ─── Apply filter ────────────────────────────────────────────────
    if args.filter_by == 'mean':
        mask = (means >= args.score_low) & (means <= args.score_high)
        filter_desc = f"mean ∈ [{args.score_low}, {args.score_high}]"
    else:
        mask = stds > args.score_low
        filter_desc = f"std > {args.score_low}"
 
    print(f"\n" + "=" * 70)
    print(f"FILTERING by {filter_desc}")
    print("=" * 70)
    print(f"  Keeping {mask.sum()}/{len(df)} prompts ({100 * mask.mean():.1f}%)")
 
    if mask.sum() == 0:
        print("\n  ❌ Filter kept zero prompts. Loosen --score-low / --score-high.")
        sys.exit(1)
    if mask.sum() < 100:
        print(f"\n  ⚠ Only {mask.sum()} prompts survived. Consider widening the range.")
 
    filtered = df[mask].reset_index(drop=True)
    drop_cols = [c for c in ['rollout_scores', 'rollout_mean', 'rollout_std',
                             'rollout_min', 'rollout_max'] if c in filtered.columns]
    filtered = filtered.drop(columns=drop_cols)
    filtered.to_parquet(args.output_parquet)
 
    print(f"\n✅ Filtered train set saved to {args.output_parquet}")
    print(f"   {len(filtered)} prompts ready for GRPO retraining\n")
    print(f"   Update your bash script:")
    print(f"     data.train_files={args.output_parquet}")
    print(f"   And the other fixes from our earlier discussion:")
    print(f"     actor_rollout_ref.actor.entropy_coeff=1e-3")
    print(f"     actor_rollout_ref.rollout.temperature=1.2")
    print(f"     actor_rollout_ref.actor.optim.lr=5e-7")
 
 
if __name__ == '__main__':
    main()