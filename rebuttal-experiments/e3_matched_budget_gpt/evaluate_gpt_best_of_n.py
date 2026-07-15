#!/usr/bin/env python3
"""
E3 — Matched-budget closed-source comparison (answers VJ8B W4: "FORTE uses
Best-of-5 plus a trained judge selector, while closed-model baselines are
greedy single-shot — not fair").

Runs GPT-5.5 (or any OpenAI Responses-API model) under FORTE's IDENTICAL
inference protocol: N=5 candidates, one per temperature in
{0.2, 0.4, 0.6, 0.8, 1.0}, all scored by the same frozen judge, best selected
by judge score (ties: confidence, then shorter plan).

The GPT call logic (Responses API, reasoning-effort minimal for gpt-5*,
automatic drop of unsupported params) is copied verbatim from
evaluate_gpt_baseline.py, and the record format matches the Best-of-N results
files so E1 execution eval and aggregate_results.py consume the output as-is.

IMPORTANT CAVEAT (record and report honestly): if the target model's API
rejects the `temperature` parameter (GPT-5 family may), the script still
draws 5 independent samples but temperature diversity is not enforced; the
output config records `temperature_supported` per run so the rebuttal can
state exactly what was run.

Requires: judge server (1 GPU) + OPENAI_API_KEY. ~100 queries x 5 samples
x 2 datasets = 1000 GPT calls.

Usage:
  python evaluate_gpt_best_of_n.py --dataset toolhop \\
      --test-parquet $ROOT/planner_rl/data/verl_rl_full_clean/test.parquet \\
      --model gpt-5.5 --perfect-only --full
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import (  # noqa: E402
    SYSTEM_PROMPT, build_user_message, load_test_parquet, parse_plan_steps,
    evaluate_plan_vs_gt, score_plan_via_judge, set_judge_url,
    check_judge_server, save_json, ARTIFACT_ERROR_TYPES,
    DEFAULT_TEMPERATURE_LADDER, mean,
)


# ══════════════════════════════════════════════════════════════════════════════
# VENDORED from evaluate_gpt_baseline.py
# ══════════════════════════════════════════════════════════════════════════════

def _clean_response(text: str) -> str:
    """Strip markdown code fences. Step-line filtering is handled by the parser."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def call_gpt(client, system: str, user: str, model: str,
             temperature: float, max_tokens: int,
             max_retries: int = 4) -> tuple:
    """
    OpenAI Responses API call with automatic fallback for unsupported params.
    Returns (text, error_str, temperature_used: bool).
    """
    prompt = f"{system}\n\n{user}"

    use_temperature = True
    use_reasoning_minimal = "gpt-5" in model.lower()  # only GPT-5 supports this

    last_err = ""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "input": prompt,
                "max_output_tokens": max_tokens,
                "text": {"format": {"type": "text"}},
            }
            if use_temperature:
                kwargs["temperature"] = temperature
            if use_reasoning_minimal:
                kwargs["reasoning"] = {"effort": "minimal"}

            resp = client.responses.create(**kwargs)
            return resp.output_text.strip(), "", use_temperature

        except Exception as e:
            last_err = str(e)
            low = last_err.lower()

            dropped = False
            if use_temperature and "temperature" in low and \
                    ("unsupported" in low or "not supported" in low):
                use_temperature = False
                dropped = True
            if use_reasoning_minimal and "reasoning" in low and \
                    ("unsupported" in low or "not supported" in low):
                use_reasoning_minimal = False
                dropped = True
            if dropped:
                continue

            if "rate" in low or "429" in low:
                time.sleep(min(2 ** attempt, 30))
            elif "timeout" in low:
                time.sleep(min(2 ** attempt, 10))
            elif attempt < max_retries - 1:
                time.sleep(1.0)
            else:
                return "", last_err, use_temperature
    return "", f"Max retries exceeded: {last_err}", use_temperature


# ══════════════════════════════════════════════════════════════════════════════
# Best-of-N over GPT samples (mirrors best_of_n_selection_*.py)
# ══════════════════════════════════════════════════════════════════════════════

def _pick_best(fast_scores, candidates):
    best_idx = 0
    best_key = None
    for i, s in enumerate(fast_scores):
        key = (s["quality_score"], s["confidence"], -len(candidates[i]))
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i
    return best_idx


def evaluate_bon(client, examples, perfect_gt_by_qid, dataset, args, pool):
    temps = DEFAULT_TEMPERATURE_LADDER if args.n == 5 else \
        list(np.linspace(0.2, 1.0, args.n))
    results = []
    temp_support_seen = set()

    for idx, ex in enumerate(examples):
        question, tools, gt = ex["question"], ex["tools"], ex["ground_truth"]
        gt_steps = parse_plan_steps(gt)
        user_msg = build_user_message(question, tools)

        # 5 GPT samples, one per temperature (parallel)
        def one(t):
            raw, err, temp_used = call_gpt(client, SYSTEM_PROMPT, user_msg,
                                           args.model, t, args.max_output_tokens)
            return _clean_response(raw) if raw else "", err, temp_used

        gen_out = list(pool.map(one, temps))
        candidates = [g[0] for g in gen_out]
        gpt_errors = [g[1] for g in gen_out]
        temp_support_seen.update(g[2] for g in gen_out)

        # judge all candidates (fast pass)
        fast_scores = list(pool.map(
            lambda p: score_plan_via_judge(question, p or "(empty plan)", tools,
                                           max_tokens=args.judge_max_tokens),
            candidates))
        best_idx = _pick_best(fast_scores, candidates)
        best_plan = candidates[best_idx]

        # rich re-score of the winner
        best_score = score_plan_via_judge(question, best_plan or "(empty plan)",
                                          tools, max_tokens=args.judge_eval_max_tokens)

        ref_is_perfect = (ex["error_type"] == "none" and ex["quality_score"] >= 100)
        judge_success = best_score["quality_score"] >= 80

        if (ex["error_type"] in ARTIFACT_ERROR_TYPES
                and ex["query_id"] in perfect_gt_by_qid):
            struct_gt = parse_plan_steps(perfect_gt_by_qid[ex["query_id"]])
        else:
            struct_gt = gt_steps
        struct_eval = evaluate_plan_vs_gt(parse_plan_steps(best_plan), struct_gt,
                                          tools=tools)

        error_type_handled = (judge_success if ref_is_perfect
                              else best_score["quality_score"] >= ex["quality_score"])

        results.append({
            "dataset":               dataset,
            "query_id":              ex["query_id"],
            "question":              question,
            "error_type":            ex["error_type"],
            "ref_quality_score":     ex["quality_score"],
            "ref_is_perfect":        ref_is_perfect,
            "ground_truth":          gt,
            "best_plan":             best_plan,
            "best_candidate_idx":    best_idx,
            "best_candidate_temperature": temps[best_idx],
            "judge_success":         judge_success,
            "best_judge_score":      best_score["quality_score"],
            "best_success":          best_score["success_prediction"],
            "best_confidence":       best_score["confidence"],
            "judge_full_parse":      best_score.get("_full_parse", False),
            "bon1_judge_score":      fast_scores[0]["quality_score"],
            "bon1_temperature":      temps[0],
            "all_candidate_scores":  [s["quality_score"] for s in fast_scores],
            "all_candidate_temperatures": list(temps),
            "mean_candidate_score":  float(np.mean([s["quality_score"] for s in fast_scores])),
            "candidate_score_std":   float(np.std([s["quality_score"] for s in fast_scores])),
            "gpt_errors":            [e for e in gpt_errors if e],
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
            "error_type_handled":         error_type_handled,
            "judge_agrees_with_ref":      (ref_is_perfect == judge_success),
            "n_candidates":               args.n,
            "method":                     f"GPT-BoN-{args.model}",
        })
        if (idx + 1) % 10 == 0:
            jsr = mean([1.0 if r["judge_success"] else 0.0 for r in results])
            print(f"  [{idx + 1}/{len(examples)}] running JSR={jsr * 100:.1f}%", flush=True)

    return results, temp_support_seen


def compute_stats(results, label):
    if not results:
        return {}
    return {
        "label": label,
        "n_examples": len(results),
        "jsr": mean([1.0 if r["judge_success"] else 0.0 for r in results]),
        "mean_judge_score": mean([r["best_judge_score"] for r in results]),
        "functional_match": mean([1.0 if r["functional_match"] else 0.0 for r in results]),
        "param_accuracy": mean([r["param_accuracy"] for r in results]),
        "dependency_accuracy": mean([r["dependency_accuracy"] for r in results]),
        "exact_match": mean([1.0 if r["exact_match"] else 0.0 for r in results]),
        "step_count_match": mean([1.0 if r["step_count_match"] else 0.0 for r in results]),
        "bon_gain_vs_first_sample": (mean([r["best_judge_score"] for r in results])
                                     - mean([r["bon1_judge_score"] for r in results])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--dataset", default="auto", choices=["auto", "toolhop", "nestful"])
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--api-key", default=None, help="default: $OPENAI_API_KEY")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--judge-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--judge-max-tokens", type=int, default=32)
    ap.add_argument("--judge-eval-max-tokens", type=int, default=300)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--perfect-only", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--output", default="e3_gpt_bon_results.json")
    ap.add_argument("--stats-output", default=None)
    args = ap.parse_args()

    if not (args.perfect_only or args.full):
        sys.exit("Specify at least one of --perfect-only / --full "
                 "(--perfect-only = the 100 gold-reference rows, matching Table 3).")

    import os
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY or pass --api-key.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    set_judge_url(args.judge_url)
    if not check_judge_server(args.judge_url):
        sys.exit(f"Judge server not reachable at {args.judge_url}.")

    dataset = args.dataset
    if dataset == "auto":
        dataset = "nestful" if "nestful" in args.test_parquet.lower() else "toolhop"

    # never persist the API key in result files
    safe_config = {**vars(args), "resolved_dataset": dataset}
    if safe_config.get("api_key"):
        safe_config["api_key"] = "REDACTED"
    all_output = {"config": dict(safe_config), "runs": {}}
    all_stats = {"config": dict(safe_config), "runs": {}}

    run_specs = []
    if args.perfect_only:
        run_specs.append(("perfect_only", True))
    if args.full:
        run_specs.append(("full", False))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, perfect in run_specs:
            examples, perfect_gt_by_qid = load_test_parquet(args.test_parquet, perfect)
            if args.limit:
                examples = examples[: args.limit]
            print(f"[run {name}] {len(examples)} examples, model={args.model}, "
                  f"N={args.n}")
            results, temp_support = evaluate_bon(client, examples, perfect_gt_by_qid,
                                                 dataset, args, pool)
            all_output["runs"][name] = results
            stats = compute_stats(results, f"GPT-BoN-{args.model} — {name}")
            stats["temperature_supported"] = (False not in temp_support)
            all_stats["runs"][name] = stats
            print(json.dumps(stats, indent=2))
            if not stats["temperature_supported"]:
                print("  [!] model rejected `temperature`; the 5 samples are "
                      "independent draws without enforced temperature diversity. "
                      "State this in the rebuttal.")

    save_json(all_output, args.output)
    save_json(all_stats, args.stats_output or args.output.replace(".json", ".stats.json"))


if __name__ == "__main__":
    main()
