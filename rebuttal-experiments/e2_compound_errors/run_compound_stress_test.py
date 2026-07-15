#!/usr/bin/env python3
"""
E2 — Compound-error judge stress test (answers fiFx W2: "single-error
negatives are too simple; the judge may memorize the injected perturbations").

Protocol:
  1. Take the GOLD plans of the held-out test split (never used to train the
     judge).
  2. Inject 2 and 3 SIMULTANEOUS taxonomy errors per plan by composing the
     paper's own deterministic injectors (imported from
     toolhop_plan_generator.py / nestful_annotator.py), a distribution the
     judge never saw in training. Every injection is verified to actually
     change the serialized plan; silent no-ops are re-drawn.
  3. Score gold plans, the corpus's stored single-error negatives, and the new
     compound negatives with the FROZEN trained judge (vLLM server).
  4. Report detection rate (score < 80) and gold-vs-negative score separation
     for single / double / triple errors.

Requires the judge server (start_judge_server.sh) — 1 GPU.

Usage:
  python run_compound_stress_test.py --dataset toolhop --judge-url http://localhost:8001/v1/chat/completions
  python run_compound_stress_test.py --dataset nestful --n-double 3 --n-triple 2 --judge-url http://localhost:8002/v1/chat/completions
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.forte_common import (  # noqa: E402
    find_data_file, import_module_from_file, load_annotated_corpus,
    load_canonical_splits, corpus_records_by_qid, gold_record,
    format_steps_as_plan_string, set_judge_url, check_judge_server,
    batch_score_plans, mean, save_json, ANNOTATED_ERROR_ORDER,
)

# error types eligible for composition (all nine)
ERROR_TYPES = [e for e in ANNOTATED_ERROR_ORDER if e != "none"]

# pairs that cannot compose meaningfully (each list is (first, second) to skip):
# incomplete_plan drops the final step, so injecting it LAST keeps other edits;
# we always apply incomplete_plan last if drawn, and never with circular or
# forward on 2-step plans (handled dynamically by change-verification).
_APPLY_LAST = {"incomplete_plan"}


def build_injector(dataset: str, repo_root):
    """Load the paper's generator module and return (module, make_generator)."""
    if dataset == "toolhop":
        path = find_data_file("toolhop_plan_generator", repo_root=repo_root)
        mod = import_module_from_file(path, "forte_toolhop_generator")
    else:
        path = find_data_file("nestful_annotator", repo_root=repo_root)
        mod = import_module_from_file(path, "forte_nestful_annotator")
    print(f"[inject] using injectors from {path}")
    return mod


def steps_to_plan_obj(mod, steps, error_type="none"):
    """Rebuild a generator-module Plan object from corpus step dicts."""
    step_objs = [mod.Step(**{k: s.get(k) for k in
                             ("step_id", "tool_name", "parameters",
                              "output_variable", "expected_output")})
                 for s in steps]
    return mod.Plan(steps=[mod.Step(**{
        "step_id": s.step_id, "tool_name": s.tool_name,
        "parameters": dict(s.parameters), "output_variable": s.output_variable,
        "expected_output": s.expected_output}) for s in step_objs],
        error_type=error_type)


def serialize_plan(mod, plan) -> str:
    lines = []
    for s in plan.steps:
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in s.parameters.items())
        lines.append(f"Step {s.step_id}: {s.output_variable} = {s.tool_name}({params_str})")
    return "\n".join(lines)


def apply_injector(mod, gen, plan, error_type):
    """Apply one injector; normalize the (Plan) vs (Plan, bool) return styles.
    Returns (new_plan, changed: bool) where changed is verified by comparing
    serialized plans, not trusted from the injector."""
    fn = getattr(gen, f"_inject_{error_type}")
    before = serialize_plan(mod, plan)
    out = fn(plan)
    new_plan = out[0] if isinstance(out, tuple) else out
    after = serialize_plan(mod, new_plan)
    return new_plan, after != before


def make_compound(mod, gen, gold_steps, k, rng, max_tries=25):
    """Compose k distinct injectors on a fresh copy of the gold plan.
    Returns (plan_str, applied_error_list) or None if impossible."""
    for _ in range(max_tries):
        errors = rng.sample(ERROR_TYPES, k)
        # apply plan-truncating edits last so they don't erase earlier edits
        errors.sort(key=lambda e: (e in _APPLY_LAST))
        plan = steps_to_plan_obj(mod, gold_steps)
        applied = []
        for e in errors:
            plan, changed = apply_injector(mod, gen, plan, e)
            if changed:
                applied.append(e)
        if len(applied) == k:
            return serialize_plan(mod, plan), applied
    return None


def get_tools_for_query(dataset, qid, toolhop_data, nestful_by_sample, corpus_rec, mod):
    """Tools dict in the shape the judge prompt builder expects."""
    if dataset == "toolhop":
        entry = toolhop_data[qid]
        return mod.reindex_tools_by_api_name(entry["tools"])
    sample = nestful_by_sample.get(corpus_rec.get("sample_id"))
    if sample is None:
        return {}
    return mod.NestfulSchemaAdapter.normalize_tools(sample["tools"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["toolhop", "nestful"], required=True)
    ap.add_argument("--judge-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--n-double", type=int, default=3,
                    help="compound 2-error negatives per query (default 3)")
    ap.add_argument("--n-triple", type=int, default=2,
                    help="compound 3-error negatives per query (default 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-max-tokens", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap test queries (smoke test)")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--skip-singles", action="store_true",
                    help="skip re-scoring the stored single-error negatives")
    ap.add_argument("--output", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="generate compound negatives without scoring (no judge needed)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if not args.dry_run:
        set_judge_url(args.judge_url)
        if not check_judge_server(args.judge_url):
            sys.exit(f"Judge server not reachable at {args.judge_url}. "
                     f"Start it with scripts/planner_rl/start_judge_server.sh "
                     f"<judge_merged_model_path> <gpu_id>")

    mod = build_injector(args.dataset, args.repo_root)
    corpus = load_annotated_corpus(args.dataset, repo_root=args.repo_root)
    by_qid = corpus_records_by_qid(corpus)
    splits = load_canonical_splits(args.dataset, repo_root=args.repo_root)
    qids = splits["test_qids"]
    if args.limit:
        qids = qids[: args.limit]

    toolhop_data = None
    nestful_by_sample = {}
    if args.dataset == "toolhop":
        toolhop_data = json.load(open(find_data_file("ToolHop.json",
                                                     repo_root=args.repo_root)))
    else:
        nf_path = find_data_file("nestful_data", repo_root=args.repo_root)
        with open(nf_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    nestful_by_sample[s["sample_id"]] = s

    # ── build the evaluation pool ────────────────────────────────────────────
    items, meta = [], []   # meta: (qid, kind, applied_errors)
    n_gen_fail = 0
    for qid in qids:
        recs = by_qid.get(qid, [])
        g = gold_record(recs)
        if g is None:
            continue
        tools = get_tools_for_query(args.dataset, qid, toolhop_data,
                                    nestful_by_sample, g, mod)
        gen = mod.PlanGenerator(tools)
        gold_steps = g["plan"]["steps"]
        query = g["query"]

        items.append({"query": query, "plan_str": format_steps_as_plan_string(gold_steps),
                      "tools": tools})
        meta.append((qid, "gold", []))

        if not args.skip_singles:
            for rec in recs:
                et = rec["plan"]["error_type"]
                if et == "none":
                    continue
                # verify the stored negative actually differs from gold
                neg_str = format_steps_as_plan_string(rec["plan"]["steps"])
                if neg_str == format_steps_as_plan_string(gold_steps):
                    continue  # corpus no-op fallback; not a real negative
                items.append({"query": query, "plan_str": neg_str, "tools": tools})
                meta.append((qid, "single", [et]))

        for _ in range(args.n_double):
            out = make_compound(mod, gen, gold_steps, 2, rng)
            if out is None:
                n_gen_fail += 1
                continue
            items.append({"query": query, "plan_str": out[0], "tools": tools})
            meta.append((qid, "double", out[1]))
        for _ in range(args.n_triple):
            out = make_compound(mod, gen, gold_steps, 3, rng)
            if out is None:
                n_gen_fail += 1
                continue
            items.append({"query": query, "plan_str": out[0], "tools": tools})
            meta.append((qid, "triple", out[1]))

    kinds = [m[1] for m in meta]
    print(f"[pool] {len(items)} plans over {len(qids)} queries "
          f"(gold={kinds.count('gold')}, single={kinds.count('single')}, "
          f"double={kinds.count('double')}, triple={kinds.count('triple')}; "
          f"generation failures={n_gen_fail})")

    default_out = str(Path(__file__).parent / f"e2_compound_{args.dataset}.json")
    out_path = args.output or default_out

    if args.dry_run:
        pool_dump = [{"qid": m[0], "kind": m[1], "errors": m[2],
                      "plan_str": it["plan_str"], "query": it["query"]}
                     for it, m in zip(items, meta)]
        save_json({"config": vars(args), "pool": pool_dump}, out_path)
        print("[dry-run] pool written; no judge scoring performed.")
        return

    # ── judge scoring ────────────────────────────────────────────────────────
    annotations = batch_score_plans(items, max_tokens=args.judge_max_tokens,
                                    workers=args.workers, desc="E2 judge")

    records = []
    for (qid, kind, errors), it, ann in zip(meta, items, annotations):
        records.append({
            "query_id": qid, "kind": kind, "errors": errors,
            "judge_score": ann["quality_score"],
            "judge_success_pred": ann.get("success_prediction"),
            "judge_reasoning": ann.get("reasoning", ""),
            "judge_call_failed": bool(ann.get("_call_failed", False)),
            "detected": ann["quality_score"] < 80,
            "plan_str": it["plan_str"],
        })

    def stats_for(kind):
        rs = [r for r in records if r["kind"] == kind]
        if not rs:
            return {}
        return {
            "n": len(rs),
            "mean_score": mean([r["judge_score"] for r in rs]),
            "detection_rate": mean([1.0 if r["detected"] else 0.0 for r in rs]),
        }

    gold_mean = stats_for("gold").get("mean_score")
    stats = {
        "dataset": args.dataset,
        "gold": stats_for("gold"),
        "single": stats_for("single"),
        "double": stats_for("double"),
        "triple": stats_for("triple"),
    }
    for kind in ("single", "double", "triple"):
        if stats[kind] and gold_mean is not None:
            stats[kind]["separation_from_gold"] = gold_mean - stats[kind]["mean_score"]

    save_json({"config": vars(args), "stats": stats, "records": records}, out_path)

    print("\n════════ E2 COMPOUND-ERROR SUMMARY ════════")
    print(f"{'kind':8s} {'n':>5s} {'mean score':>11s} {'detected(<80)':>14s} {'sep. vs gold':>13s}")
    for kind in ("gold", "single", "double", "triple"):
        s = stats[kind]
        if not s:
            continue
        det = f"{s['detection_rate'] * 100:.1f}%" if kind != "gold" else "-"
        sep = f"{s.get('separation_from_gold', 0):.1f}" if kind != "gold" else "-"
        print(f"{kind:8s} {s['n']:5d} {s['mean_score']:11.1f} {det:>14s} {sep:>13s}")


if __name__ == "__main__":
    main()
