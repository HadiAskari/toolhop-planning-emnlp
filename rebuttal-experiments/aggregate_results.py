#!/usr/bin/env python3
"""
Aggregate all rebuttal-experiment outputs into the numbers the rebuttal draft
needs, keyed to its [X.X] placeholders.

Scans the rebuttal-experiments tree (and any extra --extra-dirs) for:
  e1_execution/*.execution.json + e1_gold_check_*.json      (E1)
  e2_compound_errors/e2_compound_*.json                     (E2)
  e3_matched_budget_gpt/*stats.json                         (E3)
  e4_judge_transfer/e4_transfer_*.json                      (E4)
  e5_organic_errors/e5_organic_errors.json                  (E5)
  e6_human_agreement/e6_agreement_*.json                    (E6)

Usage:  python aggregate_results.py [--extra-dirs DIR ...]
"""

import argparse
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] could not read {p}: {e}")
        return None


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%" if isinstance(x, (int, float)) else "—"


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-dirs", nargs="*", default=[])
    args = ap.parse_args()
    roots = [HERE] + [Path(d) for d in args.extra_dirs]

    def find(pattern):
        hits = []
        for r in roots:
            hits.extend(sorted(glob.glob(str(r / pattern), recursive=True)))
        return hits

    # ── E1 ───────────────────────────────────────────────────────────────────
    section("E1 — Execution-based end-task accuracy  "
            "(Gen. Response pt 1; kocM C1; VJ8B W1/W2; fiFx W3c)")
    for p in find("e1_execution/e1_gold_check_*.json"):
        d = _load(p)
        if d:
            s = d["stats"]
            print(f"  GOLD CEILING {s['label']:35s} "
                  f"exec-acc={pct(s.get('end_task_accuracy_strict'))} "
                  f"coverage={pct(s.get('coverage'))}")
    rows = []
    exec_hits = find("**/*.execution.json")
    exec_hits += sorted(glob.glob(str(HERE.parent / "scripts" / "**" /
                                      "*.execution.json"), recursive=True))
    for p in dict.fromkeys(exec_hits):
        d = _load(p)
        if d:
            s = d["stats"]
            rows.append((s["label"],
                         s.get("end_task_accuracy_strict"),
                         s.get("execution_completion_rate"),
                         s.get("exec_accuracy_given_judge_pass"),
                         s.get("n_covered")))
    if rows:
        print(f"  {'method (results file:run)':45s} {'exec-acc':>9s} "
              f"{'completes':>10s} {'acc|judge-pass':>15s} {'n':>5s}")
        for label, acc, comp, ajp, n in sorted(rows, key=lambda r: -(r[1] or 0)):
            print(f"  {label[:45]:45s} {pct(acc):>9s} {pct(comp):>10s} "
                  f"{pct(ajp):>15s} {n:>5d}")
        print("  -> placeholders: FORTE vs baseline vs GPT execution accuracy; "
              "'X% of judge-passing plans execute correctly' (VJ8B W2 = acc|judge-pass)")
    else:
        print("  (no .execution.json yet — run E1 on the method result files)")

    # ── E2 ───────────────────────────────────────────────────────────────────
    section("E2 — Compound-error stress test  (fiFx W2a)")
    for p in find("e2_compound_errors/e2_compound_*.json"):
        d = _load(p)
        if not d or "stats" not in d:
            continue
        s = d["stats"]
        print(f"  [{s['dataset']}]")
        for kind in ("gold", "single", "double", "triple"):
            k = s.get(kind) or {}
            if not k:
                continue
            det = pct(k.get("detection_rate")) if kind != "gold" else "—"
            sep = (f"{k.get('separation_from_gold', 0):.1f}"
                   if kind != "gold" else "—")
            print(f"    {kind:8s} n={k['n']:5d} mean={k['mean_score']:.1f} "
                  f"detected={det} separation={sep}")

    # ── E3 ───────────────────────────────────────────────────────────────────
    section("E3 — Matched-budget GPT Best-of-5  (VJ8B W4; Gen. Response pt 3)")
    for p in find("e3_matched_budget_gpt/*stats.json"):
        d = _load(p)
        if not d:
            continue
        model = d.get("config", {}).get("model", "?")
        ds = d.get("config", {}).get("resolved_dataset", "?")
        for run, s in d.get("runs", {}).items():
            print(f"  [{ds}:{run}] {model}: JSR={pct(s.get('jsr'))} "
                  f"FM={pct(s.get('functional_match'))} "
                  f"PA={pct(s.get('param_accuracy'))} "
                  f"DA={pct(s.get('dependency_accuracy'))} "
                  f"(temp supported: {s.get('temperature_supported')})")
        print("    vs paper Table 3 greedy GPT-5.5: ToolHop 59.0/50.0/56.9/82.5, "
              "NESTFUL 89.0/43.0/55.0/73.0; FORTE-Qwen7B: 96.6/97.3/77.1/98.9, "
              "98.6/44.3/61.6/89.0")

    # ── E4 ───────────────────────────────────────────────────────────────────
    section("E4 — Cross-dataset judge transfer  (VJ8B W3)")
    cells = {}
    for p in find("e4_judge_transfer/e4_transfer_*.json"):
        d = _load(p)
        if not d:
            continue
        s = d["stats"]
        cells[(s["judge_label"], s["dataset"])] = s
    for (jl, ds), s in sorted(cells.items()):
        domain = ("IN-DOMAIN " if ds in jl else "CROSS     ")
        print(f"  {domain} {jl:16s} on {ds:8s}: "
              f"AUC={s['auc_gold_vs_negative']:.3f} "
              f"sep={s['separation']:.1f} "
              f"neg-detect={pct(s['negative_detection_rate'])} "
              f"gold-pass={pct(s['gold_pass_rate'])}")

    # ── E5 ───────────────────────────────────────────────────────────────────
    section("E5 — Judge on organic (non-injected) failures  (fiFx W2b)")
    for p in find("e5_organic_errors/e5_organic_errors*.json"):
        d = _load(p)
        if not d:
            continue
        for m, s in (d.get("per_method") or {}).items():
            print(f"  {m:28s} failures={s['n_organic_failures']:4d} "
                  f"judge-mean={s['judge_mean_on_failures']:.1f} "
                  f"flagged={pct(s['judge_flag_rate_on_failures'])} "
                  f"(criterion: {s['failure_criterion']})")
        c = d.get("combined") or {}
        if c:
            print(f"  COMBINED: n={c.get('n_failures_total')} "
                  f"mean={c.get('judge_mean_on_failures', float('nan')):.1f} "
                  f"flagged={pct(c.get('judge_flag_rate_on_failures'))}")
            if c.get("gold_reference"):
                print(f"  gold reference mean: "
                      f"{c['gold_reference']['judge_mean_on_gold']:.1f}")
        if d.get("rescored"):
            print(f"  predicted error types on organic failures: "
                  f"{d['rescored']['predicted_error_type_distribution']}")

    # ── E6 ───────────────────────────────────────────────────────────────────
    section("E6 — Human agreement  (fiFx W3b)")
    for p in find("e6_human_agreement/e6_agreement_*.json"):
        d = _load(p)
        if not d:
            continue
        for sec_name, pairs in d["stats"].items():
            if not isinstance(pairs, dict):
                continue
            for name, s in pairs.items():
                if isinstance(s, dict):
                    print(f"  {sec_name:22s} {name:16s} n={s['n']:4d} "
                          f"agree={pct(s['percent_agreement'])} "
                          f"kappa={s['cohens_kappa']:.3f}")

    print("\nDone. Paste these numbers into the rebuttal draft placeholders.")


if __name__ == "__main__":
    main()
