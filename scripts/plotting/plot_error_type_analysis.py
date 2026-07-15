#!/usr/bin/env python3
"""
Per-error-type reduction analysis plots for the FORTE paper.

Generates four figures from the per_error_type stats in baseline JSON files:

  Each figure is saved as BOTH .pdf (for the paper) and .png (for quick
  preview in VS Code / Jupyter without needing to download a PDF reader).

  1. fig_error_rate_by_type.{pdf,png}
     Grouped bar chart: 9 error types × N methods, showing error rate
     (1 - judge_success_rate). Shows which errors your method reduces most.

  2. fig_category_breakdown.{pdf,png}
     3-category aggregated bar chart (argument-level / dependency / structural).
     Tells the narrative of which error class drives your gains.

  3. fig_judge_score_distribution.{pdf,png}
     Box plot of judge scores per error type, one subplot per method.
     Shows distribution shift, not just mean.

  4. fig_pairwise_gain_heatmap.{pdf,png}
     Heatmap: error types × methods, cell = judge_success_rate improvement
     over SFT baseline. Redundant with #1 but reads differently.

Usage:
    python plot_error_type_analysis.py \
        --stats sft_stats.json ppo_stats.json lats_stats.json ... \
        --labels "SFT" "PPO (ours)" "LATS" ... \
        --outdir figures/

All JSONs must have 'runs.full.per_error_type' populated (or
'runs.perfect_only.per_error_type' — pass --run perfect_only).
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ── Error type taxonomy with category mapping ────────────────────────────────
# Adjust these based on your paper's taxonomy. The grouping drives Figure 2.
ERROR_TYPES = [
    "parameter_typo",
    "type_mismatch",
    "missing_dependency",
    "wrong_tool",
    "circular_dependency",
    "forward_reference",
    "inefficient_order",
    "unnecessary_steps",
    "incomplete_plan",
]

CATEGORY_MAP = {
    # Argument-level: errors in individual parameter values or types
    "parameter_typo":     "Argument-level",
    "type_mismatch":      "Argument-level",
    # Dependency: errors in how steps reference each other via {{N}}
    "missing_dependency":   "Dependency",
    "circular_dependency":  "Dependency",
    "forward_reference":    "Dependency",
    # Structural: errors in plan shape, step selection, or ordering
    "wrong_tool":         "Structural",
    "inefficient_order":  "Structural",
    "unnecessary_steps":  "Structural",
    "incomplete_plan":    "Structural",
}

CATEGORIES = ["Argument-level", "Dependency", "Structural"]

# Human-readable labels for axes
ERROR_LABEL = {
    "parameter_typo":       "Param Typo",
    "type_mismatch":        "Type Mismatch",
    "missing_dependency":   "Missing Dep",
    "wrong_tool":           "Wrong Tool",
    "circular_dependency":  "Circular Dep",
    "forward_reference":    "Forward Ref",
    "inefficient_order":    "Inefficient Order",
    "unnecessary_steps":    "Unnecessary Steps",
    "incomplete_plan":      "Incomplete Plan",
}

# Consistent colors across all figures — ours (PPO) always in red-orange,
# SFT baseline in neutral gray, other methods in a cool palette.
def build_palette(method_labels: List[str]) -> Dict[str, str]:
    palette: Dict[str, str] = {}
    cool_colors = ["#4C78A8", "#72B7B2", "#54A24B", "#B279A2", "#9D755D"]
    cool_idx = 0
    for label in method_labels:
        low = label.lower()
        if "ours" in low or "forte" in low or "ppo" in low:
            palette[label] = "#E45756"    # red-orange — our method
        elif "sft" in low and "bon" not in low:
            palette[label] = "#B0B0B0"    # neutral gray — SFT baseline
        elif "sft" in low and "bon" in low:
            palette[label] = "#707070"    # darker gray — SFT+BoN
        else:
            palette[label] = cool_colors[cool_idx % len(cool_colors)]
            cool_idx += 1
    return palette


# ── Plot style ───────────────────────────────────────────────────────────────
def set_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def save_both_formats(fig, outpath: Path):
    """
    Save figure as both PDF (for paper) and PNG (for VS Code preview).
    `outpath` should end in .pdf; the PNG is written to the same stem.
    """
    outpath = Path(outpath)
    pdf_path = outpath.with_suffix(".pdf")
    png_path = outpath.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)  # lower dpi for PNG = smaller preview file
    print(f"  ✓ Saved {pdf_path}")
    print(f"  ✓ Saved {png_path}")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_per_error_from_stats(path: str, run_key: str = "full") -> Dict[str, Dict]:
    """
    Returns {error_type: {judge_success_rate, mean_judge_score, ...}} for one method.
    """
    with open(path) as f:
        data = json.load(f)
    per_err = data.get("runs", {}).get(run_key, {}).get("per_error_type", {})
    if not per_err:
        raise ValueError(
            f"{path}: runs.{run_key}.per_error_type is empty or missing. "
            f"Available runs: {list(data.get('runs', {}).keys())}"
        )
    return per_err


def load_raw_results(path: str, run_key: str = "full") -> List[Dict]:
    """
    Returns the raw per-example result list (for box plots).
    Looks for runs.full if stats file, else expects results schema.
    """
    with open(path) as f:
        data = json.load(f)
    runs = data.get("runs", {})
    if run_key not in runs:
        return []
    block = runs[run_key]
    # Block could be either stats-dict or raw-list
    if isinstance(block, list):
        return block
    return []


# ── Figure 1: error rate by type ─────────────────────────────────────────────
def plot_error_rate_by_type(
    per_error_by_method: Dict[str, Dict[str, Dict]],
    method_order: List[str],
    palette: Dict[str, str],
    outpath: Path,
):
    """Grouped bar chart: 9 error types × N methods, y = error rate."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    n_methods = len(method_order)
    n_types = len(ERROR_TYPES)
    bar_width = 0.8 / n_methods
    x = np.arange(n_types)

    for m_idx, method in enumerate(method_order):
        error_rates = []
        for et in ERROR_TYPES:
            entry = per_error_by_method[method].get(et, {})
            jsr = entry.get("judge_success_rate", 0.0)
            error_rates.append(100 * (1 - jsr))
        offset = (m_idx - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, error_rates, bar_width,
            label=method, color=palette[method],
            edgecolor="black", linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ERROR_LABEL[et] for et in ERROR_TYPES],
                       rotation=30, ha="right")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Per-Error-Type Failure Rate by Method")
    ax.legend(loc="upper left", ncol=min(n_methods, 3), framealpha=0.9)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    save_both_formats(fig, outpath)
    plt.close(fig)


# ── Figure 2: category breakdown ─────────────────────────────────────────────
def plot_category_breakdown(
    per_error_by_method: Dict[str, Dict[str, Dict]],
    method_order: List[str],
    palette: Dict[str, str],
    outpath: Path,
):
    """3 categories × N methods, y = mean error rate across types in category."""
    fig, ax = plt.subplots(figsize=(7, 4.0))

    n_methods = len(method_order)
    n_cats = len(CATEGORIES)
    bar_width = 0.8 / n_methods
    x = np.arange(n_cats)

    for m_idx, method in enumerate(method_order):
        cat_error_rates = []
        for cat in CATEGORIES:
            types_in_cat = [et for et, c in CATEGORY_MAP.items() if c == cat]
            rates = []
            for et in types_in_cat:
                entry = per_error_by_method[method].get(et, {})
                jsr = entry.get("judge_success_rate", 0.0)
                n = entry.get("n", 0)
                if n > 0:
                    rates.append((1 - jsr, n))
            if rates:
                total_n = sum(n for _, n in rates)
                weighted = sum(r * n for r, n in rates) / total_n
                cat_error_rates.append(100 * weighted)
            else:
                cat_error_rates.append(0.0)
        offset = (m_idx - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, cat_error_rates, bar_width,
            label=method, color=palette[method],
            edgecolor="black", linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Mean Error Rate (%)")
    ax.set_title("Error Category Breakdown\n(averaged across error types within category)")
    ax.legend(loc="upper right", ncol=1, framealpha=0.9)

    fig.tight_layout()
    save_both_formats(fig, outpath)
    plt.close(fig)


# ── Figure 3: judge score distribution box plots ─────────────────────────────
def plot_score_distribution(
    raw_results_by_method: Dict[str, List[Dict]],
    method_order: List[str],
    palette: Dict[str, str],
    outpath: Path,
):
    """One row per method, 9 boxes per row (one per error type)."""
    methods_with_data = [m for m in method_order if raw_results_by_method.get(m)]
    if not methods_with_data:
        print("  ⚠  No raw results available for distribution plot — skipping")
        return

    n_methods = len(methods_with_data)
    fig, axes = plt.subplots(n_methods, 1, figsize=(11, 2.2 * n_methods),
                              sharex=True, sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_with_data):
        results = raw_results_by_method[method]
        by_et = {et: [] for et in ERROR_TYPES}
        for r in results:
            et = r.get("error_type", "none")
            if et in by_et:
                by_et[et].append(r.get("judge_score", 0))
        data = [by_et[et] for et in ERROR_TYPES]

        bp = ax.boxplot(
            data, positions=np.arange(len(ERROR_TYPES)),
            widths=0.6, patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(palette[method])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.4)
            patch.set_alpha(0.75)

        ax.set_ylabel(f"{method}\nJudge score")
        ax.axhline(80, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                   label="Success threshold" if ax == axes[0] else None)
        ax.set_ylim(0, 105)

    axes[-1].set_xticks(np.arange(len(ERROR_TYPES)))
    axes[-1].set_xticklabels([ERROR_LABEL[et] for et in ERROR_TYPES],
                              rotation=30, ha="right")
    axes[0].legend(loc="lower left")
    axes[0].set_title("Judge Score Distribution per Error Type")

    fig.tight_layout()
    save_both_formats(fig, outpath)
    plt.close(fig)


# ── Figure 4: gain heatmap vs baseline ───────────────────────────────────────
def plot_gain_heatmap(
    per_error_by_method: Dict[str, Dict[str, Dict]],
    method_order: List[str],
    baseline_method: str,
    outpath: Path,
):
    """Heatmap: rows=methods (excl. baseline), cols=error types, cell=delta % pts."""
    other_methods = [m for m in method_order if m != baseline_method]
    if not other_methods:
        print("  ⚠  Need at least 2 methods for gain heatmap — skipping")
        return

    baseline_rates = {}
    for et in ERROR_TYPES:
        entry = per_error_by_method[baseline_method].get(et, {})
        baseline_rates[et] = entry.get("judge_success_rate", 0.0)

    gains = np.zeros((len(other_methods), len(ERROR_TYPES)))
    for i, m in enumerate(other_methods):
        for j, et in enumerate(ERROR_TYPES):
            entry = per_error_by_method[m].get(et, {})
            jsr = entry.get("judge_success_rate", 0.0)
            gains[i, j] = 100 * (jsr - baseline_rates[et])

    fig, ax = plt.subplots(figsize=(11, 0.7 + 0.5 * len(other_methods)))

    vmax = max(abs(gains.min()), abs(gains.max()), 1.0)
    im = ax.imshow(gains, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(ERROR_TYPES)))
    ax.set_xticklabels([ERROR_LABEL[et] for et in ERROR_TYPES],
                       rotation=30, ha="right")
    ax.set_yticks(np.arange(len(other_methods)))
    ax.set_yticklabels(other_methods)
    ax.set_title(f"Judge Success Rate Gain vs. {baseline_method} (percentage points)")

    # Annotate each cell
    for i in range(len(other_methods)):
        for j in range(len(ERROR_TYPES)):
            val = gains[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Δ judge success rate (pts)")
    fig.tight_layout()
    save_both_formats(fig, outpath)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", nargs="+", required=True,
                        help="Stats JSON files, one per method")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Method labels, same order as --stats")
    parser.add_argument("--results", nargs="*", default=None,
                        help="Raw results JSONs (for distribution plot); "
                             "same order and count as --stats if provided")
    parser.add_argument("--run", default="full",
                        choices=["full", "perfect_only"])
    parser.add_argument("--baseline", default="SFT",
                        help="Method label to use as baseline for gain heatmap")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    if len(args.stats) != len(args.labels):
        parser.error(f"--stats has {len(args.stats)} files but --labels has {len(args.labels)}")
    if args.results and len(args.results) != len(args.stats):
        parser.error(f"--results count must match --stats count")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    palette = build_palette(args.labels)

    # Load per-error-type data for each method
    per_error_by_method: Dict[str, Dict[str, Dict]] = {}
    for stats_path, label in zip(args.stats, args.labels):
        try:
            per_error_by_method[label] = load_per_error_from_stats(stats_path, args.run)
            print(f"  Loaded: {label} ← {stats_path}")
        except Exception as e:
            print(f"  ⚠  Skipping {label}: {e}")

    # Figure 1: per-error-type error rate
    plot_error_rate_by_type(
        per_error_by_method, args.labels, palette,
        outdir / "fig_error_rate_by_type.pdf",
    )

    # Figure 2: category breakdown
    plot_category_breakdown(
        per_error_by_method, args.labels, palette,
        outdir / "fig_category_breakdown.pdf",
    )

    # Figure 3: score distribution (needs raw results)
    if args.results:
        raw_by_method = {}
        for res_path, label in zip(args.results, args.labels):
            try:
                raw_by_method[label] = load_raw_results(res_path, args.run)
            except Exception as e:
                print(f"  ⚠  No raw results for {label}: {e}")
                raw_by_method[label] = []
        plot_score_distribution(
            raw_by_method, args.labels, palette,
            outdir / "fig_judge_score_distribution.pdf",
        )

    # Figure 4: gain heatmap
    if args.baseline in args.labels:
        plot_gain_heatmap(
            per_error_by_method, args.labels, args.baseline,
            outdir / "fig_pairwise_gain_heatmap.pdf",
        )
    else:
        print(f"  ⚠  --baseline '{args.baseline}' not in labels; skipping heatmap")


if __name__ == "__main__":
    main()