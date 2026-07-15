#!/bin/bash
# E1 — Execution-based end-task accuracy. NO GPU REQUIRED.
#
# Step 0 (once): pip install python-dateutil pytz dicttoxml babel holidays roman numpy sympy
#
# Step 1 — validate the harness on this machine (expected: ToolHop grounded
#          ~99%, NESTFUL ~100% on the covered subset):
#   python run_execution_eval.py --dataset toolhop --gold-check
#   python run_execution_eval.py --dataset nestful --gold-check
#
# Step 2 — execute every method's saved results. Edit the paths below to the
#          actual result files on the server, then run this script.
#          Works on Best-of-N outputs (FORTE), baseline outputs
#          (ReAct/ToolPlanner/α-UMi/GNN4TaskPlan), GPT baselines, and the E3
#          matched-budget GPT output. Each run writes <results>.execution.json.

set -e
cd "$(dirname "$0")"

ROOT=${FORTE_ROOT:-${FORTE_ROOT}}

# ── ToolHop (edit paths; add one line per method/backbone) ───────────────────
TOOLHOP_RESULTS=(
  # "$ROOT/verl-integration-of-trained-planner-and-judge/results/<forte_bon_results>.json"
  # "$ROOT/baselines/tool_planner_Qwen2.5-7B-Instruct_results.json"
  # "$ROOT/baselines/react_few_shot_Qwen2.5-7B-Instruct_results.json"
  # "$ROOT/<gpt_baseline_results>.json"
)
for f in "${TOOLHOP_RESULTS[@]}"; do
  echo "=== E1 toolhop: $f"
  python run_execution_eval.py --dataset toolhop --results "$f" --run full
done

# ── NESTFUL ──────────────────────────────────────────────────────────────────
NESTFUL_RESULTS=(
  # "$ROOT/<nestful forte bon results>.json"
  # "$ROOT/baselines_nestful/qwen7b/tool_planner_...json"
)
for f in "${NESTFUL_RESULTS[@]}"; do
  echo "=== E1 nestful: $f"
  python run_execution_eval.py --dataset nestful --results "$f" --run full
done

echo "Done. Aggregate with: python ../aggregate_results.py"
