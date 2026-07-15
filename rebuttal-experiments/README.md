# FORTE ARR Rebuttal Experiments (E1–E6)

Code for the six rebuttal experiments for ACL ARR May 2026 submission #15152.
Every metric/parser/judge-prompt is **vendored byte-for-byte** from the code
that produced the paper's numbers (`common/forte_common.py` documents the
sources), so all new numbers are directly comparable to the paper's.

| Exp | Answers | Needs | Time |
|-----|---------|-------|------|
| E1 execution accuracy | kocM C1, VJ8B W1/W2, fiFx W3c — **highest priority** | CPU only + method result JSONs | minutes |
| E2 compound errors | fiFx W2 (synthetic errors too simple) | judge server (1 GPU) | ~10 min/dataset |
| E3 matched-budget GPT | VJ8B W4 (unfair closed-source comparison) | judge server + `OPENAI_API_KEY` | ~30–60 min, ~$5–15 |
| E4 judge transfer | VJ8B W3 (dataset-specificity) | judge server ×2 judges | ~30 min total |
| E5 organic errors | fiFx W2 (real vs injected failures) | CPU only (aggregate mode) | seconds |
| E6 human agreement | fiFx W3 (no human annotations) | two humans, ~1–2 h each | — |

All scripts accept `--limit N` for a quick smoke test and `--repo-root` /
`FORTE_ROOT` env var (default fallback: `${FORTE_ROOT}`).
Paths are resolved for both the local-git layout (`scripts/...`) and the lab
server layout automatically.

## Setup (lab server, conda env `verl`)

```bash
pip install python-dateutil pytz dicttoxml babel holidays roman sympy   # E1 real-mode ToolHop tool code
export FORTE_ROOT=${FORTE_ROOT}
export OPENAI_API_KEY=...   # E3 only
```

Judge server (E2/E3/E4; also E5 `--rescore`): use the existing launcher —

```bash
bash $FORTE_ROOT/planner_rl/start_judge_server.sh \
     $FORTE_ROOT/judge_finetuning/models/judge/merged  <GPU_ID>          # ToolHop judge
# NESTFUL judge: .../models/judge-nestful/merged
```

---

## E1 — Execution-based end-task accuracy (run this first)

**Protocols** (validated locally; see "Validation results" below):

* **NESTFUL — real execution.** Plans run against IBM's official executable
  math functions (vendored in `e1_execution/nestful_exec/basic_functions.py`).
  Covers the MathQA subset (141/180 test queries; the coding subset can be
  added with `--nestful-exec-dir <clone of IBM/NESTFUL>/data_v2/executable_functions`).
* **ToolHop — grounded execution (default).** ToolHop's per-query tools are
  GPT-written simulators with exact-string lookup tables; even gold plans
  execute verbatim against them only ~8% of the time (the original benchmark
  is interactive, with retries). Grounded mode removes only that brittle
  string-lookup layer: each tool maps to its benchmark-provided sub-question
  hop; a call succeeds **only if its resolved arguments carry an upstream
  hop's gold sub-answer or a question entity** (so broken dependency
  threading, wrong hardcoded values, mis-ordering, skipped hops, forward /
  circular references, and incomplete plans all fail), and returns the gold
  sub-answer. `--toolhop-mode real` runs the shipped tool code instead.

```bash
cd rebuttal-experiments/e1_execution

# 1. harness self-validation (expected: ~99% toolhop grounded, 100% nestful)
python run_execution_eval.py --dataset toolhop --gold-check
python run_execution_eval.py --dataset nestful --gold-check

# 2. execute every method's saved plans (edit run_e1.sh paths, or one-by-one):
python run_execution_eval.py --dataset toolhop --run full \
    --results $FORTE_ROOT/<path-to-forte-bon-results>.json
python run_execution_eval.py --dataset toolhop --run full \
    --results $FORTE_ROOT/baselines/tool_planner_Qwen2.5-7B-Instruct_results.json
# ... repeat for ReAct / α-UMi / GNN4TaskPlan / GPT baseline / E3 output,
# and the NESTFUL equivalents with --dataset nestful
```

Each run writes `<results>.execution.json` with per-record execution status +
aggregate `end_task_accuracy_strict` (the headline number), and
`exec_accuracy_given_judge_pass` (VJ8B W2's "how many judge-passing plans
actually succeed").

### Validation results (already run locally on the held-out test split)

| check | result |
|---|---|
| NESTFUL gold plans, real execution | **100%** answer-correct (141/141 covered) |
| ToolHop gold plans, grounded | **99%** (1 genuinely flawed gold plan, correctly caught) |
| ToolHop gold plans, real tool code | 8% (motivates grounded mode) |
| ToolHop seeded negatives passing grounded execution | circular/forward/incomplete/order: **0%**; missing-dep 19%; wrong-tool 53%; typo/type-mismatch mostly pass (value-level errors are invisible to grounded mode — real execution on NESTFUL covers those: typo 5%, type-mismatch 0% pass) |
| NESTFUL seeded negatives passing real execution | 14% overall; the passes are semantically benign cases (e.g. duplicated step still yields the right value) or corpus no-op negatives |

## E2 — Compound-error judge stress test

```bash
cd rebuttal-experiments/e2_compound_errors
python run_compound_stress_test.py --dataset toolhop   # judge server must be up
python run_compound_stress_test.py --dataset nestful
# --dry-run generates the compound plans without scoring (no GPU; already tested)
```

Composes the paper's own injectors (2 and 3 simultaneous errors, seeded,
change-verified — no-op injections are re-drawn) on held-out gold plans, then
scores gold / stored single-error / compound negatives with the frozen judge.
Reports detection rate (<80) and separation from gold.

## E3 — Matched-budget GPT-5.5 Best-of-5

```bash
cd rebuttal-experiments/e3_matched_budget_gpt
python evaluate_gpt_best_of_n.py --model gpt-5.5 --perfect-only \
    --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_full_clean/test.parquet \
    --api-key \
    --output e3_gpt_bon_toolhop.json
python evaluate_gpt_best_of_n.py --model gpt-5.5 --perfect-only \
    --test-parquet ${FORTE_ROOT}/planner_rl/data/verl_rl_nestful_full/test.parquet \
     --api-key \
    --output e3_gpt_bon_nestful.json
```

`--perfect-only` = the same 100 gold-reference rows as the paper's Table 3.
Identical protocol to FORTE's BoN: 5 samples at τ∈{0.2,…,1.0}, judge-scored,
judge-selected. If the API rejects `temperature`, the run records
`temperature_supported: false` — state that honestly in the rebuttal. Output
records use the BoN schema, so E1 can execute them
(`--results e3_gpt_bon_toolhop.json`).

## E4 — Cross-dataset judge transfer

```bash
cd rebuttal-experiments/e4_judge_transfer
# with ToolHop judge served:
python run_judge_transfer.py --dataset toolhop --judge-label judge-toolhop
python run_judge_transfer.py --dataset nestful --judge-label judge-toolhop
# restart server with judge-nestful/merged, then:
python run_judge_transfer.py --dataset nestful --judge-label judge-nestful
python run_judge_transfer.py --dataset toolhop --judge-label judge-nestful
```

Reports gold-vs-negative AUC, separation, detection per error type. The
aggregate pairs in-domain vs cross-domain.

## E5 — Judge on organic failures

```bash
cd rebuttal-experiments/e5_organic_errors
python run_organic_analysis.py \
    --results "$FORTE_ROOT/baselines/*_results.json" \
              "$FORTE_ROOT/baselines_nestful/**/*_results.json" \
    --gold-scores ../e2_compound_errors/e2_compound_toolhop.json
# optional, needs judge server: --rescore  (collects predicted error types)
# optional, after E1: --use-execution (failure = execution failure)
```

No GPU in default mode — it aggregates the judge scores already stored in the
baseline result files over records whose plans fail functional match.

## E6 — Human agreement

Sheets are already generated in `e6_human_agreement/annotation_sheets/`
(50 ToolHop + 50 NESTFUL items, stratified over gold + all 9 error types,
blinded). Two authors each fill `annotation_sheet_<ds>_annotator{A,B}.csv`
per `INSTRUCTIONS.txt` (~1–2 h), then:

```bash
python compute_agreement.py --dataset toolhop nestful
```

## Aggregate everything

```bash
python rebuttal-experiments/aggregate_results.py
```

Prints every number keyed to the rebuttal draft's placeholders.
