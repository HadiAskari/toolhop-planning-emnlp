# FORTE — Failure-taxonomy Reinforcement for Tool-planning

Anonymized artifact repository accompanying the paper submission
*"FORTE: Robust Multi-Step Tool Planning by Failure-Avoidance Training."*

This repository contains the augmented contrastive corpora, the error-injection
pipeline, the judge/planner training and evaluation code, and every script used
to produce the numbers reported in the paper and the response period.

> **Anonymity note.** All absolute paths have been replaced with the
> `${FORTE_ROOT}` placeholder, no API keys are stored in any file, and the git
> history has been reset. Model checkpoints are hosted separately (see below).

---

## Quick start

```bash
# 1. Point FORTE_ROOT at this checkout (all scripts resolve data relative to it)
export FORTE_ROOT=$(pwd)

# 2. Restore shipped data (decompress *.gz, check external benchmark)
bash prepare_data.sh

# 3. Environment (see "Installation" for the vLLM / verl caveat)
pip install -r requirements.txt      # or: conda env create -f environment.yml
```

## Installation

`requirements.txt` lists the core dependencies with the versions used for the
reported runs. Two components are installed separately from source because the
CUDA build must match your machine:

- **verl** — the GRPO training framework (`scripts/planner_rl/`).
- **vLLM** — serves the judge and planner (`scripts/planner_rl/judge_server.py`).

Everything except planner **training** (`planner_rl`, `planner_finetuning`) and
the vLLM judge server runs CPU-only.

---

## Repository map

| Artifact (as promised in the response) | Location |
|---|---|
| **Augmented contrastive corpora** (9,950 ToolHop · 17,880 NESTFUL rubric-annotated plans; per-error type, severity, step localization, point deductions) | `scripts/toolhop_annotated_v1_remapped.json.gz`, `scripts/NESTFUL/data/nestful_annotated_combined.json.gz` |
| **Error-injection pipeline** (nine taxonomy injectors) | `toolhop_plan_generator.py`, `nestful_annotator.py` |
| ↳ compound-error composition (rebuttal stress test) | `rebuttal-experiments/e2_compound_errors/` |
| **Annotation code** (GPT rubric annotation; prompts in the paper appendix) | `nestful_annotator.py` (`LLMJudgeAnnotator`) |
| **Judge fine-tuning** (LoRA on Qwen2.5-7B-Instruct) | `scripts/judge_finetuning/` |
| **Planner SFT + GRPO training** | `scripts/planner_finetuning/`, `scripts/planner_rl/` |
| **Structural metrics** (JSR / FM / PA / DA) | `rebuttal-experiments/common/forte_common.py` |
| **Judge-independent execution harness** + validation (gold-plan ceiling, seeded-negative checks) | `rebuttal-experiments/e1_execution/` |
| **Baselines** (ReAct, α-UMi, GNN4TaskPlan, Tool-Planner, LATS) | `scripts/baselines/`, `scripts/baselines_nestful/` |
| **Best-of-N inference** with the trained judge | `scripts/verl-integration-of-trained-planner-and-judge/` |

### Response-period experiments (`rebuttal-experiments/`)

| Exp | What | Directory |
|---|---|---|
| E1 | Judge-independent execution accuracy | `e1_execution/` |
| E2 | Compound (2- & 3-error) judge stress test | `e2_compound_errors/` |
| E3 | Matched-budget GPT Best-of-5 comparison | `e3_matched_budget_gpt/` |
| E4 | Cross-benchmark judge transfer | `e4_judge_transfer/` |
| E5 | Organic (baseline-produced) failure analysis | `e5_organic_errors/` |
| E6 | Human–rubric agreement study | `e6_human_agreement/` |

See `rebuttal-experiments/README.md` for per-experiment run commands.

---

## Data & checkpoints

**Included** (decompressed by `prepare_data.sh`):
- The two rubric-annotated contrastive corpora (`*_annotated_*.json.gz`).
- NESTFUL benchmark inputs (`nestful_data.jsonl.gz`).
- All baseline, judge-transfer, and rebuttal-experiment result JSONs, so every
  reported number is reproducible from the shipped outputs.

**Not redistributed here:**
- **ToolHop.json** — third-party benchmark; download from the original ToolHop
  release and place at `./ToolHop.json` and `./scripts/ToolHop.json`
  (`prepare_data.sh` checks for it).
- **Trained checkpoints** — the judge LoRA adapters (Qwen2.5-7B-Instruct) and the
  six planner checkpoints are multi-GB and exceed the anonymous-host file limit.
  They will be released on a public model host at camera-ready. In the code they
  are referenced as `${FORTE_ROOT}/planner_finetuning/checkpoints_*/...` and
  `${FORTE_ROOT}/judge_finetuning/models/...`; set those paths to your local
  checkpoint copies (or retrain with the provided pipeline).

## Path convention

No absolute paths appear in any script. Everything resolves against the
`FORTE_ROOT` environment variable (default: the repository root). Result JSONs
retain checkpoint identifiers as `${FORTE_ROOT}/.../checkpoints_.../global_step_N`
so the exact model behind each number is still recorded, without machine paths.

## License

Code: MIT. Data (annotated corpora): CC-BY. The ToolHop and NESTFUL benchmarks
retain their original licenses.
