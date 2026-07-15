#!/bin/bash
#
# Train Planner LLM with SFT using verl
#
# This script fine-tunes a model to generate tool execution plans.
#

set -x

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
export TMPDIR=/workspace/tmp
export RAY_TMPDIR=/workspace/ray
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

mkdir -p $TMPDIR $RAY_TMPDIR $XDG_CACHE_HOME 2>/dev/null || true

# Data paths (UPDATE THESE!)
TRAIN_DATA="${FORTE_ROOT}/planner_finetuning/data/planner_sft/train_sft.parquet"
VAL_DATA="${FORTE_ROOT}/planner_finetuning/data/planner_sft/val_sft.parquet"

# Model
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"  # or Qwen2.5-3B-Instruct for faster training

# Output
CHECKPOINT_DIR="${FORTE_ROOT}/verl-workspace/verl/toolhop-planning/planner_finetuning/checkpoints_planner_sft"
PROJECT_NAME="toolhop_planner_sft"
EXPERIMENT_NAME="planner_qwen_0.5b_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
NUM_GPUS=4
MICRO_BATCH_SIZE=2  # Smaller for 7B model
TOTAL_EPOCHS=4

# Gradient accumulation for effective batch size
# Effective batch size = NUM_GPUS * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS=4  # Effective batch size = 8 * 2 * 4 = 64

# Learning rate
LEARNING_RATE=2e-5  # Standard for SFT

# Prompt/response keys
PROMPT_KEY="prompt_flat"  # or "prompt" for chat template
RESPONSE_KEY="response"

# Logging
LOGGER='["console","wandb"]'  # or just '["console"]' if no wandb

# ============================================================================
# VALIDATE DATA EXISTS
# ============================================================================

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Training data not found: $TRAIN_DATA"
    echo "Run create_planner_sft_dataset.py first!"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "❌ Validation data not found: $VAL_DATA"
    echo "Run create_planner_sft_dataset.py first!"
    exit 1
fi

echo "✓ Training data: $TRAIN_DATA"
echo "✓ Validation data: $VAL_DATA"

# ============================================================================
# RUN SFT TRAINING
# ============================================================================

echo ""
echo "=================================="
echo "STARTING PLANNER SFT TRAINING"
echo "=================================="
echo ""
echo "Model: $BASE_MODEL"
echo "Epochs: $TOTAL_EPOCHS"
echo "Batch size: $MICRO_BATCH_SIZE per GPU"
echo "Grad accum: $GRAD_ACCUM_STEPS steps"
echo "Learning rate: $LEARNING_RATE"
echo "Output: $CHECKPOINT_DIR"
echo ""

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.prompt_key=$PROMPT_KEY \
    data.response_key=$RESPONSE_KEY \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.train_batch_size=$((NUM_GPUS * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS)) \
    data.max_length=2048 \
    model.partial_pretrain=$BASE_MODEL \
    model.trust_remote_code=true \
    model.strategy=fsdp2 \
    model.enable_gradient_checkpointing=true \
    optim.lr=$LEARNING_RATE \
    optim.lr_scheduler=cosine \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    trainer.logger=$LOGGER \
    trainer.resume_mode=auto

# ============================================================================
# POST-TRAINING
# ============================================================================

echo ""
echo "=================================="
echo "TRAINING COMPLETE"
echo "=================================="
echo ""
echo "Model saved to: $CHECKPOINT_DIR"
echo ""
echo "Next steps:"
echo "  1. Test the model: python test_planner_sft.py --model $CHECKPOINT_DIR/global_step_XXX"
echo "  2. Run inference: python planner_inference.py"
echo "  3. Start RL training with this as base model"
echo ""
