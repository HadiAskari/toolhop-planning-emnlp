#!/bin/bash
# Start the vLLM judge server BEFORE launching PPO training.
#
# Usage:
#   bash start_judge_server.sh [JUDGE_MODEL_PATH] [GPU_ID]
set -e

JUDGE_MODEL=${1:-"models/judge/merged"}
JUDGE_GPU=${2:-7}
PORT=8001
MODEL_NAME="judge"

echo "================================================"
echo "Starting Judge vLLM Server"
echo "  Model : $JUDGE_MODEL"
echo "  GPU   : $JUDGE_GPU"
echo "  Port  : $PORT"
echo "================================================"

if [ ! -d "$JUDGE_MODEL" ]; then
    echo "❌ Error: Judge model not found at $JUDGE_MODEL"; exit 1
fi
if [ ! -f "$JUDGE_MODEL/config.json" ]; then
    echo "❌ Error: No config.json in $JUDGE_MODEL — not a merged HF model."; exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "❌ No GPUs found via nvidia-smi"; exit 1
fi
MAX_GPU_IDX=$((GPU_COUNT - 1))
if [ "$JUDGE_GPU" -gt "$MAX_GPU_IDX" ]; then
    echo "⚠️  GPU $JUDGE_GPU does not exist (GPUs 0-$MAX_GPU_IDX available), using $MAX_GPU_IDX"
    JUDGE_GPU=$MAX_GPU_IDX
fi
echo "Using GPU $JUDGE_GPU of $GPU_COUNT available"

# Disable torch.compile/inductor — torch/vLLM version mismatch
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
# Force vLLM v0 engine — v1 has a warmup bug (RuntimeError: Could not infer dtype of numpy.int64)
export VLLM_USE_V1=0
# Explicitly set which GPU this process sees (CUDA_VISIBLE_DEVICES=N makes it "GPU 0" inside)
export CUDA_VISIBLE_DEVICES=$JUDGE_GPU

python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$MODEL_NAME" \
    --port $PORT \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    2>&1 | tee judge_server.log &

JUDGE_PID=$!
echo "Judge server PID: $JUDGE_PID"
echo $JUDGE_PID > judge_server.pid

echo "Waiting for judge server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Judge server is ready (took ${i}s)"
        break
    fi
    if ! kill -0 $JUDGE_PID 2>/dev/null; then
        echo "❌ Judge server process died. Check judge_server.log:"
        tail -30 judge_server.log
        exit 1
    fi
    if [ $i -eq 120 ]; then
        echo "❌ Timed out. Check judge_server.log:"; tail -30 judge_server.log; exit 1
    fi
    sleep 1
done

echo ""
echo "Judge server running on port $PORT (GPU $JUDGE_GPU)"
echo "To stop: kill \$(cat judge_server.pid)"
echo "================================================"