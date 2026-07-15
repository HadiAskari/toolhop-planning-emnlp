#!/usr/bin/env python3
"""
Batched Transformers judge server — OpenAI-compatible /v1/chat/completions endpoint.

Uses a dynamic batching queue: incoming requests are held briefly (up to
BATCH_TIMEOUT_MS ms or BATCH_MAX_SIZE requests) then processed together in a
single GPU forward pass. This gives 8-16x throughput vs one-at-a-time Flask.

Usage:
    CUDA_VISIBLE_DEVICES=0 python judge_server.py \
        --model ${FORTE_ROOT}/judge_finetuning/models/judge-new/merged \
        --port 8001 \
        --batch-size 64 \
        --batch-timeout-ms 200
     
    #NESTFUL   
    CUDA_VISIBLE_DEVICES=1 python judge_server.py \
        --model ${FORTE_ROOT}/judge_finetuning/models/judge-nestful/merged \
        --port 8002 \
        --batch-size 64 \
        --batch-timeout-ms 200
        
    # Wait for it
    sleep 15
    curl -s http://localhost:8001/health
"""

import argparse
import json
import time
import threading
import queue
import uuid
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# ── Globals set at startup ───────────────────────────────────────────────────
_model       = None
_tokenizer   = None
_batch_size  = 16       # max requests per batch
_timeout_ms  = 50       # ms to wait for a full batch before flushing

# ── Batching queue ───────────────────────────────────────────────────────────
# Each item: {"id": str, "prompt": str, "max_tokens": int, "temperature": float,
#             "result_queue": queue.Queue}
_request_queue = queue.Queue()


def load_model(model_path: str):
    global _model, _tokenizer
    print(f"Loading judge model from {model_path}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"   # required for batched generation

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    device = next(_model.parameters()).device
    print(f"✅ Judge model loaded on {device}")


def _process_batch(batch: list):
    """Run one batched forward pass and post results back to each caller."""
    prompts     = [item["prompt"]      for item in batch]
    max_tokens  = max(item["max_tokens"]  for item in batch)
    temperature = batch[0]["temperature"]   # all requests use same temp (0.0)

    inputs = _tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(next(_model.parameters()).device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    for i, item in enumerate(batch):
        generated_ids = outputs[i][input_len:]
        content = _tokenizer.decode(generated_ids, skip_special_tokens=True)
        item["result_queue"].put({
            "content":           content,
            "prompt_tokens":     input_len,
            "completion_tokens": len(generated_ids),
        })


def batch_worker():
    """Background thread: drain the request queue in batches."""
    while True:
        batch = []

        # Block until at least one request arrives
        try:
            item = _request_queue.get(timeout=1.0)
            batch.append(item)
        except queue.Empty:
            continue

        # Collect more requests up to _batch_size within _timeout_ms
        deadline = time.monotonic() + _timeout_ms / 1000.0
        while len(batch) < _batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = _request_queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break

        try:
            _process_batch(batch)
        except Exception as e:
            print(f"[judge_server] BATCH WORKER ERROR (batch_size={len(batch)}): {e}", flush=True)
            for item in batch:
                item["result_queue"].put({"error": str(e)})


# ── Flask endpoints ──────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "queue_depth": _request_queue.qsize(),
        "batch_size":  _batch_size,
        "timeout_ms":  _timeout_ms,
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data        = request.get_json()
    messages    = data.get("messages", [])
    temperature = float(data.get("temperature", 0.0))
    max_tokens  = int(data.get("max_tokens", 1024))

    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    result_q = queue.Queue()
    _request_queue.put({
        "id":           str(uuid.uuid4()),
        "prompt":       prompt,
        "max_tokens":   max_tokens,
        "temperature":  temperature,
        "result_queue": result_q,
    })

    # Block until the batch worker posts the result (120s timeout)
    result = result_q.get(timeout=120)

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({
        "id":      f"chatcmpl-{int(time.time())}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "judge",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": result["content"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens":      result["prompt_tokens"] + result["completion_tokens"],
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            required=True, help="Path to merged judge model")
    parser.add_argument("--port",             type=int, default=8001)
    parser.add_argument("--host",             default="0.0.0.0")
    parser.add_argument("--batch-size",       type=int, default=16,
                        help="Max requests per GPU batch (default 16)")
    parser.add_argument("--batch-timeout-ms", type=int, default=50,
                        help="Max ms to wait for a full batch before flushing (default 50)")
    args = parser.parse_args()

    _batch_size = args.batch_size
    _timeout_ms = args.batch_timeout_ms

    load_model(args.model)
    
    # Add after load_model(args.model) in __main__:
    print("Warming up with dummy inference...")
    _dummy = _tokenizer("test", return_tensors="pt").to(next(_model.parameters()).device)
    with torch.no_grad():
        _model.generate(**_dummy, max_new_tokens=1)
    print("✓ Warmup done")

    
    def _worker_with_restart():
        while True:
            try:
                batch_worker()
            except Exception as e:
                print(f"[judge_server] Worker crashed: {e} — restarting in 2s", flush=True)
                import time as _t; _t.sleep(2.0)

    worker_thread = threading.Thread(target=_worker_with_restart, daemon=True)
    worker_thread.start()
    # Start background batching thread
    # worker_thread = threading.Thread(target=batch_worker, daemon=True)
    # worker_thread.start()
    print(f"✅ Batch worker started (max_batch={_batch_size}, timeout={_timeout_ms}ms)")

    print(f"Starting judge server on {args.host}:{args.port}")
    # threaded=True so Flask handles concurrent HTTP requests while batch worker runs
    app.run(host=args.host, port=args.port, threaded=True, debug=False)