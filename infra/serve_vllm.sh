#!/usr/bin/env bash
# Launch vLLM in the official vllm-tpu Docker container.
# Image entrypoint uses VLLM_ARGS env var convention (Google's vllm-tpu image).
# Run inside tmux so it survives SSH disconnects:
#   tmux new -s vllm
#   bash ~/serve_vllm.sh
#   (Ctrl-b d to detach)
set -euo pipefail

MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# Toggleable knobs (set to "" to disable). Default: both ON.
PREFIX_CACHING_FLAG="${PREFIX_CACHING_FLAG:---enable-prefix-caching}"
CHUNKED_PREFILL_FLAG="${CHUNKED_PREFILL_FLAG:---enable-chunked-prefill}"

# Assemble VLLM_ARGS string for the container entrypoint.
VLLM_ARGS="--model $MODEL"
VLLM_ARGS+=" --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
VLLM_ARGS+=" --max-model-len $MAX_MODEL_LEN"
VLLM_ARGS+=" --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"
VLLM_ARGS+=" $PREFIX_CACHING_FLAG $CHUNKED_PREFILL_FLAG"
VLLM_ARGS+=" --host 0.0.0.0 --port $VLLM_PORT"
VLLM_ARGS+=" --no-enable-log-requests"

# Stop any previous container.
sudo docker rm -f vllm 2>/dev/null || true

echo "Launching vLLM in Docker:"
echo "  VLLM_ARGS=$VLLM_ARGS"

# --privileged: TPU device access
# --net host:   simpler port exposure
# --shm-size:   ray/torch IPC needs >>default 64MB
# -v hf cache:  reuse host-side model cache (avoids redownloading 14 GB)
exec sudo docker run --name vllm \
  --privileged \
  --net host \
  --shm-size=16g \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_ARGS="$VLLM_ARGS" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  vllm/vllm-tpu:nightly
