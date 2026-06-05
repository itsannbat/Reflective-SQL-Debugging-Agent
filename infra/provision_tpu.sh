#!/usr/bin/env bash
# Create the v5litepod-4 TPU VM for vLLM serving.
# Idempotent: if the VM already exists, prints state and exits 0.
set -euo pipefail

PROJECT="${PROJECT:-x-object-492801-h3}"
ZONE="${ZONE:-us-south1-a}"
TPU_NAME="${TPU_NAME:-mlsystems-vllm}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-4}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"

existing="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project="$PROJECT" --zone="$ZONE" --format='value(state)' 2>/dev/null || true)"

if [[ -n "$existing" ]]; then
  echo "TPU VM '$TPU_NAME' already exists in $ZONE (state: $existing). Skipping create."
  exit 0
fi

echo "Creating TPU VM '$TPU_NAME' ($ACCELERATOR_TYPE) in $ZONE..."
echo "WARNING: v5litepod-4 costs ~\$115/day while RUNNING (~\$4.80/hr). Stop it when idle."
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --version="$RUNTIME_VERSION"

echo "Done. Wait ~1 min then run setup_vm.sh"
