#!/usr/bin/env bash
# Stop the TPU VM to stop billing. Disk and config are preserved.
set -euo pipefail

PROJECT="${PROJECT:-x-object-492801-h3}"
ZONE="${ZONE:-us-south1-a}"
TPU_NAME="${TPU_NAME:-mlsystems-vllm}"

gcloud compute tpus tpu-vm stop "$TPU_NAME" \
  --project="$PROJECT" --zone="$ZONE"
echo "Stopped $TPU_NAME. To resume: gcloud compute tpus tpu-vm start $TPU_NAME --zone=$ZONE --project=$PROJECT"
