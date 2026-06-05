# Cost Discipline — read this before touching the TPU

## The math

| State | Spend rate |
|-------|------------|
| **RUNNING** (v5litepod-4) | ~$4.80/hr ≈ **$115/day** |
| **STOPPED** | $0/hr (disk storage is a few cents/day, negligible) |
| **DELETED** | $0 |

You only pay while the VM is in state `READY`. **Stopping the VM is the safety switch**, not deleting it — stopped VMs keep their disk, model weights, postgres data, venv, everything. Start ≈ 3 min to be back at READY with the model cached locally.

## Project budget (back-of-envelope)

Mini-project deadline: **2026-06-11** (6 days from 2026-06-05).

| Discipline level | Hours running / day | Cost over 6 days |
|------------------|---------------------|------------------|
| Always-on | 24 | ~$690 |
| Active dev only | 6 | ~$170 |
| **Bursts (recommended)** | 2-3 | **~$60-85** |
| Smoke-test only (today) | 0.5 | ~$2.50 |

Goal: stay in the bottom two rows.

## The cardinal rules

1. **Default state is STOPPED.** If you're not actively running a benchmark or developing against the endpoint, the VM should be stopped.
2. **Stop before walking away from your laptop.** Don't trust yourself to come back and stop it.
3. **Check at end of each session.** Add this to your terminal close ritual:
   ```bash
   gcloud compute tpus tpu-vm list --project=x-object-492801-h3 --zone=us-south1-a
   ```
   If it says `READY` and you're done → stop it.
4. **Coordinate with teammates.** Don't have two people leave the VM running thinking the other person is using it. Pin the "VM status" in your team chat.

## Standard session workflow

```powershell
# --- START OF SESSION ---
# 1. Start the VM (3 min to READY)
gcloud compute tpus tpu-vm start mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3

# 2. Reattach to the vLLM tmux session (server auto-restarts inside tmux)
gcloud compute tpus tpu-vm ssh mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 `
  --command="tmux attach -t vllm || tmux new -d -s vllm 'bash ~/serve_vllm.sh 2>&1 | tee -a ~/vllm.log'"

# 3. Open SSH tunnel in a separate terminal — leave running
gcloud compute tpus tpu-vm ssh mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 -- -L 8000:localhost:8000 -N

# ...do work, run benchmarks, etc...

# --- END OF SESSION (CRITICAL) ---
bash mini-project-infra/stop_tpu.sh

# Verify it actually stopped
gcloud compute tpus tpu-vm describe mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 --format="value(state)"
# Should print STOPPED, not READY
```

## What if I forget?

Worst case: VM runs overnight. ~$115. Annoying, not catastrophic, but avoidable.

To make forgetting harder, you can:
- Set a daily calendar reminder at end of work hours: "Stop the TPU"
- Set a GCP budget alert: Console → Billing → Budgets & Alerts → create budget at e.g. $50/week with 50%/90%/100% email triggers

## Pre-download the model during setup

The `setup_vm.sh` script pre-downloads Mistral-7B-Instruct-v0.3 (~14 GB) to the VM's local HF cache. This means:
- First `vllm serve` after fresh install: ~3 min to ready (model load + compile, no download)
- Every subsequent restart: ~3 min to ready (model in cache, compile cached)
- Without pre-download, first serve would take ~10 min (download + load + compile)

This is the single biggest "minimum-time" optimization in the workflow.

## What got built today

To minimize spend during initial bring-up:
1. Provisioned v5litepod-4
2. Ran setup (apt + venv + vllm install + model pre-download + postgres)
3. Smoke-tested the endpoint
4. **Immediately stopped the VM** once smoke test passed

Total "running time" for the initial setup: ~25 min ≈ $2.
