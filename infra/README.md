# infra — vLLM on Cloud TPU (Docker)

Operator runbook for Person 1. Serves **NousResearch/Meta-Llama-3.1-8B-Instruct** on a v5litepod-4 via the official `vllm/vllm-tpu:nightly` Docker image. Exposes an OpenAI-compatible API and Prometheus `/metrics`.

| Field | Value |
|-------|-------|
| Project | `x-object-492801-h3` |
| Zone | `us-south1-a` |
| TPU name | `mlsystems-vllm` |
| Accelerator | `v5litepod-4` (4 chips, 64 GB HBM) |
| Model | `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated) |
| Container | `vllm/vllm-tpu:nightly` |
| Endpoint | `http://localhost:8000/v1` (via SSH tunnel) |
| Cost | **~$115/day while RUNNING** ($4.80/hr) — see [COST_DISCIPLINE.md](COST_DISCIPLINE.md) |

**Why Docker, not pip?** vLLM-on-TPU requires a tight pinning of torch, torch_xla, libtpu, and jax. `pip install vllm` on a TPU VM hits a cascade of version conflicts (segfaults, missing ops, API drift). The Docker image ships a tested combo of all of these, so the host VM only needs Docker + Postgres.

**Model choice:** `NousResearch/Meta-Llama-3.1-8B-Instruct` — the proposal's Llama-3-8B target via Nous's ungated mirror (no Meta HF token needed). vLLM-TPU has a well-tested JAX-native `LlamaForCausalLM` path. We tried Mistral first (its `MistralForCausalLM` isn't JAX-native and the PyTorch fallback segfaults) and Qwen2.5 (vllm-tpu's loader assumes a multimodal config with `text_config` that Qwen2's text-only config doesn't have).

## First-time bring-up

```powershell
# 1. Provision the TPU VM (creates if absent; idempotent otherwise)
bash infra/provision_tpu.sh

# 2. SCP setup + serve + smoke scripts to the VM
gcloud compute tpus tpu-vm scp infra/setup_vm_docker.sh infra/serve_vllm.sh infra/smoke_test.py `
  mlsystems-vllm:/home/jopin/ --zone=us-south1-a --project=x-object-492801-h3

# 3. Install Docker + Postgres + pull the vllm-tpu image (~5 min)
gcloud compute tpus tpu-vm ssh mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 `
  --command="bash /home/jopin/setup_vm_docker.sh"

# 4. Launch vLLM in tmux on the VM
gcloud compute tpus tpu-vm ssh mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 `
  --command="tmux new -d -s vllm 'bash /home/jopin/serve_vllm.sh 2>&1 | tee /home/jopin/vllm.log'"

# 5. Open SSH tunnel from your laptop (leave running in its own terminal)
gcloud compute tpus tpu-vm ssh mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 `
  -- -L 8000:localhost:8000 -N

# 6. Smoke test (after vllm.log shows "Application startup complete")
python infra/smoke_test.py
```

## Daily workflow

```powershell
# Start the VM for the day
gcloud compute tpus tpu-vm start mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3

# Relaunch the container in tmux (idempotent: kills old + starts fresh)
gcloud compute tpus tpu-vm ssh mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 `
  --command="tmux kill-session -t vllm 2>/dev/null; tmux new -d -s vllm 'bash /home/jopin/serve_vllm.sh 2>&1 | tee /home/jopin/vllm.log'"

# Watch boot progress (Ctrl-C to detach when "Application startup complete")
gcloud compute tpus tpu-vm ssh mlsystems-vllm --zone=us-south1-a --project=x-object-492801-h3 `
  --command="tail -f /home/jopin/vllm.log"

# When done for the day — STOP to halt billing
bash infra/stop_tpu.sh
```

## Toggling the experiment knobs

Four presets live in `configs/` for the prefix-cache × chunked-prefill A/B:

| Preset | Prefix cache | Chunked prefill |
|--------|--------------|-----------------|
| `prefix_on_chunked_on.env` | ON | ON |
| `prefix_on_chunked_off.env` | ON | OFF |
| `prefix_off_chunked_on.env` | OFF | ON |
| `prefix_off_chunked_off.env` | OFF | OFF |

To switch presets, source the file before relaunching `serve_vllm.sh`:

```bash
# On the VM:
set -a; source ~/configs/prefix_off_chunked_off.env; set +a
bash ~/serve_vllm.sh
```

## For Person 2 (agent loop)

- Endpoint URL: `http://localhost:8000/v1` (over SSH tunnel)
- Model string: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Max context: 8192 tokens
- OpenAI-compatible — use the `openai` Python SDK with `base_url="http://localhost:8000/v1"` and any non-empty `api_key`.
- Postgres on the same VM: `postgresql://sqlagent@localhost/spider_eval`

## For Person 3 (benchmarks)

- Prometheus endpoint: `http://localhost:8000/metrics`
- Key metric names (vLLM hard-codes "gpu" prefixes even on TPU):
  - `vllm:time_to_first_token_seconds` — TTFT histogram
  - `vllm:e2e_request_latency_seconds` — end-to-end latency
  - `vllm:gpu_cache_usage_perc` — KV cache utilization
  - `vllm:gpu_prefix_cache_hit_rate` — prefix cache effectiveness
  - `vllm:num_requests_waiting` — queue depth
  - `vllm:num_requests_running` — active concurrency

## Postgres (SQL executor target)

Installed by `setup_vm_docker.sh`:
- DB: `spider_eval`
- Role: `sqlagent` (superuser; localhost only)
- Listening on `localhost:5432`

Person 2's agent connects via `postgresql://sqlagent@localhost/spider_eval`.

## Troubleshooting

- **Container exits immediately** → check `/home/jopin/vllm.log` for the actual error. Common causes: arg renamed in newer vllm (e.g., `--disable-log-requests` is now `--no-enable-log-requests`).
- **Segfault during model init** → model architecture not in vllm-tpu's JAX-native list. Use Llama/Qwen/Gemma/DeepSeek instead.
- **`RESOURCE_EXHAUSTED` on TPU create** → fallback zone `us-east5-b`.
- **OOM on long contexts** → drop `--max-model-len 4096` or raise `--gpu-memory-utilization 0.95` in `serve_vllm.sh`.
- **SSH tunnel disconnected** → reopen it. The container in tmux keeps running.
- **Want a shell inside the running container** → `sudo docker exec -it vllm bash`.
