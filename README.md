# Reflective SQL Debugging Agent

CSE 590A mini-project: an agentic system that takes a broken or semantically incorrect SQL query, executes it against a live Postgres database, observes the resulting error or execution plan, and iteratively revises the query until it succeeds (or exhausts a reflection budget).

Reflection / self-correction design pattern. Inference backend: **vLLM serving NousResearch/Meta-Llama-3.1-8B-Instruct on Cloud TPU v5e** (via the official `vllm/vllm-tpu:nightly` Docker image).

> **Model note:** Sticks with the proposal's Llama-3-8B choice, served via the `NousResearch/Meta-Llama-3.1-8B-Instruct` ungated mirror (no Meta HF token needed). We tried Mistral-7B (PyTorch fallback path segfaulted) and Qwen2.5-7B (`Qwen2Config` lacks `text_config` in vllm-tpu's loader); Llama-3.1 uses the well-tested `LlamaForCausalLM` JAX-native path.

## Repo layout

```
.
├── agent/                 # Reflection loop, SQL tools — Person 2
│   ├── loop.py            # Stateful reflection loop (AgentConfig, ReflectionAgent)
│   ├── tools.py           # execute_sql and explain_query (Postgres, always rollback)
│   ├── prompts.py         # System prompt templates and verbosity formatting
│   ├── metrics.py         # Per-round and per-task metric tracking → JSON
│   ├── mock_llm.py        # Drop-in fake LLM client for local testing
│   └── run.py             # CLI entrypoint
├── data/                  # Spider dataset + dataset builder
│   ├── dataset-builder.py
│   ├── dataset.json       # 100 tasks (easy/medium/hard) with broken queries
│   └── spider/
├── infra/                 # vLLM serving on TPU — Person 1
│   ├── README.md          # full operator runbook
│   ├── COST_DISCIPLINE.md # min-time workflow, $ math, cardinal rules
│   ├── provision_tpu.sh
│   ├── stop_tpu.sh
│   ├── setup_vm_docker.sh
│   ├── serve_vllm.sh
│   ├── smoke_test.py
│   └── configs/           # 4 presets for prefix-cache × chunked-prefill A/B
├── scripts/
│   └── load_schemas.py    # Load Spider DB schemas into local Postgres
├── docker-compose.yml     # Local Postgres (spider_eval, port 5432)
└── requirements-agent.txt
```

## Local development

No GCP needed for dev. Three levels depending on how much you want running.

### Level 1 — mock LLM, no database

Confirms the loop, CLI, and JSON output shape work. No Docker required.

```bash
pip install -r requirements-agent.txt
python -m agent.run --task-id spider_easy_001 --mock
```

### Level 2 — real Postgres, mock LLM (recommended for dev)

`execute_sql` hits a real database so you get real SQL errors back. Requires Docker.

```bash
# Start Postgres
docker compose up -d

# Load all Spider schemas (one-time, idempotent)
python scripts/load_schemas.py

# Run a task — real SQL errors, mock LLM
python -m agent.run --task-id spider_easy_001 --mock

# Test sweep variables
python -m agent.run --task-id spider_easy_001 --mock --max-rounds 3
python -m agent.run --task-id spider_easy_001 --mock --verbosity compact
```

### Level 3 — real Postgres, real local LLM via Ollama

Full end-to-end without touching GCP. Requires [Ollama](https://ollama.com) and enough RAM to run an 8B model (~6 GB).

```bash
brew install ollama
ollama pull llama3.1:8b
ollama serve   # runs on localhost:11434

# In another terminal — point the agent at Ollama's OpenAI-compatible API
python -m agent.run --task-id spider_easy_001 \
  --endpoint http://localhost:11434/v1 \
  --model llama3.1:8b
```

> Note: Ollama does not expose Prometheus metrics, so this is only useful for validating agent logic — not for the final experiment measurements.

---

## Final experiment (GCP TPU)

The final experiment requires the TPU VM for vLLM's Prometheus metrics (prefix cache hit rate, TTFT, chunked prefill latency). Run this only when you're ready to collect results — the VM costs ~$115/day while running.

## Inference endpoint (Person 1's deliverable)

The TPU VM `mlsystems-vllm` is provisioned in `us-south1-a` of GCP project `x-object-492801-h3`. vLLM serves an **OpenAI-compatible API** on port 8000 with Prometheus `/metrics` exposed.

### Start / stop the VM

The VM costs ~$115/day while RUNNING and $0 while STOPPED. **Default state should be STOPPED.** See [`infra/COST_DISCIPLINE.md`](infra/COST_DISCIPLINE.md) for the rules.

```bash
# Start (takes ~3 min to READY)
gcloud compute tpus tpu-vm start mlsystems-vllm \
  --zone=us-south1-a --project=x-object-492801-h3

# Stop (do this whenever you walk away)
bash infra/stop_tpu.sh

# Check state
gcloud compute tpus tpu-vm describe mlsystems-vllm \
  --zone=us-south1-a --project=x-object-492801-h3 --format="value(state)"
```

### Run the API

```bash
# 1. (If just started) launch vllm inside tmux on the VM. Idempotent: attaches if already running.
gcloud compute tpus tpu-vm ssh mlsystems-vllm \
  --zone=us-south1-a --project=x-object-492801-h3 \
  --command="tmux has-session -t vllm 2>/dev/null || tmux new -d -s vllm 'bash ~/serve_vllm.sh 2>&1 | tee -a ~/vllm.log'"

# 2. Open an SSH tunnel from your laptop (leave running in its own terminal)
gcloud compute tpus tpu-vm ssh mlsystems-vllm \
  --zone=us-south1-a --project=x-object-492801-h3 \
  -- -L 8000:localhost:8000 -N

# 3. Hit the endpoint from your laptop
python infra/smoke_test.py
# or:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"NousResearch/Meta-Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"SELECT 1"}],"max_tokens":32}'
```

### For Person 2 (agent code)

- Endpoint: `http://localhost:8000/v1` (over the SSH tunnel)
- Model string: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Max context: 8192 tokens
- OpenAI-compatible: use the `openai` SDK with `base_url="http://localhost:8000/v1"` and any non-empty `api_key`.
- Postgres on the same VM: `postgresql://sqlagent@localhost/spider_eval` (database `spider_eval`, superuser role `sqlagent`).

### For Person 3 (benchmarks / sweeps)

- Prometheus metrics: `http://localhost:8000/metrics`
- Key metrics to scrape:
  - `vllm:time_to_first_token_seconds` (TTFT, histogram)
  - `vllm:e2e_request_latency_seconds`
  - `vllm:gpu_cache_usage_perc` (KV cache util — name applies on TPU too)
  - `vllm:gpu_prefix_cache_hit_rate`
  - `vllm:num_requests_waiting` / `vllm:num_requests_running` (queue depth, concurrency)
- A/B knob presets live in [`infra/configs/`](infra/configs/) — source one of them before launching `serve_vllm.sh` to toggle prefix caching × chunked prefill.

### First-time bring-up

See [`infra/README.md`](infra/README.md) for the one-time setup walkthrough (provision → SCP scripts → `setup_vm_docker.sh` → pre-download model → launch).

## Dataset Builder

To generate the dataset, run:

```bash
python3 data/dataset-builder.py \
  --spider_dev data/spider/dev.json \
  --spider_tables data/spider/tables.json \
  --output data/dataset.json
```

## Project deliverables

- **Proposal**: submitted 2026-05-18 (see `Mini-Project MasterDoc.docx` in the parent dir).
- **Final writeup + code**: due **2026-06-11** — 6-page double-column workshop-style paper covering system design, performance results, and optimization analysis.
