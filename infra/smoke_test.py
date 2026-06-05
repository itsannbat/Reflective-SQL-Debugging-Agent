"""Smoke test for the vLLM endpoint. Run from your laptop after opening the SSH tunnel:

    gcloud compute tpus tpu-vm ssh mlsystems-vllm \
      --zone=us-south1-a --project=x-object-492801-h3 \
      -- -L 8000:localhost:8000 -N &

    python smoke_test.py

Success: prints a SQL completion AND non-zero Prometheus values for TTFT / e2e latency.
"""
import time
import sys
import requests

BASE = "http://localhost:8000"
MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

def main():
    # 1. /health
    h = requests.get(f"{BASE}/health", timeout=10)
    print(f"/health -> {h.status_code}")
    h.raise_for_status()

    # 2. /v1/chat/completions
    prompt = "Write a single SQL query to count the rows in a table named orders."
    t0 = time.time()
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 96,
            "temperature": 0,
        },
        timeout=120,
    )
    dt = time.time() - t0
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]["content"]
    print(f"\n--- completion ({dt:.2f}s end-to-end) ---")
    print(msg.strip())

    # 3. Sample /metrics for the metrics Person 3 cares about
    print("\n--- /metrics (filtered) ---")
    metrics = requests.get(f"{BASE}/metrics", timeout=10).text
    keys = (
        "vllm:time_to_first_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:gpu_cache_usage_perc",
        "vllm:gpu_prefix_cache_hit_rate",
        "vllm:num_requests_waiting",
        "vllm:num_requests_running",
    )
    for line in metrics.splitlines():
        if line.startswith("#"):
            continue
        if any(k in line for k in keys):
            print(line)

if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
