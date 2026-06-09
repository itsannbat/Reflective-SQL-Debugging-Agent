#!/usr/bin/env python3
"""Scrape vLLM Prometheus /metrics and return key engine metrics as a flat dict.

Standalone usage:
    python -m benchmark.prometheus_scraper --url http://localhost:8000/metrics
"""
from __future__ import annotations

import datetime
from typing import Any

# Metric names to extract from the vLLM Prometheus exposition
_METRICS_OF_INTEREST = {
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:gpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:num_requests_waiting",
    "vllm:num_requests_running",
    "vllm:num_requests_finished",
}


def snapshot(url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Fetch Prometheus /metrics and return a flat dict of metric → value.

    Returns {"error": <msg>} if the endpoint is unreachable.
    Derives mean TTFT and mean e2e latency from histogram sum/count pairs.
    """
    try:
        import requests
    except ImportError:
        return {"error": "requests not installed; pip install requests"}

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        return {"error": str(exc)}

    result: dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "url": url,
    }

    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Strip label set {key="val",...} to get the bare metric name
        name = parts[0].split("{")[0]
        if name not in _METRICS_OF_INTEREST:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        # Accumulate across label variants (e.g. different model= labels)
        result[name] = result.get(name, 0.0) + value

    # Derive mean latencies from histogram sum/count
    for base in (
        "vllm:time_to_first_token_seconds",
        "vllm:e2e_request_latency_seconds",
    ):
        s = result.get(f"{base}_sum", 0.0)
        c = result.get(f"{base}_count", 0.0)
        result[f"{base}_mean"] = (s / c) if c > 0 else None

    return result


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Snapshot vLLM Prometheus metrics.")
    p.add_argument("--url", default="http://localhost:8000/metrics")
    p.add_argument("--timeout", type=float, default=5.0)
    args = p.parse_args()
    print(json.dumps(snapshot(args.url, args.timeout), indent=2))
