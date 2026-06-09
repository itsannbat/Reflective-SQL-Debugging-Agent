#!/usr/bin/env python3
"""Experiment harness: run the SQL reflection agent over the full dataset.

Supports two sweep dimensions:
  --verbosity  full | compact        (agent prompt verbosity)
  --concurrency N                    (parallel workers, also stresses the engine)

Optionally snaps Prometheus metrics before/after the run when --prom-url is set.
Results are written as JSONL (one record per task) so partial runs survive crashes.

Quick smoke test (mock LLM, no DB):
    python -m benchmark.harness --mock --limit 3 --output results/smoke.jsonl

Full verbosity sweep (requires SSH tunnel + Postgres):
    python -m benchmark.harness --verbosity full   --output results/full_c1.jsonl
    python -m benchmark.harness --verbosity compact --output results/compact_c1.jsonl

Concurrency sweep (stresses the engine):
    python -m benchmark.harness --verbosity full --concurrency 8 \\
        --prom-url http://localhost:8000/metrics  \\
        --output results/full_c8.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_DSN = "postgresql://sqlagent@localhost/spider_eval"
DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_DATASET = "data/dataset.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_sql(sql: str) -> str:
    """Lowercase + collapse whitespace + strip trailing semicolons."""
    sql = sql.lower().strip().rstrip(";")
    return re.sub(r"\s+", " ", sql)


def _tier_from_task_id(task_id: str) -> str:
    """Extract perturbation tier label from task_id prefix."""
    if "_easy_" in task_id:
        return "easy"
    if "_med_" in task_id:
        return "medium"
    if "_hard_" in task_id:
        return "hard"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-task runner (called from thread pool)
# ---------------------------------------------------------------------------

def _run_one(task: dict, client, model: str, config, condition_meta: dict) -> dict:
    """Run the reflection agent on a single task; return enriched result dict."""
    from agent.loop import ReflectionAgent

    agent = ReflectionAgent(client=client, model=model, config=config)
    t0 = time.perf_counter()
    metrics = agent.run(
        task_id=task["task_id"],
        broken_query=task["broken_query"],
        schema=task.get("schema_ddl", ""),
        question=task.get("question", ""),
    )
    wall_s = time.perf_counter() - t0

    m = metrics.to_dict()

    # Task metadata
    m["difficulty"] = task.get("difficulty", "")
    m["perturbation_tier"] = _tier_from_task_id(task["task_id"])
    m["error_type"] = task.get("error_type", "")
    m["ground_truth_query"] = task.get("ground_truth_query", "")

    # Approximate exact-match against ground truth (structural, not semantic)
    final = m.get("final_query", "")
    gt = m["ground_truth_query"]
    m["query_exact_match"] = bool(final) and (_normalize_sql(final) == _normalize_sql(gt))

    # Wall-clock time (covers threading overhead, distinct from agent's own timer)
    m["wall_latency_s"] = round(wall_s, 3)

    m.update(condition_meta)
    return m


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    *,
    dataset: list[dict],
    client,
    model: str,
    dsn: str,
    verbosity: str,
    max_rounds: int,
    parse_mode: str,
    concurrency: int,
    output_path: str,
    condition_meta: dict,
    prom_url: str | None = None,
) -> list[dict]:
    """Run the full sweep and write results to ``output_path`` (JSONL, appended)."""
    from agent.loop import AgentConfig
    from benchmark.prometheus_scraper import snapshot as prom_snap

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    prom_before = prom_snap(prom_url) if prom_url else {}

    def _make_config(task: dict) -> AgentConfig:
        return AgentConfig(
            dsn=dsn,
            max_rounds=max_rounds,
            verbosity=verbosity,
            parse_mode=parse_mode,
            pg_search_path=task.get("database", ""),
        )

    results: list[dict] = []
    lock = threading.Lock()
    done = 0
    total = len(dataset)
    t_sweep_start = time.perf_counter()

    with open(output_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _run_one, task, client, model, _make_config(task), condition_meta
                ): task
                for task in dataset
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_id": task["task_id"],
                        "perturbation_tier": _tier_from_task_id(task["task_id"]),
                        "difficulty": task.get("difficulty", ""),
                        "error_type": task.get("error_type", ""),
                        "harness_error": str(exc),
                        "success": False,
                    }
                    result.update(condition_meta)

                with lock:
                    done += 1
                    results.append(result)
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()

                    status = "OK" if result.get("success") else "FAIL"
                    err = f" [{result['harness_error']}]" if "harness_error" in result else ""
                    print(
                        f"  [{done:>3}/{total}] {result.get('task_id', '?'):<22} "
                        f"{status}  rounds={result.get('total_rounds', '?')}  "
                        f"tokens={result.get('total_tokens', '?')}{err}",
                        flush=True,
                    )

    elapsed = time.perf_counter() - t_sweep_start
    n_success = sum(1 for r in results if r.get("success"))
    print(
        f"\nSweep complete: {len(results)} tasks, "
        f"success={n_success}/{len(results)} ({n_success/len(results):.1%}), "
        f"elapsed={elapsed:.1f}s"
    )

    prom_after = prom_snap(prom_url) if prom_url else {}
    if prom_before or prom_after:
        prom_path = output_path.replace(".jsonl", "_prom.json")
        with open(prom_path, "w") as f:
            json.dump({"before": prom_before, "after": prom_after}, f, indent=2)
        print(f"Prometheus delta → {prom_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the SQL reflection agent over the full dataset (or a subset).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to dataset.json")
    p.add_argument("--output", required=True, help="JSONL output path (appended to)")
    p.add_argument("--verbosity", choices=["full", "compact"], default="full")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--parse-mode", choices=["json", "xml"], default="json")
    p.add_argument("--concurrency", type=int, default=1, help="Parallel worker threads")
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prom-url", default=None, help="vLLM /metrics URL for engine snapshots")
    p.add_argument("--mock", action="store_true", help="Use MockLLMClient (no real endpoint)")
    # Filtering
    p.add_argument("--tier", choices=["easy", "medium", "hard"],
                   help="Only run tasks from this perturbation tier")
    p.add_argument("--limit", type=int, default=None, help="Cap number of tasks (debugging)")
    args = p.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.tier:
        tier_tag = {"easy": "_easy_", "medium": "_med_", "hard": "_hard_"}[args.tier]
        dataset = [t for t in dataset if tier_tag in t["task_id"]]
    if args.limit:
        dataset = dataset[:args.limit]

    print(
        f"Dataset: {len(dataset)} tasks  |  "
        f"verbosity={args.verbosity}  max_rounds={args.max_rounds}  "
        f"concurrency={args.concurrency}  mock={args.mock}"
    )

    condition_meta = {
        "cond_verbosity": args.verbosity,
        "cond_max_rounds": args.max_rounds,
        "cond_concurrency": args.concurrency,
        "cond_model": "mock" if args.mock else args.model,
        "cond_endpoint": "mock" if args.mock else args.endpoint,
    }

    if args.mock:
        from agent.mock_llm import MockLLMClient
        client = MockLLMClient()
        model = "mock"
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("pip install openai", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(base_url=args.endpoint, api_key="not-needed")
        model = args.model

    run_sweep(
        dataset=dataset,
        client=client,
        model=model,
        dsn=args.dsn,
        verbosity=args.verbosity,
        max_rounds=args.max_rounds,
        parse_mode=args.parse_mode,
        concurrency=args.concurrency,
        output_path=args.output,
        condition_meta=condition_meta,
        prom_url=args.prom_url,
    )


if __name__ == "__main__":
    main()
