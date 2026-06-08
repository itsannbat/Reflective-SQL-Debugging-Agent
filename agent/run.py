#!/usr/bin/env python3
"""CLI entrypoint for running the reflection agent on a single task.

Load a task from dataset.json by task_id:
    python -m agent.run --task-id spider_easy_001 --mock

Run against the real vLLM endpoint (requires SSH tunnel):
    python -m agent.run --task-id spider_easy_001

Pass a query directly (skips dataset lookup):
    python -m agent.run --broken-query "SELET 1" --schema "" --mock
"""
from __future__ import annotations

import argparse
import json
import sys

DEFAULT_DSN = "postgresql://sqlagent@localhost/spider_eval"
DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_DATASET = "data/dataset.json"


def _load_task(dataset_path: str, task_id: str) -> dict:
    try:
        with open(dataset_path) as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    task = next((t for t in dataset if t["task_id"] == task_id), None)
    if task is None:
        ids = [t["task_id"] for t in dataset]
        print(f"Task {task_id!r} not found. Available: {ids[:10]} ...", file=sys.stderr)
        sys.exit(1)
    return task


def main() -> None:
    p = argparse.ArgumentParser(description="Run the SQL reflection agent on one task.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--task-id", help="Load task from dataset.json by ID")
    src.add_argument("--broken-query", help="Broken query (inline mode, skips dataset)")

    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to dataset.json")
    p.add_argument("--schema", default="", help="DDL schema (inline mode)")
    p.add_argument("--question", default="", help="Natural language question (inline mode)")
    p.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="vLLM base URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name for the OpenAI API call")
    p.add_argument("--max-rounds", type=int, default=5, help="Reflection budget")
    p.add_argument("--verbosity", choices=["full", "compact"], default="full")
    p.add_argument("--parse-mode", choices=["json", "xml"], default="json")
    p.add_argument("--mock", action="store_true", help="Use MockLLMClient instead of vLLM")
    args = p.parse_args()

    if not args.task_id and not args.broken_query:
        p.error("Provide --task-id or --broken-query.")

    if args.task_id:
        task = _load_task(args.dataset, args.task_id)
        broken_query = task["broken_query"]
        schema = task.get("schema_ddl", "")
        question = task.get("question", "")
        task_id = task["task_id"]
        pg_search_path = task.get("database", "")
    else:
        broken_query = args.broken_query
        schema = args.schema
        question = args.question
        task_id = "adhoc"
        pg_search_path = ""

    # Import here so the file is importable without openai installed in mock mode.
    from agent.loop import AgentConfig, ReflectionAgent
    from agent.mock_llm import MockLLMClient

    if args.mock:
        client = MockLLMClient()
        model = "mock"
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("Install openai: pip install openai", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(base_url=args.endpoint, api_key="not-needed")
        model = args.model

    config = AgentConfig(
        dsn=args.dsn,
        max_rounds=args.max_rounds,
        verbosity=args.verbosity,
        parse_mode=args.parse_mode,
        pg_search_path=pg_search_path,
    )
    agent = ReflectionAgent(client=client, model=model, config=config)
    result = agent.run(task_id=task_id, broken_query=broken_query, schema=schema, question=question)

    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
