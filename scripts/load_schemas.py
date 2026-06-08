#!/usr/bin/env python3
"""Load all Spider database schemas from dataset.json into a local Postgres instance.

Each Spider database (e.g. concert_singer) becomes a Postgres schema (namespace)
inside spider_eval, so all 6 databases coexist without table-name collisions.

Usage:
    python scripts/load_schemas.py
    python scripts/load_schemas.py --dsn postgresql://sqlagent@localhost/spider_eval
    python scripts/load_schemas.py --dataset data/dataset.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

DEFAULT_DSN = "postgresql://sqlagent@localhost/spider_eval"
DEFAULT_DATASET = "data/dataset.json"


def load(dsn: str, dataset_path: str, dry_run: bool) -> None:
    try:
        import psycopg2
    except ImportError:
        print("pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path) as f:
        tasks = json.load(f)

    # Collect unique (database_name → schema_ddl) pairs.
    db_schemas: dict[str, str] = {}
    for t in tasks:
        db = t["database"]
        if db not in db_schemas:
            db_schemas[db] = t["schema_ddl"]

    print(f"Found {len(db_schemas)} unique databases: {sorted(db_schemas)}")

    if dry_run:
        for db in sorted(db_schemas):
            print(f"\n-- Would create schema '{db}' with DDL:\n{db_schemas[db][:200]}...")
        return

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()

    for db_name, ddl in sorted(db_schemas.items()):
        print(f"\n==> Loading schema '{db_name}' ...", end=" ", flush=True)
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db_name}")
        cur.execute(f"SET search_path TO {db_name}")

        # Drop existing tables so re-runs are idempotent.
        cur.execute("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = %s
        """, (db_name,))
        existing = [r[0] for r in cur.fetchall()]
        for tbl in existing:
            cur.execute(f"DROP TABLE IF EXISTS {db_name}.{tbl} CASCADE")

        # Run each CREATE TABLE statement separately.
        for stmt in ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)

        cur.execute(f"SET search_path TO public")
        print("OK")

    cur.close()
    conn.close()
    print(f"\nDone. All schemas loaded into {dsn}")
    print("\nVerify with:")
    print(f"  psql {dsn} -c '\\dn'")
    print(f"  psql {dsn} -c '\\dt concert_singer.*'")


def main() -> None:
    p = argparse.ArgumentParser(description="Load Spider schemas into Postgres.")
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dry-run", action="store_true", help="Print DDL without executing")
    args = p.parse_args()
    load(args.dsn, args.dataset, args.dry_run)


if __name__ == "__main__":
    main()
