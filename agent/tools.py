from __future__ import annotations

from dataclasses import dataclass

import psycopg2

_MAX_ROWS = 50


@dataclass
class ToolResult:
    success: bool
    output: str
    row_count: int = 0


def execute_sql(query: str, dsn: str, search_path: str = "") -> ToolResult:
    conn = None
    try:
        opts = f"-csearch_path={search_path}" if search_path else None
        conn = psycopg2.connect(dsn, **({"options": opts} if opts else {}))
        with conn.cursor() as cur:
            cur.execute(query)
            if cur.description is None:
                conn.rollback()
                return ToolResult(success=True, output="OK (no rows returned)", row_count=0)
            rows = cur.fetchmany(_MAX_ROWS)
            cols = [d.name for d in cur.description]
            conn.rollback()
        lines = ["\t".join(cols)] + ["\t".join("" if v is None else str(v) for v in r) for r in rows]
        truncated = " [results truncated]" if len(rows) == _MAX_ROWS else ""
        lines.append(f"({len(rows)} rows){truncated}")
        return ToolResult(success=True, output="\n".join(lines), row_count=len(rows))
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        msg = (e.pgerror or str(e)).strip()
        return ToolResult(success=False, output=f"SQL ERROR [{e.pgcode}]: {msg}")
    except Exception as e:
        return ToolResult(success=False, output=f"CONNECTION ERROR: {type(e).__name__}: {e}")
    finally:
        if conn:
            conn.close()


def explain_query(query: str, dsn: str, search_path: str = "") -> ToolResult:
    # EXPLAIN ANALYZE executes the query — rollback ensures no side effects.
    sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {query}"
    conn = None
    try:
        opts = f"-csearch_path={search_path}" if search_path else None
        conn = psycopg2.connect(dsn, **({"options": opts} if opts else {}))
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            conn.rollback()
        plan = "\n".join(r[0] for r in rows)
        return ToolResult(success=True, output=plan, row_count=len(rows))
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        msg = (e.pgerror or str(e)).strip()
        return ToolResult(success=False, output=f"EXPLAIN ERROR [{e.pgcode}]: {msg}")
    except Exception as e:
        return ToolResult(success=False, output=f"CONNECTION ERROR: {type(e).__name__}: {e}")
    finally:
        if conn:
            conn.close()
