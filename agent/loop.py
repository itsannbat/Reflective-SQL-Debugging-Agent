from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .metrics import RoundMetrics, TaskMetrics
from .prompts import build_initial_user_message, build_system_prompt, format_tool_result
from .tools import ToolResult, execute_sql, explain_query

_KNOWN_TOOLS = {"execute_sql", "explain_query", "final_answer"}

_PARSE_ERROR_MSG = (
    "[PARSE ERROR] Could not find a valid tool call in your response. "
    "Output exactly one JSON code block with keys 'tool' and 'query', nothing else."
)

_PARSE_ERROR_MSG_XML = (
    "[PARSE ERROR] Could not find a valid <tool_call> block in your response. "
    "Output exactly one <tool_call> block with <name> and <query> children, nothing else."
)


@dataclass
class AgentConfig:
    dsn: str
    max_rounds: int = 5
    verbosity: str = "full"   # "full" | "compact"
    parse_mode: str = "json"  # "json" | "xml"
    pg_search_path: str = ""  # Postgres schema name (e.g. "concert_singer")


class ReflectionAgent:
    def __init__(self, client, model: str, config: AgentConfig):
        self._client = client
        self._model = model
        self._cfg = config

    def run(
        self,
        task_id: str,
        broken_query: str,
        schema: str,
        question: str = "",
    ) -> TaskMetrics:
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(self._cfg.parse_mode)},
            {"role": "user", "content": build_initial_user_message(schema, broken_query, question)},
        ]

        task = TaskMetrics(task_id=task_id, broken_query=broken_query)
        t_task_start = time.perf_counter()

        for round_num in range(1, self._cfg.max_rounds + 1):
            t_round = time.perf_counter()

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=512,
                temperature=0,
            )

            assistant_text: str = response.choices[0].message.content or ""
            usage = response.usage

            round_m = RoundMetrics(
                round_num=round_num,
                latency_s=time.perf_counter() - t_round,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )

            tool_call = self._parse_tool_call(assistant_text)

            if tool_call is None:
                # Unparseable response — inject correction and continue.
                err = _PARSE_ERROR_MSG_XML if self._cfg.parse_mode == "xml" else _PARSE_ERROR_MSG
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "user", "content": err})
                task.rounds.append(round_m)
                continue

            tool_name = tool_call.get("tool", "")
            query = tool_call.get("query", "").strip()
            round_m.query_attempt = query
            round_m.tool_name = tool_name

            if tool_name == "final_answer":
                task.success = True
                task.final_query = query
                task.rounds.append(round_m)
                break

            result = self._dispatch_tool(tool_name, query)
            round_m.tool_success = result.success

            result_text = format_tool_result(tool_name, result, self._cfg.verbosity)
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": result_text})
            task.rounds.append(round_m)

        task.total_latency_s = time.perf_counter() - t_task_start
        task.total_prompt_tokens = sum(r.prompt_tokens for r in task.rounds)
        task.total_completion_tokens = sum(r.completion_tokens for r in task.rounds)
        return task

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_tool_call(self, text: str) -> dict | None:
        if self._cfg.parse_mode == "xml":
            return self._parse_xml(text)
        return self._parse_json(text)

    def _parse_json(self, text: str) -> dict | None:
        # Prefer ```json ... ``` fenced block.
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if "tool" in obj and "query" in obj:
                    return obj
            except json.JSONDecodeError:
                pass

        # Fallback: any bare { ... } that has both keys.
        for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
            try:
                obj = json.loads(m.group(0))
                if "tool" in obj and "query" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

        return None

    def _parse_xml(self, text: str) -> dict | None:
        m = re.search(
            r"<tool_call>\s*<name>(.*?)</name>\s*<query>(.*?)</query>\s*</tool_call>",
            text,
            re.DOTALL,
        )
        if m:
            return {"tool": m.group(1).strip(), "query": m.group(2).strip()}
        return None

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, tool_name: str, query: str) -> ToolResult:
        sp = self._cfg.pg_search_path
        if tool_name == "execute_sql":
            return execute_sql(query, self._cfg.dsn, search_path=sp)
        if tool_name == "explain_query":
            return explain_query(query, self._cfg.dsn, search_path=sp)
        return ToolResult(success=False, output=f"Unknown tool '{tool_name}'. Use: {', '.join(_KNOWN_TOOLS)}")
