"""Mock OpenAI client for local testing without a vLLM endpoint.

Usage:
    from agent.mock_llm import MockLLMClient
    client = MockLLMClient()                       # default: always reaches final_answer in 2 rounds
    client = MockLLMClient(response_fn=my_fn)      # custom response logic

The default response function re-submits the broken query as execute_sql on round 1,
then declares final_answer on round 2 — good enough to exercise the full loop.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Minimal stubs matching the openai SDK response shape that loop.py relies on.
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _ChatCompletion:
    choices: list[_Choice]
    usage: _Usage


class _MockCompletions:
    def __init__(self, response_fn: Callable[[list[dict]], str]) -> None:
        self._fn = response_fn

    def create(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0,
        **kwargs,
    ) -> _ChatCompletion:
        content = self._fn(messages)
        prompt_tokens = sum(len(m["content"]) // 4 for m in messages)
        completion_tokens = max(1, len(content) // 4)
        return _ChatCompletion(
            choices=[_Choice(_Message(content))],
            usage=_Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )


class _MockChat:
    def __init__(self, response_fn: Callable[[list[dict]], str]) -> None:
        self.completions = _MockCompletions(response_fn)


class MockLLMClient:
    """Drop-in replacement for openai.OpenAI() that returns scripted responses."""

    def __init__(self, response_fn: Callable[[list[dict]], str] | None = None) -> None:
        self.chat = _MockChat(response_fn or _default_response)


# ---------------------------------------------------------------------------
# Default response function
# ---------------------------------------------------------------------------

def _default_response(messages: list[dict]) -> str:
    """Round 1: execute_sql with the original broken query. Round 2+: final_answer."""
    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")

    # Extract the broken query from the initial user message (the ```sql block).
    query = "SELECT 1"
    for m in messages:
        if m["role"] == "user":
            match = re.search(r"```sql\n(.*?)```", m["content"], re.DOTALL)
            if match:
                query = match.group(1).strip()
                break

    tool = "execute_sql" if assistant_turns == 0 else "final_answer"
    return "```json\n" + json.dumps({"tool": tool, "query": query}) + "\n```"
