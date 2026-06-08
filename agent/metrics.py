from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoundMetrics:
    round_num: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    query_attempt: str = ""
    tool_name: str = ""
    tool_success: bool | None = None

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "query_attempt": self.query_attempt,
            "tool_name": self.tool_name,
            "tool_success": self.tool_success,
        }


@dataclass
class TaskMetrics:
    task_id: str
    broken_query: str
    success: bool = False
    final_query: str = ""
    total_latency_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    rounds: list[RoundMetrics] = field(default_factory=list)

    def rounds_to_success(self) -> int | None:
        return len(self.rounds) if self.success else None

    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "rounds_to_success": self.rounds_to_success(),
            "total_rounds": len(self.rounds),
            "total_latency_s": round(self.total_latency_s, 3),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens(),
            "final_query": self.final_query,
            "rounds": [r.to_dict() for r in self.rounds],
        }
