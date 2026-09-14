from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from agent.permissions import Risk


@dataclass
class ToolResult:
    ok: bool
    output: str
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ToolContext(Protocol):
    workspace: Any
    session: Any
    llm: Any
    hooks: Any
    memory: Any
    skills: Any
    emit: Callable[[Any], Awaitable[None] | None]


class Tool:
    """Self-describing tool. Fail-closed: unsafe and serial unless opted in."""

    name: str = ""
    description: str = ""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    risk: Risk = "write"
    max_result_chars: int = 12_000

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return False

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return self.risk == "read"

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError
