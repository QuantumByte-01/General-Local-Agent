from __future__ import annotations

from typing import Any, Protocol

from agent.llm.client import LlmTurn
from agent.state import ChatMessage


class LlmClient(Protocol):
    async def generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tool_defs: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> LlmTurn: ...
