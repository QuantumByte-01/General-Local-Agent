from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.llm.client import LlmTurn
from agent.state import ChatMessage


@dataclass
class ScriptedLLM:
    """Deterministic stand-in for Gemini. Each generate() pops the next turn."""

    turns: list[LlmTurn] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    latched_model: str | None = "scripted"
    exhausted_text: str = "(script exhausted)"

    async def generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tool_defs: list[dict[str, Any]],
        max_output_tokens: int,
        **_kwargs: Any,
    ) -> LlmTurn:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tool_count": len(tool_defs),
                "max_output_tokens": max_output_tokens,
            }
        )
        if not self.turns:
            return LlmTurn(text=self.exhausted_text, model="scripted")
        turn = self.turns.pop(0)
        if not turn.model:
            turn.model = "scripted"
        return turn

    async def stream_generate(self, **kwargs: Any):
        turn = await self.generate(**kwargs)
        if turn.text:
            yield "text", turn.text
        yield "turn", turn
