from __future__ import annotations

from dataclasses import dataclass, field

from agent.events import Event, PermissionRequest, Terminal
from agent.loop import run_query
from agent.runtime import Engine


@dataclass
class Trace:
    events: list[Event] = field(default_factory=list)
    terminal: Terminal | None = None
    pending: bool = False

    @property
    def tool_names(self) -> list[str]:
        return [e.name for e in self.events if getattr(e, "kind", "") == "tool_finished"]

    @property
    def assistant_text(self) -> str:
        parts = [e.text for e in self.events if getattr(e, "kind", "") == "assistant"]
        if self.terminal and self.terminal.text and self.terminal.text not in parts:
            parts.append(self.terminal.text)
        return "\n".join(parts)

    def kinds(self) -> list[str]:
        return [getattr(e, "kind", "") for e in self.events]


async def run_harness(
    engine: Engine,
    user_text: str,
    *,
    resumes: list[str] | None = None,
    max_pauses: int = 6,
) -> Trace:
    """Drive the query loop, answering permission prompts from `resumes`."""
    replies = list(resumes or [])
    trace = Trace()
    prompt: str | None = user_text
    resume: str | None = None
    pauses = 0

    while True:
        async for event in run_query(engine, prompt, resume=resume):
            trace.events.append(event)
            if isinstance(event, Terminal):
                trace.terminal = event
        prompt = None
        resume = None
        last = trace.events[-1] if trace.events else None
        if last is None or isinstance(last, Terminal):
            return trace
        if isinstance(last, PermissionRequest):
            trace.pending = True
            pauses += 1
            if pauses > max_pauses or not replies:
                return trace
            resume = replies.pop(0)
            continue
        return trace
