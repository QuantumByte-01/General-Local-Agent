from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TerminalReason = Literal[
    "completed",
    "aborted",
    "max_turns",
    "token_budget",
    "stop_hook",
    "error",
]


@dataclass
class Event:
    kind: str


@dataclass
class TextDelta(Event):
    kind: str = "text_delta"
    text: str = ""


@dataclass
class AssistantMessage(Event):
    kind: str = "assistant"
    text: str = ""


@dataclass
class ToolStarted(Event):
    kind: str = "tool_started"
    call_id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    speculative: bool = False


@dataclass
class ToolFinished(Event):
    kind: str = "tool_finished"
    call_id: str = ""
    name: str = ""
    ok: bool = True
    output: str = ""
    duration_ms: int = 0


@dataclass
class PermissionRequest(Event):
    kind: str = "permission_request"
    call_id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Status(Event):
    kind: str = "status"
    text: str = ""


@dataclass
class Terminal(Event):
    kind: str = "terminal"
    reason: TerminalReason = "completed"
    text: str = ""
    error: str | None = None
