from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.permissions import PermissionMode


@dataclass
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileReadRecord:
    mtime: float
    size: int
    excerpt_hash: str


@dataclass
class SessionState:
    """Infrastructure singleton: rarely changes, not UI-reactive."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace: Path = field(default_factory=Path.cwd)
    project_root: Path = field(default_factory=Path.cwd)
    model: str | None = None  # sticky latch: once set, stay on it
    permission_mode: PermissionMode = "default"
    max_turns: int = 24
    max_output_tokens: int = 8192
    started_at: float = field(default_factory=time.time)
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    aborted: bool = False
    trusted: bool = True
    file_reads: dict[str, FileReadRecord] = field(default_factory=dict)


@dataclass
class AppState:
    """UI-facing store: messages, pending approvals, live status."""

    messages: list[ChatMessage] = field(default_factory=list)
    pending_calls: list[dict[str, Any]] = field(default_factory=list)
    pending_reason: str = ""
    status: str = ""
    last_terminal: str = ""
