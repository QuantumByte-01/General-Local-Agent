from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.permissions import PermissionDecision

_RANK = {"deny": 3, "ask": 2, "allow": 1}


def _stronger(
    current: PermissionDecision | None, incoming: PermissionDecision
) -> PermissionDecision:
    if current is None:
        return incoming
    return incoming if _RANK.get(incoming, 0) > _RANK.get(current, 0) else current


@dataclass
class HookHit:
    decision: PermissionDecision | None = None
    block: bool = False
    message: str = ""
    extra_context: str = ""


class HookEngine:
    """Lifecycle hooks from a config snapshot frozen at startup (no TOCTOU reread)."""

    def __init__(self, snapshot: dict[str, Any] | None, trusted: bool = True) -> None:
        self.snapshot = snapshot or {}
        self.trusted = trusted
        self.disabled = bool(self.snapshot.get("disableAllHooks"))

    @classmethod
    def from_path(cls, path: Path | None, trusted: bool) -> "HookEngine":
        if path is None or not path.exists():
            return cls({}, trusted=trusted)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data, trusted=trusted)

    def _handlers(self, event: str) -> list[dict[str, Any]]:
        hooks = self.snapshot.get("hooks") or {}
        return list(hooks.get(event) or [])

    def _match(self, handler: dict[str, Any], tool_name: str) -> bool:
        matcher = handler.get("matcher")
        if not matcher or matcher == "*":
            return True
        names = [part.strip() for part in str(matcher).split("|")]
        return tool_name in names

    def pre_tool_use(self, tool_name: str, arguments: dict[str, Any]) -> HookHit:
        if self.disabled or not self.trusted:
            return HookHit()
        decision: PermissionDecision | None = None
        messages: list[str] = []
        for handler in self._handlers("PreToolUse"):
            if not self._match(handler, tool_name):
                continue
            behavior = handler.get("permissionBehavior")
            if behavior in ("deny", "ask", "allow"):
                decision = _stronger(decision, behavior)
            if handler.get("block"):
                return HookHit(decision="deny", block=True, message=str(handler.get("message") or "blocked by hook"))
            if handler.get("message"):
                messages.append(str(handler["message"]))
        return HookHit(decision=decision, message="\n".join(messages))

    def post_tool_use(self, tool_name: str, output: str) -> str:
        if self.disabled or not self.trusted:
            return ""
        notes = []
        for handler in self._handlers("PostToolUse"):
            if self._match(handler, tool_name) and handler.get("message"):
                notes.append(str(handler["message"]))
        return "\n".join(notes)

    def stop(self, last_text: str) -> HookHit:
        if self.disabled or not self.trusted:
            return HookHit()
        for handler in self._handlers("Stop"):
            if handler.get("block"):
                return HookHit(
                    block=True,
                    message=str(handler.get("message") or "Stop hook requested continuation"),
                )
        return HookHit()
