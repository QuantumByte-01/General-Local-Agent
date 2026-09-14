from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult
from agent.tools.paths import resolve_in_workspace


class MemoryTool(Tool):
    name = "memory_write"
    description = (
        "Persist a memory note for future sessions. kind is one of: "
        "user, project, session, feedback."
    )
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["kind", "title", "body"],
    }
    risk = "write"

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        kind = str(arguments["kind"]).strip().lower()
        if kind not in {"user", "project", "session", "feedback"}:
            return ToolResult(ok=False, output="", error="kind must be user|project|session|feedback")
        note = ctx.memory.append(
            kind=kind,
            title=str(arguments["title"]),
            body=str(arguments["body"]),
            session_id=ctx.session.session_id,
        )
        return ToolResult(ok=True, output=f"saved {kind} memory: {note}")
