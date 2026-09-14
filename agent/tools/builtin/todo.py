from __future__ import annotations

from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult

_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


class TodoTool(Tool):
    name = "todo"
    description = (
        "Replace the in-session task list. Use for multi-step work so progress stays visible. "
        "status must be pending, in_progress, completed, or cancelled."
    )
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["items"],
    }
    risk = "read"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = arguments.get("items") or []
        if not isinstance(raw, list):
            return ToolResult(ok=False, output="", error="items must be a list")
        items: list[dict[str, str]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                return ToolResult(ok=False, output="", error=f"item {i} must be an object")
            content = str(item.get("content") or "").strip()
            status = str(item.get("status") or "pending").strip().lower()
            if not content:
                return ToolResult(ok=False, output="", error=f"item {i} missing content")
            if status not in _STATUSES:
                return ToolResult(ok=False, output="", error=f"item {i} bad status {status!r}")
            items.append(
                {
                    "id": str(item.get("id") or i + 1),
                    "content": content,
                    "status": status,
                }
            )
        ctx.app.todos = items  # type: ignore[attr-defined]
        if not items:
            return ToolResult(ok=True, output="todo list cleared")
        lines = [f"- [{it['status']}] {it['id']}: {it['content']}" for it in items]
        return ToolResult(ok=True, output="\n".join(lines))
