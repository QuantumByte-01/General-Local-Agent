from __future__ import annotations

from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult


class SkillTool(Tool):
    name = "load_skill"
    description = (
        "Load the full body of a skill by name. Only metadata is in the system prompt; "
        "call this when the skill applies, then follow its instructions."
    )
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "string", "description": "Optional extra context for the skill"},
        },
        "required": ["name"],
    }
    risk = "read"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = str(arguments["name"])
        extra = str(arguments.get("arguments") or "")
        body = ctx.skills.render(name, extra=extra, session_id=ctx.session.session_id)
        if body is None:
            known = ", ".join(ctx.skills.menu().keys()) or "(none)"
            return ToolResult(ok=False, output="", error=f"unknown skill {name}. known: {known}")
        return ToolResult(ok=True, output=body)
