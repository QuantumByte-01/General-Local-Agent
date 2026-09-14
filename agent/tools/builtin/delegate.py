from __future__ import annotations

from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult


class DelegateTool(Tool):
    name = "delegate_task"
    description = (
        "Spawn a sub-agent with its own message history for a focused subtask. "
        "Use for bounded research or exploration. The child cannot spawn further agents."
    )
    schema = {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "agent_type": {
                "type": "string",
                "description": "explore (read/search) or worker (can write)",
            },
        },
        "required": ["goal"],
    }
    risk = "exec"

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        from agent.tasks.subagent import run_subagent

        goal = str(arguments["goal"])
        agent_type = str(arguments.get("agent_type") or "explore")
        summary = await run_subagent(parent=ctx, goal=goal, agent_type=agent_type)
        return ToolResult(ok=True, output=summary)
