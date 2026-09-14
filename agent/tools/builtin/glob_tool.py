from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult


class GlobTool(Tool):
    name = "glob_files"
    description = "Find files under the workspace matching a glob pattern (e.g. **/*.py)."
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "head_limit": {"type": "integer", "description": "max matches, default 200"},
        },
        "required": ["pattern"],
    }
    risk = "read"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = str(arguments["pattern"])
        limit = int(arguments.get("head_limit") or 200)
        root = Path(ctx.workspace)
        matches = [p for p in root.glob(pattern) if p.is_file()]
        matches.sort()
        clipped = matches[:limit]
        lines = [str(p.relative_to(root)) for p in clipped]
        extra = f"\n(truncated, showing {limit} of {len(matches)})" if len(matches) > limit else ""
        return ToolResult(ok=True, output="\n".join(lines) + extra if lines else "(no matches)")
