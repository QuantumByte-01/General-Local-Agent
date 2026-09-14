from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult
from agent.tools.files import record_file
from agent.tools.paths import resolve_in_workspace


class WriteTool(Tool):
    name = "write_file"
    description = "Create or overwrite a text file in the workspace. Parent folders are created."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    risk = "write"

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_in_workspace(Path(ctx.workspace), str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(arguments["content"])
        path.write_text(content, encoding="utf-8")
        record_file(path, content, ctx.session)
        return ToolResult(ok=True, output=f"wrote {path} ({len(content)} chars)")
