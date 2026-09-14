from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult
from agent.tools.files import record_file
from agent.tools.paths import resolve_in_workspace


class EditTool(Tool):
    name = "edit_file"
    description = (
        "Replace exact text in a file. Fails if old_string is missing or not unique "
        "(unless replace_all). Re-read the file if it changed since the last read."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old_string", "new_string"],
    }
    risk = "write"

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_in_workspace(Path(ctx.workspace), str(arguments["path"]), must_exist=True)
        record = ctx.session.file_reads.get(str(path))
        stat = path.stat()
        if record and abs(stat.st_mtime - record.mtime) > 0.01:
            return ToolResult(
                ok=False,
                output="",
                error="file changed since last read; call read_file first",
            )
        original = path.read_text(encoding="utf-8", errors="replace")
        old = str(arguments["old_string"])
        new = str(arguments["new_string"])
        replace_all = bool(arguments.get("replace_all"))
        count = original.count(old)
        if count == 0:
            return ToolResult(ok=False, output="", error="old_string not found")
        if count > 1 and not replace_all:
            return ToolResult(
                ok=False,
                output="",
                error=f"old_string matched {count} times; pass replace_all or more context",
            )
        updated = original.replace(old, new) if replace_all else original.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        record_file(path, updated, ctx.session)
        n = count if replace_all else 1
        return ToolResult(ok=True, output=f"updated {path} ({n} replacement(s))")
