from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult

_SKIP_DIRS = {".git", ".svn", ".hg", ".venv", "node_modules", "__pycache__"}


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents in the workspace. Uses ripgrep when installed, "
        "otherwise a Python walk. Paginate with head_limit/offset."
    )
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex or fixed string"},
            "glob": {"type": "string", "description": "Optional file glob like *.py"},
            "head_limit": {"type": "integer"},
            "offset": {"type": "integer"},
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
        glob = arguments.get("glob")
        limit = int(arguments.get("head_limit") or 250)
        offset = int(arguments.get("offset") or 0)
        root = Path(ctx.workspace)
        rg = shutil.which("rg")
        lines: list[str] = []
        if rg:
            cmd = [rg, "--line-number", "--color", "never", "-e", pattern, str(root)]
            if glob:
                cmd.extend(["--glob", str(glob)])
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            except Exception:
                lines = []
        if not lines and not rg:
            rx = re.compile(pattern)
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
                for name in filenames:
                    if glob and not Path(name).match(glob):
                        continue
                    path = Path(dirpath) / name
                    try:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    for i, line in enumerate(text.splitlines(), 1):
                        if rx.search(line):
                            rel = path.relative_to(root)
                            lines.append(f"{rel}:{i}:{line}")
        window = lines[offset : offset + limit]
        note = ""
        if len(lines) > offset + limit:
            note = f"\nappliedLimit: {limit} total: {len(lines)} (use offset to paginate)"
        return ToolResult(ok=True, output="\n".join(window) + note if window else "(no matches)")
