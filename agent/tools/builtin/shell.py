from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult

_READ_HEADS = (
    "dir", "ls", "type ", "cat ", "get-content", "get-childitem", "pwd",
    "echo ", "where ", "whoami", "hostname", "git status", "git log",
    "git diff", "git branch", "python --version", "uv --version",
)


def _looks_read_only(command: str) -> bool:
    cmd = command.strip().lower()
    if not cmd:
        return False
    if any(tok in cmd for tok in ("&&", "|", ">", "<", "`", "$(", "rm ", "del ", "move ", "copy ")):
        if not cmd.startswith("git diff") and not cmd.startswith("git log"):
            if any(tok in cmd for tok in (">", "rm ", "del ", "move ")):
                return False
    return any(cmd == h.strip() or cmd.startswith(h) for h in _READ_HEADS)


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run one shell command on this Windows machine. Prefer PowerShell syntax. "
        "Do not chain unrelated commands. Use for git, system queries, and programs "
        "that are not covered by dedicated tools."
    )
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Single command to run"},
            "timeout_seconds": {"type": "integer", "description": "Timeout, default 30"},
            "workdir": {"type": "string", "description": "Optional working directory"},
        },
        "required": ["command"],
    }
    risk = "exec"

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return _looks_read_only(str(arguments.get("command", "")))

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return self.is_read_only(arguments)

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(arguments["command"])
        timeout = int(arguments.get("timeout_seconds") or 30)
        workdir = arguments.get("workdir") or str(ctx.workspace)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        def _run() -> tuple[int, str, str]:
            import subprocess

            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env=env,
            )
            return proc.returncode, proc.stdout, proc.stderr

        try:
            code, stdout, stderr = await asyncio.to_thread(_run)
        except Exception as exc:
            return ToolResult(ok=False, output="", error=str(exc))
        text = stdout.strip()
        if stderr.strip():
            text = (text + "\nSTDERR:\n" + stderr.strip()).strip()
        return ToolResult(
            ok=code == 0,
            output=text or f"(exit {code}, no output)",
            extra={"returncode": code, "command": command},
            error=None if code == 0 else f"exit {code}",
        )
