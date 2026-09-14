from __future__ import annotations

from dataclasses import dataclass

from agent.compact import autocompact
from agent.permissions import PermissionMode


HELP = """Slash commands:
/help — this list
/status — workspace, mode, tools, MCP
/mode plan|default|accept_edits|dont_ask — change permission mode
/compact — shrink old tool output in this session
/clear — new session (keeps MCP and the model latch)
/abort — stop the current turn
/allow — approve pending tool calls
/deny — reject pending tool calls"""


MODES: tuple[PermissionMode, ...] = ("plan", "default", "accept_edits", "dont_ask", "bypass")


@dataclass
class CommandResult:
    handled: bool
    message: str = ""
    resume: str | None = None
    clear: bool = False
    abort: bool = False


def parse_slash(text: str) -> tuple[str, str] | None:
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None
    body = raw[1:].strip()
    if not body:
        return None
    cmd, _, rest = body.partition(" ")
    return cmd.lower(), rest.strip()


def apply_command(engine, text: str) -> CommandResult | None:
    parsed = parse_slash(text)
    if parsed is None:
        return None
    cmd, rest = parsed
    if cmd in {"help", "?"}:
        return CommandResult(True, HELP)
    if cmd == "status":
        return CommandResult(True, engine.status_summary())
    if cmd in {"clear", "new", "reset"}:
        return CommandResult(True, "Session cleared.", clear=True)
    if cmd == "abort":
        engine.session.aborted = True
        return CommandResult(True, "Abort requested.", abort=True)
    if cmd == "compact":
        budget = getattr(engine.settings, "compact_chars", 80_000)
        before = len(engine.app.messages)
        engine.app.messages = autocompact(engine.app.messages, min(budget, 20_000))
        return CommandResult(True, f"Compacted {before} → {len(engine.app.messages)} messages.")
    if cmd == "mode":
        mode = (rest or "").strip().lower()
        if mode not in MODES:
            return CommandResult(True, f"Unknown mode. Use one of: {', '.join(MODES)}")
        engine.session.permission_mode = mode  # type: ignore[assignment]
        engine._system_prompt = None
        engine._prompt_sig = None
        return CommandResult(True, f"Permission mode is now `{mode}`.")
    if cmd in {"allow", "yes", "approve"}:
        return CommandResult(True, "", resume="yes")
    if cmd in {"deny", "no", "reject"}:
        return CommandResult(True, "", resume="no")
    return CommandResult(True, f"Unknown command `/{cmd}`. Try /help.")
