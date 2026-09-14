from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PermissionMode = Literal["plan", "default", "accept_edits", "dont_ask", "bypass"]
PermissionDecision = Literal["allow", "deny", "ask"]
Risk = Literal["read", "write", "exec", "network"]


@dataclass
class PermissionResult:
    decision: PermissionDecision
    reason: str = ""


def resolve_permission(
    *,
    mode: PermissionMode,
    risk: Risk,
    hook_decision: PermissionDecision | None = None,
) -> PermissionResult:
    """Named-mode resolution. Hook deny always wins; otherwise the active mode decides."""
    if hook_decision == "deny":
        return PermissionResult("deny", "blocked by PreToolUse hook")
    if hook_decision == "allow":
        return PermissionResult("allow", "allowed by PreToolUse hook")
    if hook_decision == "ask":
        return PermissionResult("ask", "hook requested confirmation")

    if mode == "bypass":
        return PermissionResult("allow", "bypass mode")
    if mode == "dont_ask":
        return PermissionResult("allow", "dont_ask mode")
    if mode == "plan":
        if risk == "read":
            return PermissionResult("allow", "plan mode allows reads")
        return PermissionResult("deny", "plan mode blocks mutations and network")
    if mode == "accept_edits":
        if risk in ("read", "write"):
            return PermissionResult("allow", "accept_edits auto-approves file I/O")
        return PermissionResult("ask", "accept_edits still confirms exec/network")
    # default
    if risk == "read":
        return PermissionResult("allow", "reads are allowed")
    return PermissionResult("ask", f"{risk} actions need confirmation")


def is_affirmative(text: str | bool | None) -> bool | None:
    if isinstance(text, bool):
        return text
    if not text or not str(text).strip():
        return None
    token = str(text).strip().lower()
    yes = {
        "yes", "y", "ok", "okay", "sure", "proceed", "continue",
        "confirm", "go ahead", "run", "execute", "allow", "approve",
    }
    no = {"no", "n", "stop", "cancel", "abort", "deny", "reject"}
    if token in yes or token.startswith("yes"):
        return True
    if token in no or token.startswith("no"):
        return False
    return None
