from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.permissions import PermissionDecision, PermissionResult, Risk, resolve_permission
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


@dataclass
class PreparedCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    tool: Tool | None
    error: str | None = None


def _validate(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    required = schema.get("required") or []
    for key in required:
        if key not in arguments or arguments[key] in (None, ""):
            return f"missing required argument: {key}"
    props = schema.get("properties") or {}
    for key in arguments:
        if props and key not in props:
            return f"unknown argument: {key}"
    return None


def prepare_call(
    registry: ToolRegistry,
    call_id: str,
    name: str,
    arguments: dict[str, Any] | None,
) -> PreparedCall:
    arguments = arguments or {}
    tool = registry.get(name)
    if tool is None:
        return PreparedCall(call_id, name, arguments, None, f"unknown tool: {name}")
    err = _validate(tool.schema, arguments)
    if err:
        return PreparedCall(call_id, name, arguments, tool, err)
    return PreparedCall(call_id, name, arguments, tool)


def permission_for(
    prepared: PreparedCall,
    mode: str,
    hook_decision: PermissionDecision | None,
) -> PermissionResult:
    if prepared.tool is None:
        return PermissionResult("deny", prepared.error or "unknown tool")
    risk: Risk = prepared.tool.risk
    if prepared.tool.is_read_only(prepared.arguments):
        risk = "read"
    return resolve_permission(mode=mode, risk=risk, hook_decision=hook_decision)  # type: ignore[arg-type]


def budget_output(result: ToolResult, limit: int) -> ToolResult:
    text = result.output or ""
    if result.error:
        text = (text + "\n" + result.error).strip()
    if len(text) > limit:
        omitted = len(text) - limit
        text = text[:limit] + f"\n…[truncated {omitted} chars]"
        result = ToolResult(ok=result.ok, output=text, extra=result.extra, error=result.error)
    return result
