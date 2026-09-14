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


def _coerce_value(expected: str, value: Any) -> tuple[Any, str | None]:
    expected = (expected or "string").lower()
    if expected == "string":
        if isinstance(value, (dict, list)):
            return value, "expected string"
        return str(value), None
    if expected == "integer":
        if isinstance(value, bool):
            return value, "expected integer"
        if isinstance(value, int):
            return value, None
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            return int(value.strip()), None
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        return value, "expected integer"
    if expected in {"number", "float"}:
        if isinstance(value, bool):
            return value, "expected number"
        if isinstance(value, (int, float)):
            return float(value), None
        try:
            return float(value), None
        except (TypeError, ValueError):
            return value, "expected number"
    if expected == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "yes", "no", "1", "0"}:
            return value.strip().lower() in {"true", "yes", "1"}, None
        return value, "expected boolean"
    if expected == "array":
        if isinstance(value, list):
            return value, None
        return value, "expected array"
    if expected == "object":
        if isinstance(value, dict):
            return value, None
        return value, "expected object"
    return value, None


def _validate(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    required = schema.get("required") or []
    for key in required:
        if key not in arguments or arguments[key] in (None, ""):
            return f"missing required argument: {key}"
    props = schema.get("properties") or {}
    for key in list(arguments):
        if props and key not in props:
            return f"unknown argument: {key}"
        spec = props.get(key) if isinstance(props, dict) else None
        if not isinstance(spec, dict):
            continue
        expected = str(spec.get("type") or "string")
        coerced, err = _coerce_value(expected, arguments[key])
        if err:
            return f"argument {key}: {err}"
        arguments[key] = coerced
        allowed = spec.get("enum")
        if allowed is not None and arguments[key] not in allowed:
            return f"argument {key}: value not in enum"
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
