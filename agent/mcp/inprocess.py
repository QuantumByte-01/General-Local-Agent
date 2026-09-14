from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from agent.mcp.client import McpHub, McpTool, _safe_name
from agent.tools.registry import ToolRegistry

Handler = Callable[[dict[str, Any]], Any]


@dataclass
class _Text:
    text: str


@dataclass
class _CallResult:
    content: list[_Text]
    isError: bool = False


class InProcessSession:
    """MCP-shaped session with no subprocess and no sockets."""

    def __init__(self, tools: dict[str, tuple[str, dict[str, Any], Handler]]) -> None:
        self._tools = tools

    async def list_tools(self) -> Any:
        listed = []
        for name, (description, schema, _handler) in self._tools.items():
            listed.append(
                SimpleNamespace(
                    name=name,
                    description=description,
                    inputSchema=schema,
                    annotations=SimpleNamespace(readOnlyHint=True),
                )
            )
        return SimpleNamespace(tools=listed)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> _CallResult:
        if name not in self._tools:
            return _CallResult(content=[_Text(f"unknown tool {name}")], isError=True)
        _desc, _schema, handler = self._tools[name]
        try:
            result = handler(arguments or {})
            if isinstance(result, Awaitable):
                result = await result
            return _CallResult(content=[_Text(str(result))])
        except Exception as exc:
            return _CallResult(content=[_Text(str(exc))], isError=True)


def harness_echo_tools() -> dict[str, tuple[str, dict[str, Any], Handler]]:
    return {
        "echo": (
            "Echo text back. For harness and smoke tests.",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            lambda args: str(args.get("text", "")),
        ),
        "add": (
            "Add two integers and return the sum.",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            lambda args: int(args.get("a") or 0) + int(args.get("b") or 0),
        ),
    }


def attach_inprocess(
    registry: ToolRegistry,
    *,
    server: str = "harness",
    tools: dict[str, tuple[str, dict[str, Any], Handler]] | None = None,
) -> McpHub:
    tools = tools or harness_echo_tools()
    session = InProcessSession(tools)
    hub = McpHub()
    hub.sessions[server] = session
    for name, (description, schema, _handler) in tools.items():
        tool = McpTool(
            name=_safe_name(server, name),
            description=f"[{server}] {description}",
            schema=schema,
            session=session,
            remote_name=name,
            read_only=True,
        )
        registry.register(tool)
    return hub
