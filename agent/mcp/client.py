from __future__ import annotations

import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult
from agent.tools.registry import ToolRegistry


def _safe_name(server: str, tool: str) -> str:
    raw = f"mcp_{server}_{tool}"
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)[:64]


class McpTool(Tool):
    risk = "network"

    def __init__(self, name: str, description: str, schema: dict[str, Any], session: Any, remote_name: str) -> None:
        self.name = name
        self.description = description or f"MCP tool {remote_name}"
        self.schema = schema or {"type": "object", "properties": {}}
        self._session = session
        self._remote_name = remote_name

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result = await self._session.call_tool(self._remote_name, arguments)
        chunks: list[str] = []
        contents = getattr(result, "content", None) or []
        for block in contents:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
            else:
                chunks.append(str(block))
        is_error = bool(getattr(result, "isError", False))
        text = "\n".join(chunks) if chunks else json.dumps(result.model_dump() if hasattr(result, "model_dump") else str(result))
        return ToolResult(ok=not is_error, output=text, error="mcp error" if is_error else None)


class McpHub:
    def __init__(self) -> None:
        self.stack: AsyncExitStack | None = None
        self.sessions: dict[str, Any] = {}
        self.errors: list[str] = []

    async def start(self, config_path: Path | None) -> None:
        if config_path is None or not config_path.exists():
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            self.errors.append("mcp package not installed; skipping MCP servers")
            return
        data = json.loads(config_path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers") or data.get("servers") or {}
        if not servers:
            return
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        for name, spec in servers.items():
            try:
                await self._connect_one(name, spec, StdioServerParameters, stdio_client, ClientSession)
            except Exception as exc:
                self.errors.append(f"{name}: {exc}")

    async def _connect_one(
        self,
        name: str,
        spec: dict[str, Any],
        StdioServerParameters: Any,
        stdio_client: Any,
        ClientSession: Any,
    ) -> None:
        if self.stack is None:
            raise RuntimeError("MCP stack not started")
        if spec.get("url") or spec.get("transport") == "sse":
            try:
                from mcp.client.sse import sse_client
            except ImportError as exc:
                raise RuntimeError("sse transport unavailable") from exc
            url = spec.get("url")
            read, write = await self.stack.enter_async_context(sse_client(url))
        else:
            command = spec.get("command")
            args = spec.get("args") or []
            env = spec.get("env")
            params = StdioServerParameters(command=command, args=args, env=env)
            read, write = await self.stack.enter_async_context(stdio_client(params))
        session = await self.stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[name] = session

    async def attach_tools_async(self, registry: ToolRegistry) -> list[str]:
        names: list[str] = []
        for server, session in self.sessions.items():
            try:
                listing = await session.list_tools()
            except Exception as exc:
                self.errors.append(f"{server} list_tools: {exc}")
                continue
            tools = getattr(listing, "tools", listing) or []
            for remote in tools:
                schema = getattr(remote, "inputSchema", None) or getattr(remote, "input_schema", None) or {
                    "type": "object",
                    "properties": {},
                }
                if hasattr(schema, "model_dump"):
                    schema = schema.model_dump()
                tool = McpTool(
                    name=_safe_name(server, remote.name),
                    description=f"[{server}] {getattr(remote, 'description', '') or remote.name}",
                    schema=schema,
                    session=session,
                    remote_name=remote.name,
                )
                try:
                    registry.register(tool)
                    names.append(tool.name)
                except Exception as exc:
                    self.errors.append(f"{server}/{remote.name}: {exc}")
        return names

    async def aclose(self) -> None:
        if self.stack is not None:
            await self.stack.aclose()
            self.stack = None
