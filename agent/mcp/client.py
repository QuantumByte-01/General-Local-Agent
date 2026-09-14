from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult
from agent.tools.registry import ToolRegistry


def _safe_name(server: str, tool: str) -> str:
    raw = f"mcp_{server}_{tool}"
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if len(cleaned) <= 64:
        return cleaned
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:55]}_{digest}"


class McpTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        session: Any,
        remote_name: str,
        read_only: bool = False,
    ) -> None:
        self.name = name
        self.description = description or f"MCP tool {remote_name}"
        self.schema = schema or {"type": "object", "properties": {}}
        self._session = session
        self._remote_name = remote_name
        self.risk = "read" if read_only else "network"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return self.risk == "read"

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return self.risk == "read"

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
        text = "\n".join(chunks) if chunks else json.dumps(
            result.model_dump() if hasattr(result, "model_dump") else str(result)
        )
        return ToolResult(ok=not is_error, output=text, error="mcp error" if is_error else None)


class McpHub:
    def __init__(self) -> None:
        self._stacks: dict[str, AsyncExitStack] = {}
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

        default_timeout = 20.0 if sys.platform == "win32" else 8.0
        async def _one(name: str, spec: dict[str, Any]) -> None:
            await asyncio.wait_for(
                self._connect_one(name, spec, StdioServerParameters, stdio_client, ClientSession),
                timeout=float(spec.get("timeout") or default_timeout),
            )

        results = await asyncio.gather(
            *[_one(name, spec) for name, spec in servers.items()],
            return_exceptions=True,
        )
        for (name, _spec), result in zip(servers.items(), results):
            if isinstance(result, Exception):
                self.errors.append(f"{name}: {result}")

    async def _connect_one(
        self,
        name: str,
        spec: dict[str, Any],
        StdioServerParameters: Any,
        stdio_client: Any,
        ClientSession: Any,
    ) -> None:
        stack = AsyncExitStack()
        try:
            await stack.__aenter__()
            if spec.get("url") or spec.get("transport") == "sse":
                try:
                    from mcp.client.sse import sse_client
                except ImportError as exc:
                    raise RuntimeError("sse transport unavailable") from exc
                read, write = await stack.enter_async_context(sse_client(spec.get("url")))
            else:
                params = StdioServerParameters(
                    command=spec.get("command"),
                    args=spec.get("args") or [],
                    env=spec.get("env"),
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            try:
                await stack.aclose()
            except Exception:
                pass
            raise
        self._stacks[name] = stack
        self.sessions[name] = session

    async def attach_tools_async(self, registry: ToolRegistry) -> list[str]:
        names: list[str] = []
        if not self.sessions:
            return names

        async def _list(server: str, session: Any) -> tuple[str, Any]:
            listing = await asyncio.wait_for(session.list_tools(), timeout=8)
            return server, listing

        listed = await asyncio.gather(
            *[_list(server, session) for server, session in self.sessions.items()],
            return_exceptions=True,
        )
        for item in listed:
            if isinstance(item, Exception):
                self.errors.append(f"list_tools: {item}")
                continue
            server, listing = item
            tools = getattr(listing, "tools", listing) or []
            for remote in tools:
                schema = getattr(remote, "inputSchema", None) or getattr(remote, "input_schema", None) or {
                    "type": "object",
                    "properties": {},
                }
                if hasattr(schema, "model_dump"):
                    schema = schema.model_dump()
                annotations = getattr(remote, "annotations", None)
                read_only = bool(getattr(annotations, "readOnlyHint", False))
                tool = McpTool(
                    name=_safe_name(server, remote.name),
                    description=f"[{server}] {getattr(remote, 'description', '') or remote.name}",
                    schema=schema,
                    session=self.sessions[server],
                    remote_name=remote.name,
                    read_only=read_only,
                )
                try:
                    wanted = tool.name
                    registered = registry.register(tool)
                    if registered != wanted:
                        self.errors.append(f"{server}/{remote.name}: renamed to {registered} (name collision)")
                    names.append(tool.name)
                except Exception as exc:
                    self.errors.append(f"{server}/{remote.name}: {exc}")
        return names

    async def aclose(self) -> None:
        for stack in list(self._stacks.values()):
            try:
                await stack.aclose()
            except Exception:
                pass
        self._stacks.clear()
        self.sessions.clear()
