from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent.config import Settings
from agent.events import Event
from agent.hooks.engine import HookEngine
from agent.llm.client import GeminiClient
from agent.memory.store import MemoryStore
from agent.skills.loader import SkillLoader
from agent.state import AppState, SessionState
from agent.tools.registry import ToolRegistry


class Engine:
    """Runtime context shared by the query loop and tools."""

    def __init__(
        self,
        *,
        settings: Settings,
        session: SessionState,
        app: AppState,
        registry: ToolRegistry,
        llm: GeminiClient,
        hooks: HookEngine,
        memory: MemoryStore,
        skills: SkillLoader,
        mcp: Any = None,
    ) -> None:
        self.settings = settings
        self.session = session
        self.app = app
        self.registry = registry
        self.llm = llm
        self.hooks = hooks
        self.memory = memory
        self.skills = skills
        self.mcp = mcp
        self.workspace: Path = settings.workspace
        self._sink: Callable[[Event], Awaitable[None]] = self._noop

    async def _noop(self, event: Event) -> None:
        return None

    async def emit(self, event: Event) -> None:
        await self._sink(event)

    def status_summary(self) -> str:
        skills = ", ".join(self.skills.menu()) or "none"
        mcp_names = ", ".join(getattr(self.mcp, "sessions", {}) or {}) or "none"
        mcp_err = "; ".join(getattr(self.mcp, "errors", []) or [])
        n_tools = len(self.registry.all())
        line = (
            f"workspace `{self.workspace}` · {n_tools} tools · "
            f"mode `{self.session.permission_mode}` · skills: {skills} · MCP: {mcp_names}"
        )
        if mcp_err:
            line += f" · MCP issues: {mcp_err}"
        return line
