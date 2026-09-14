from __future__ import annotations

import asyncio
from pathlib import Path

from agent.config import load_settings
from agent.hooks.engine import HookEngine
from agent.llm.client import GeminiClient
from agent.mcp.client import McpHub
from agent.memory.store import MemoryStore
from agent.runtime import Engine
from agent.skills.loader import SkillLoader
from agent.state import AppState, SessionState
from agent.tools.builtin import register_builtin
from agent.tools.registry import ToolRegistry
from agent.permissions import PermissionMode


async def bootstrap(project_root: Path | None = None, permission_mode: PermissionMode | None = None) -> Engine:
    settings = load_settings(project_root)
    mode: PermissionMode = permission_mode or settings.permission_mode  # type: ignore[assignment]
    if mode not in {"plan", "default", "accept_edits", "dont_ask", "bypass"}:
        mode = "default"

    session = SessionState(
        workspace=settings.workspace,
        project_root=settings.project_root,
        permission_mode=mode,
        max_turns=settings.max_turns,
        max_output_tokens=settings.max_output_tokens,
        trusted=True,
    )
    app = AppState()
    registry = ToolRegistry()
    register_builtin(registry)

    memory = MemoryStore(settings.project_root, settings.workspace)
    skills = SkillLoader(settings.project_root, settings.workspace)
    hooks = HookEngine.from_path(settings.hooks_config, trusted=True)
    llm = GeminiClient(settings.gemini_keys, settings.models)
    mcp = McpHub()

    await asyncio.gather(
        mcp.start(settings.mcp_config),
        asyncio.sleep(0),
    )
    await mcp.attach_tools_async(registry)

    return Engine(
        settings=settings,
        session=session,
        app=app,
        registry=registry,
        llm=llm,
        hooks=hooks,
        memory=memory,
        skills=skills,
        mcp=mcp,
    )
