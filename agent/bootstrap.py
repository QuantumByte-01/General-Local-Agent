from __future__ import annotations

import asyncio
from pathlib import Path

from agent.config import load_settings
from agent.hooks.engine import HookEngine
from agent.llm.client import GeminiClient
from agent.mcp.client import McpHub
from agent.memory.store import MemoryStore
from agent.permissions import PermissionMode
from agent.runtime import Engine
from agent.skills.loader import SkillLoader
from agent.state import AppState, SessionState
from agent.tools.builtin import register_builtin
from agent.tools.registry import ToolRegistry


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
    registry = ToolRegistry()
    register_builtin(registry)

    def _sync_local():
        memory = MemoryStore(settings.project_root, settings.workspace)
        skills = SkillLoader(settings.project_root, settings.workspace)
        hooks = HookEngine.from_path(settings.hooks_config, trusted=True)
        return memory, skills, hooks

    llm = GeminiClient(
        settings.gemini_keys,
        settings.models,
        thinking_budget=settings.thinking_budget,
        stream=settings.stream,
        timeout_ms=settings.llm_timeout_ms,
    )
    mcp = McpHub()
    (memory, skills, hooks), _mcp_ok, _warm = await asyncio.gather(
        asyncio.to_thread(_sync_local),
        mcp.start(settings.mcp_config),
        llm.warmup(),
    )
    await mcp.attach_tools_async(registry)
    registry.schemas_for_llm()

    return Engine(
        settings=settings,
        session=session,
        app=AppState(),
        registry=registry,
        llm=llm,
        hooks=hooks,
        memory=memory,
        skills=skills,
        mcp=mcp,
    )
