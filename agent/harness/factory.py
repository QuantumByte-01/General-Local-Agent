from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import Settings
from agent.harness.scripted import ScriptedLLM
from agent.hooks.engine import HookEngine
from agent.memory.store import MemoryStore
from agent.permissions import PermissionMode
from agent.runtime import Engine
from agent.skills.loader import SkillLoader
from agent.state import AppState, SessionState
from agent.tools.builtin import register_builtin
from agent.tools.registry import ToolRegistry


def make_engine(
    workspace: Path,
    *,
    llm: Any | None = None,
    permission_mode: PermissionMode = "dont_ask",
    hooks: dict | None = None,
    max_turns: int = 8,
    compact_chars: int = 80_000,
    project_root: Path | None = None,
    skills_root: Path | None = None,
    turns: list | None = None,
    mcp: bool = False,
) -> Engine:
    """Build a real Engine against a temp workspace, no live Gemini."""
    project_root = project_root or workspace
    skills_root = skills_root or project_root
    workspace.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        workspace=workspace.resolve(),
        project_root=project_root.resolve(),
        permission_mode=permission_mode,
        max_turns=max_turns,
        compact_chars=compact_chars,
        mcp_config=None,
        hooks_config=None,
        stream=True,
        thinking_budget=0,
    )
    session = SessionState(
        workspace=settings.workspace,
        project_root=settings.project_root,
        permission_mode=permission_mode,
        max_turns=max_turns,
        max_output_tokens=settings.max_output_tokens,
        trusted=True,
    )
    registry = ToolRegistry()
    register_builtin(registry)
    hub = None
    if mcp:
        from agent.mcp.inprocess import attach_inprocess

        hub = attach_inprocess(registry)
    if llm is None:
        llm = ScriptedLLM(list(turns or []))
    return Engine(
        settings=settings,
        session=session,
        app=AppState(),
        registry=registry,
        llm=llm,
        hooks=HookEngine(hooks or {}, trusted=True),
        memory=MemoryStore(project_root, workspace),
        skills=SkillLoader(skills_root, workspace),
        mcp=hub,
    )
