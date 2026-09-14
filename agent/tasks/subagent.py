from __future__ import annotations

from typing import Any

from agent.loop import run_query
from agent.state import AppState, SessionState
from agent.tools.registry import ToolRegistry


def child_registry(parent: Any, agent_type: str = "explore") -> ToolRegistry:
    allow = {
        "read_file",
        "glob_files",
        "grep",
        "web_search",
        "fetch_url",
        "analyze_csv",
        "system_info",
        "load_skill",
    }
    if agent_type == "worker":
        allow |= {"write_file", "edit_file", "shell"}

    registry = ToolRegistry()
    for tool in parent.registry.all():
        if tool.name in allow:
            registry.register(tool)
            continue
        if not tool.name.startswith("mcp_"):
            continue
        try:
            mcp_ok = agent_type == "worker" or tool.is_read_only({})
        except Exception:
            mcp_ok = False
        if mcp_ok:
            registry.register(tool)
    return registry


async def run_subagent(parent: Any, goal: str, agent_type: str = "explore") -> str:
    child = _ChildEngine(parent, child_registry(parent, agent_type))
    child.session.permission_mode = "dont_ask"
    child.session.max_turns = 8
    texts: list[str] = []
    async for event in run_query(child, goal, is_subagent=True):
        kind = getattr(event, "kind", "")
        if kind in {"tool_started", "tool_finished", "status"}:
            await parent.emit(event)
        if kind == "assistant":
            texts.append(event.text)
        if kind == "terminal":
            if event.text:
                texts.append(event.text)
            if event.error:
                texts.append(f"error: {event.error}")
    parent.session.file_reads.update(child.session.file_reads)
    return "\n".join(t for t in texts if t).strip() or "(sub-agent produced no text)"


class _ChildEngine:
    def __init__(self, parent: Any, registry: ToolRegistry) -> None:
        self.settings = parent.settings
        self.session = SessionState(
            workspace=parent.session.workspace,
            project_root=parent.session.project_root,
            permission_mode="dont_ask",
            max_turns=8,
            max_output_tokens=parent.session.max_output_tokens,
            trusted=parent.session.trusted,
        )
        self.session.file_reads = dict(parent.session.file_reads)
        self.app = AppState()
        self.registry = registry
        self.llm = parent.llm
        self.hooks = parent.hooks
        self.memory = parent.memory
        self.skills = parent.skills
        self.workspace = parent.workspace
        self.mcp = parent.mcp

        async def _sink(_event) -> None:
            return None

        self._sink = _sink
