import pytest

from agent.mcp.client import McpHub, McpTool, _safe_name
from agent.mcp.inprocess import InProcessSession, attach_inprocess, harness_echo_tools
from agent.tasks.subagent import child_registry
from agent.tools.builtin import register_builtin
from agent.tools.registry import ToolRegistry


def test_safe_name():
    assert _safe_name("my-server", "do.thing") == "mcp_my_server_do_thing"


@pytest.mark.asyncio
async def test_inprocess_echo_and_add():
    registry = ToolRegistry()
    hub = attach_inprocess(registry)
    echo = registry.get("mcp_harness_echo")
    add = registry.get("mcp_harness_add")
    assert isinstance(echo, McpTool)
    assert echo.is_concurrency_safe({})
    out = await echo.run({"text": "ping"}, None)  # type: ignore[arg-type]
    assert out.ok and out.output == "ping"
    summed = await add.run({"a": 2, "b": 3}, None)  # type: ignore[arg-type]
    assert summed.ok and summed.output == "5"
    assert "harness" in hub.sessions


@pytest.mark.asyncio
async def test_hub_attaches_inprocess_session():
    registry = ToolRegistry()
    hub = McpHub()
    hub.sessions["harness"] = InProcessSession(harness_echo_tools())
    names = await hub.attach_tools_async(registry)
    assert sorted(names) == ["mcp_harness_add", "mcp_harness_echo"]
    echo = registry.get("mcp_harness_echo")
    out = await echo.run({"text": "ok"}, None)  # type: ignore[arg-type]
    assert out.ok and out.output == "ok"


def test_explore_subagent_keeps_readonly_mcp():
    parent_registry = ToolRegistry()
    register_builtin(parent_registry)
    attach_inprocess(parent_registry)

    class _Parent:
        registry = parent_registry

    child = child_registry(_Parent(), "explore")
    assert child.get("mcp_harness_echo") is not None
    assert child.get("read_file") is not None
    assert child.get("write_file") is None
