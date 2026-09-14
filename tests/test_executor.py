import pytest

from agent.tools.base import Tool, ToolResult
from agent.tools.executor import execute_batches, partition_calls
from agent.tools.pipeline import PreparedCall


class _Safe(Tool):
    name = "safe"

    def is_concurrency_safe(self, arguments):
        return True

    async def run(self, arguments, ctx):
        return ToolResult(ok=True, output="ok")


class _Unsafe(Tool):
    name = "unsafe"

    def is_concurrency_safe(self, arguments):
        return False

    async def run(self, arguments, ctx):
        return ToolResult(ok=True, output="serial")


def test_partition_groups_safe_then_isolates_unsafe():
    safe = _Safe()
    unsafe = _Unsafe()
    calls = [
        PreparedCall("1", "safe", {}, safe),
        PreparedCall("2", "safe", {}, safe),
        PreparedCall("3", "unsafe", {}, unsafe),
        PreparedCall("4", "safe", {}, safe),
    ]
    batches = partition_calls(calls)
    assert len(batches) == 3
    assert [c.call_id for c in batches[0]] == ["1", "2"]
    assert [c.call_id for c in batches[1]] == ["3"]
    assert [c.call_id for c in batches[2]] == ["4"]


class _Ctx:
    workspace = None
    session = None
    llm = None
    hooks = None
    memory = None
    skills = None

    async def emit(self, event):
        return None


@pytest.mark.asyncio
async def test_execute_preserves_order():
    safe = _Safe()
    unsafe = _Unsafe()
    calls = [
        PreparedCall("1", "safe", {}, safe),
        PreparedCall("2", "unsafe", {}, unsafe),
        PreparedCall("3", "safe", {}, safe),
    ]
    results = await execute_batches(calls, _Ctx(), 1000)
    assert [r.output for r in results] == ["ok", "serial", "ok"]
