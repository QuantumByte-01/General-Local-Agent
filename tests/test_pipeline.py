from agent.tools.builtin.read import ReadTool
from agent.tools.pipeline import prepare_call
from agent.tools.registry import ToolRegistry


def test_unknown_tool():
    registry = ToolRegistry()
    prepared = prepare_call(registry, "1", "nope", {})
    assert prepared.error
    assert prepared.tool is None


def test_missing_required():
    registry = ToolRegistry()
    registry.register(ReadTool())
    prepared = prepare_call(registry, "1", "read_file", {})
    assert prepared.error and "path" in prepared.error
