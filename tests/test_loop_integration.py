import pytest

from agent.harness.factory import make_engine
from agent.harness.runner import run_harness
from agent.harness.scripted import ScriptedLLM
from agent.llm.client import LlmTurn
from agent.tools.paths import resolve_in_workspace


@pytest.mark.asyncio
async def test_read_write_roundtrip(tmp_path):
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    llm = ScriptedLLM(
        [
            LlmTurn(
                text="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "read_file",
                        "arguments": {"path": "a.txt"},
                    }
                ],
            ),
            LlmTurn(
                text="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "write_file",
                        "arguments": {"path": "b.txt", "content": "two"},
                    }
                ],
            ),
            LlmTurn(text="Copied one to b.txt as two"),
        ]
    )
    engine = make_engine(tmp_path, llm=llm, permission_mode="dont_ask")
    trace = await run_harness(engine, "copy a to b")
    assert trace.terminal and trace.terminal.reason == "completed"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "two"
    assert engine.session.model == "scripted"


@pytest.mark.asyncio
async def test_streams_text_deltas(tmp_path):
    llm = ScriptedLLM([LlmTurn(text="hello from script")])
    engine = make_engine(tmp_path, llm=llm, permission_mode="dont_ask")
    trace = await run_harness(engine, "hi")
    assert "text_delta" in trace.kinds()
    assert "hello from script" in trace.assistant_text


def test_workspace_escape(tmp_path):
    with pytest.raises(PermissionError):
        resolve_in_workspace(tmp_path, "..\\..\\Windows\\notepad.exe")
