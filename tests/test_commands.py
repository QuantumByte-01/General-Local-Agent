import pytest

from agent.cli import build_parser, format_cli_event
from agent.commands import apply_command, parse_slash
from agent.events import Status, TextDelta
from agent.harness.factory import make_engine
from agent.harness.runner import run_harness
from agent.harness.scripted import ScriptedLLM
from agent.llm.client import LlmTurn
from agent.runtime import reset_runtime
from agent.tools.files import content_hash


def test_parse_slash():
    assert parse_slash("/help") == ("help", "")
    assert parse_slash("/mode plan") == ("mode", "plan")
    assert parse_slash("not a command") is None


def test_cli_parser_print_mode():
    args = build_parser().parse_args(["-p", "summarize AGENT.md", "--mode", "plan", "-y"])
    assert args.prompt == "summarize AGENT.md"
    assert args.mode == "plan"
    assert args.yes is True


def test_format_cli_event():
    assert "hello" in (format_cli_event(Status(text="hello")) or "")
    assert format_cli_event(TextDelta(text="tok")) == "tok"


@pytest.mark.asyncio
async def test_slash_status_and_clear(tmp_path):
    engine = make_engine(tmp_path, llm=ScriptedLLM([LlmTurn(text="hi")]))
    result = apply_command(engine, "/status")
    assert result and result.handled and "tools" in result.message
    engine._system_prompt = "cached"
    reset_runtime(engine)
    assert engine._system_prompt is None
    assert engine.app.messages == []


@pytest.mark.asyncio
async def test_write_records_content_hash(tmp_path):
    llm = ScriptedLLM(
        [
            LlmTurn(
                text="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "write_file",
                        "arguments": {"path": "a.txt", "content": "hello"},
                    }
                ],
            ),
            LlmTurn(text="wrote it"),
        ]
    )
    engine = make_engine(tmp_path, llm=llm, permission_mode="dont_ask")
    await run_harness(engine, "write a.txt")
    path = tmp_path / "a.txt"
    rec = engine.session.file_reads[str(path.resolve())]
    assert rec.excerpt_hash == content_hash("hello")


@pytest.mark.asyncio
async def test_new_request_cancels_pending(tmp_path):
    llm = ScriptedLLM(
        [
            LlmTurn(
                text="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "write_file",
                        "arguments": {"path": "notes.md", "content": "hi"},
                    }
                ],
            ),
            LlmTurn(text="hello instead"),
        ]
    )
    engine = make_engine(tmp_path, llm=llm, permission_mode="default")
    from agent.harness.runner import run_harness as rh

    trace = await rh(engine, "write notes", resumes=["just say hello instead"])
    assert trace.terminal and trace.terminal.reason == "completed"
    assert not (tmp_path / "notes.md").exists()
    assert "hello instead" in trace.assistant_text


@pytest.mark.asyncio
async def test_todo_tool(tmp_path):
    llm = ScriptedLLM(
        [
            LlmTurn(
                text="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "todo",
                        "arguments": {
                            "items": [
                                {"id": "1", "content": "read AGENT.md", "status": "in_progress"},
                                {"id": "2", "content": "summarize", "status": "pending"},
                            ]
                        },
                    }
                ],
            ),
            LlmTurn(text="tracking two steps"),
        ]
    )
    engine = make_engine(tmp_path, llm=llm)
    trace = await run_harness(engine, "plan the work")
    assert engine.app.todos[0]["status"] == "in_progress"
    assert "tracking" in trace.assistant_text
