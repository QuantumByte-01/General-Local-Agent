from pathlib import Path

from agent.hooks.engine import HookEngine


def test_snapshot_ignores_later_file_writes(tmp_path: Path):
    path = tmp_path / "hooks.json"
    path.write_text(
        '{"hooks":{"PreToolUse":[{"matcher":"shell","block":true,"message":"nope"}]}}',
        encoding="utf-8",
    )
    engine = HookEngine.from_path(path, trusted=True)
    hit = engine.pre_tool_use("shell", {"command": "echo hi"})
    assert hit.block is True
    path.write_text('{"hooks":{"PreToolUse":[]}}', encoding="utf-8")
    hit2 = engine.pre_tool_use("shell", {"command": "echo hi"})
    assert hit2.block is True


def test_untrusted_skips_hooks(tmp_path: Path):
    path = tmp_path / "hooks.json"
    path.write_text(
        '{"hooks":{"PreToolUse":[{"matcher":"*","block":true,"message":"nope"}]}}',
        encoding="utf-8",
    )
    engine = HookEngine.from_path(path, trusted=False)
    assert engine.pre_tool_use("write_file", {}).block is False


def test_deny_beats_allow():
    engine = HookEngine(
        {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "shell", "permissionBehavior": "allow"},
                    {"matcher": "shell", "permissionBehavior": "deny"},
                ]
            }
        },
        trusted=True,
    )
    hit = engine.pre_tool_use("shell", {"command": "echo hi"})
    assert hit.decision == "deny"


def test_stop_once_only_blocks_first_time():
    engine = HookEngine(
        {"hooks": {"Stop": [{"block": True, "once": True, "message": "again"}]}},
        trusted=True,
    )
    first = engine.stop("done")
    second = engine.stop("done")
    assert first.block is True
    assert second.block is False
