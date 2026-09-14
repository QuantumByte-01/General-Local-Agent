from pathlib import Path

import pytest

from agent.tools.paths import resolve_in_workspace


def test_relative_stays_inside(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("hi", encoding="utf-8")
    resolved = resolve_in_workspace(tmp_path, "a.txt", must_exist=True)
    assert resolved == target.resolve()


def test_escape_rejected(tmp_path: Path):
    with pytest.raises(PermissionError):
        resolve_in_workspace(tmp_path, "..\\..\\Windows\\System32")
