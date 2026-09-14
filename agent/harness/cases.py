from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent.harness.factory import make_engine
from agent.harness.runner import Trace, run_harness
from agent.harness.scripted import ScriptedLLM
from agent.llm.client import LlmTurn
from agent.permissions import PermissionMode


def _turn(item: dict[str, Any]) -> LlmTurn:
    calls = []
    for i, call in enumerate(item.get("tool_calls") or []):
        calls.append(
            {
                "id": call.get("id") or f"c{i}",
                "name": call["name"],
                "arguments": call.get("arguments") or {},
            }
        )
    return LlmTurn(
        text=str(item.get("text") or ""),
        tool_calls=calls,
        model="scripted",
        truncated=bool(item.get("truncated")),
    )


def load_case(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"scenario must be a mapping: {path}")
    data["_path"] = str(path)
    return data


def seed_workspace(root: Path, files: dict[str, str] | None) -> None:
    for rel, body in (files or {}).items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")


def score(trace: Trace, expect: dict[str, Any], workspace: Path) -> list[str]:
    failures: list[str] = []
    if "terminal" in expect:
        reason = trace.terminal.reason if trace.terminal else None
        if reason != expect["terminal"]:
            failures.append(f"terminal {reason!r} != {expect['terminal']!r}")
    if expect.get("pending") and not trace.pending:
        failures.append("expected a permission pause")
    if expect.get("pending") is False and trace.pending:
        failures.append("did not expect a permission pause")
    if "tools" in expect:
        got = trace.tool_names
        want = list(expect["tools"])
        if got != want:
            failures.append(f"tools {got} != {want}")
    if "tools_contains" in expect:
        for name in expect["tools_contains"]:
            if name not in trace.tool_names:
                failures.append(f"missing tool {name}")
    if "answer_contains" in expect:
        needle = str(expect["answer_contains"])
        if needle.lower() not in trace.assistant_text.lower():
            failures.append(f"answer missing {needle!r}")
    for rel, body in (expect.get("files") or {}).items():
        path = workspace / rel
        if not path.exists():
            failures.append(f"missing file {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if body not in text and text != body:
            failures.append(f"file {rel} did not contain expected body")
    for rel in expect.get("files_absent") or []:
        if (workspace / rel).exists():
            failures.append(f"file {rel} should not exist")
    return failures


async def run_case(case: dict[str, Any], workspace: Path) -> tuple[Trace, list[str]]:
    seed_workspace(workspace, case.get("workspace_files"))
    llm = ScriptedLLM([_turn(item) for item in case.get("script") or []])
    mode: PermissionMode = case.get("mode") or "dont_ask"  # type: ignore[assignment]
    repo_root = Path(__file__).resolve().parents[2]
    engine = make_engine(
        workspace,
        llm=llm,
        permission_mode=mode,
        hooks=case.get("hooks"),
        max_turns=int(case.get("max_turns") or 8),
        project_root=workspace,
        skills_root=repo_root,
        mcp=bool(case.get("mcp")),
    )
    resumes = []
    for item in case.get("resumes") or []:
        if item is True:
            resumes.append("yes")
        elif item is False:
            resumes.append("no")
        else:
            resumes.append(str(item))
    trace = await run_harness(
        engine,
        str(case.get("user") or "do the task"),
        resumes=resumes,
    )
    failures = score(trace, case.get("expect") or {}, workspace)
    return trace, failures
