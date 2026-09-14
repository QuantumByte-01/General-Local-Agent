from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from agent.events import Event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gla",
        description="General Local Agent — local tool-using loop (LLM stays in the loop).",
    )
    parser.add_argument("-p", "--print", dest="prompt", help="Run one task in the terminal and exit")
    parser.add_argument(
        "--mode",
        choices=["plan", "default", "accept_edits", "dont_ask"],
        default=os.getenv("AGENT_PERMISSION_MODE", "default") or "default",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-approve permission prompts")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per event")
    parser.add_argument("--ui", action="store_true", help="Force the Gradio UI even if -p is set")
    return parser


def format_cli_event(event: Event) -> str | None:
    kind = getattr(event, "kind", "")
    if kind == "status":
        return f"… {event.text}"
    if kind == "text_delta":
        return event.text or ""
    if kind == "tool_started":
        spec = " (speculative)" if getattr(event, "speculative", False) else ""
        return f"\n→ {event.name}{spec}"
    if kind == "tool_finished":
        flag = "ok" if event.ok else "err"
        clip = (event.output or "")[:400].replace("\n", " ")
        return f"\n← {flag} {event.name} ({event.duration_ms}ms) {clip}"
    if kind == "permission_request":
        return f"\n! permission needed\n{event.reason}\nType yes / no, or pass -y."
    if kind == "assistant":
        return f"\n{event.text}"
    if kind == "terminal":
        if event.error:
            return f"\nerror ({event.reason}): {event.error}"
        if event.reason not in {"completed"} and event.text:
            return f"\n[{event.reason}] {event.text}"
        return None
    return None


def _event_payload(event: Event) -> dict[str, Any]:
    data = {"kind": getattr(event, "kind", "")}
    for key in ("text", "name", "ok", "output", "reason", "error", "duration_ms", "call_id"):
        if hasattr(event, key):
            val = getattr(event, key)
            if val not in (None, "", []):
                data[key] = val
    return data


async def print_query(engine, prompt: str, *, auto_yes: bool, as_json: bool) -> int:
    from agent.loop import run_query
    from agent.permissions import is_affirmative

    pending_goal: str | None = prompt
    resume: str | None = None
    code = 0
    while True:
        saw_delta = False
        finished = False
        async for event in run_query(engine, pending_goal, resume=resume):
            kind = getattr(event, "kind", "")
            if kind == "text_delta":
                saw_delta = True
            skip_assistant = kind == "assistant" and saw_delta and not as_json
            if as_json:
                print(json.dumps(_event_payload(event), default=str), flush=True)
            elif not skip_assistant:
                piece = format_cli_event(event)
                if piece:
                    end = "" if kind == "text_delta" else "\n"
                    print(piece, end=end, flush=True)
            if kind == "terminal":
                finished = True
                if event.reason not in {"completed"}:
                    code = 1
            if kind == "permission_request":
                if auto_yes:
                    pending_goal = None
                    resume = "yes"
                else:
                    try:
                        answer = input("allow? [yes/no] ").strip()
                    except EOFError:
                        answer = "no"
                    decision = is_affirmative(answer)
                    if decision is None:
                        pending_goal = answer or None
                        resume = None
                    else:
                        pending_goal = None
                        resume = "yes" if decision else "no"
            if kind == "status":
                saw_delta = False
        if finished and not engine.app.pending_calls:
            return code
        if getattr(engine.session, "aborted", False) and not engine.app.pending_calls:
            return 1
        if not engine.app.pending_calls and not finished:
            return code


async def _aclose_engine(engine) -> None:
    mcp = getattr(engine, "mcp", None)
    if mcp is not None and hasattr(mcp, "aclose"):
        try:
            await mcp.aclose()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.ui or not args.prompt:
        from agent.ui import main as ui_main

        ui_main()
        return 0

    async def _run() -> int:
        from agent.bootstrap import bootstrap

        engine = await bootstrap()
        if args.mode in {"plan", "default", "accept_edits", "dont_ask"}:
            engine.session.permission_mode = args.mode  # type: ignore[assignment]
            engine._system_prompt = None
        try:
            return await print_query(engine, args.prompt, auto_yes=args.yes, as_json=args.json)
        finally:
            await _aclose_engine(engine)

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
