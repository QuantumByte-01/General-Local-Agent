from __future__ import annotations

import asyncio
import os

import gradio as gr
from dotenv import load_dotenv

from agent.bootstrap import bootstrap
from agent.loop import run_query
from agent.state import AppState

load_dotenv()

_ENGINE = None
_LOCK = asyncio.Lock()


async def get_engine():
    global _ENGINE
    async with _LOCK:
        if _ENGINE is None:
            _ENGINE = await bootstrap()
        return _ENGINE


def _format_event(event) -> str | None:
    kind = getattr(event, "kind", "")
    if kind == "status":
        return f"_{event.text}_"
    if kind == "tool_started":
        spec = " speculatively" if event.speculative else ""
        return f"**tool** `{event.name}`{spec}"
    if kind == "tool_finished":
        flag = "ok" if event.ok else "err"
        clip = (event.output or "")[:500]
        return f"**{flag}** `{event.name}` ({event.duration_ms}ms)\n```\n{clip}\n```"
    if kind == "permission_request":
        return f"**permission needed**\n{event.reason}\n\nReply **yes** or **no**."
    if kind == "assistant":
        return event.text
    if kind == "terminal" and event.error:
        return f"**error:** {event.error}"
    return None


async def agent_turn(user_msg: str, history: list, mode: str, log: str):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        yield history or [], log or "idle", ""
        return

    engine = await get_engine()
    if mode in {"plan", "default", "accept_edits", "dont_ask"}:
        engine.session.permission_mode = mode  # type: ignore[assignment]

    history = list(history or [])
    history.append({"role": "user", "content": user_msg})
    live_log = log or ""

    resume = user_msg if engine.app.pending_calls else None
    goal = None if resume else user_msg

    async for event in run_query(engine, goal, resume=resume):
        piece = _format_event(event)
        if event.kind in {"status", "tool_started", "tool_finished"}:
            if piece:
                live_log = (live_log + "\n" + piece).strip()
                yield history, live_log, ""
            continue
        if event.kind == "permission_request":
            history.append({"role": "assistant", "content": piece or event.reason})
            yield history, live_log, ""
            continue
        if event.kind == "assistant":
            history.append({"role": "assistant", "content": event.text})
            yield history, live_log, ""
            continue
        if event.kind == "terminal":
            if event.error:
                history.append(
                    {"role": "assistant", "content": f"Stopped ({event.reason}): {event.error}"}
                )
            elif event.reason not in {"completed"} and event.text:
                if not history or history[-1].get("content") != event.text:
                    history.append({"role": "assistant", "content": event.text})
            yield history, live_log, ""


async def reset_session():
    engine = await get_engine()
    engine.app = AppState()
    engine.session.file_reads.clear()
    engine.session.aborted = False
    return [], "Session cleared.", ""


async def banner_text():
    engine = await get_engine()
    return engine.status_summary()


def main() -> None:
    workspace = os.getenv("AGENT_BASE_DIR") or os.getcwd()
    with gr.Blocks(title="General Local Agent", fill_height=True) as demo:
        gr.Markdown(
            f"# General Local Agent\n"
            f"Local tool-using loop · workspace `{workspace}` · "
            f"[architecture](docs/ARCHITECTURE.md)"
        )
        banner = gr.Markdown("Starting…")
        with gr.Row():
            chatbot = gr.Chatbot(label="Agent", height=480, type="messages")
            trace = gr.Textbox(label="Trace", value="idle", lines=18, interactive=False)
        user_in = gr.Textbox(
            label="Request",
            placeholder="Summarize AGENT.md, search the web, or edit a file in the workspace…",
            lines=2,
        )
        mode = gr.Radio(
            choices=["plan", "default", "accept_edits", "dont_ask"],
            value="default",
            label="Permission mode",
            info="plan = read-only · default = confirm writes/shell · accept_edits = auto file I/O · dont_ask = auto all (logged)",
        )
        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")
            reset_btn = gr.Button("New session")
        gr.Examples(
            examples=[
                ["Read AGENT.md and explain how the query loop works"],
                ["What is using CPU and RAM on this machine?"],
                ["Search the web for Model Context Protocol stdio transport and cite sources"],
            ],
            inputs=user_in,
        )

        run_btn.click(
            fn=agent_turn,
            inputs=[user_in, chatbot, mode, trace],
            outputs=[chatbot, trace, user_in],
        )
        user_in.submit(
            fn=agent_turn,
            inputs=[user_in, chatbot, mode, trace],
            outputs=[chatbot, trace, user_in],
        )
        reset_btn.click(fn=reset_session, outputs=[chatbot, trace, user_in])
        demo.load(banner_text, outputs=banner)

    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("AGENT_PORT", "7869"))
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
