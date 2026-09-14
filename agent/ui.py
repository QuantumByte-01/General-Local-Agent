from __future__ import annotations

import asyncio
import os

import gradio as gr
from dotenv import load_dotenv

from agent.bootstrap import bootstrap
from agent.commands import apply_command
from agent.loop import run_query
from agent.runtime import reset_runtime

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
        return (
            f"**permission needed**\n{event.reason}\n\n"
            "Use **Allow** / **Deny**, type yes/no, or send a new request to cancel."
        )
    if kind == "assistant":
        return event.text
    if kind == "terminal" and event.error:
        return f"**error:** {event.error}"
    return None


async def _consume_query(engine, history, live_log, goal, resume):
    history = list(history or [])
    try:
        async for event in run_query(engine, goal, resume=resume):
            piece = _format_event(event)
            if event.kind in {"status", "tool_started", "tool_finished"}:
                if piece:
                    live_log = (live_log + "\n" + piece).strip()
                    yield history, live_log, ""
                continue
            if event.kind == "text_delta":
                chunk = event.text or ""
                if history and history[-1].get("role") == "assistant":
                    history[-1] = {
                        "role": "assistant",
                        "content": (history[-1].get("content") or "") + chunk,
                    }
                else:
                    history.append({"role": "assistant", "content": chunk})
                yield history, live_log, ""
                continue
            if event.kind == "permission_request":
                history.append({"role": "assistant", "content": piece or event.reason})
                yield history, live_log, ""
                continue
            if event.kind == "assistant":
                if history and history[-1].get("role") == "assistant":
                    history[-1] = {"role": "assistant", "content": event.text}
                else:
                    history.append({"role": "assistant", "content": event.text})
                yield history, live_log, ""
                continue
            if event.kind == "terminal":
                if event.error:
                    history.append(
                        {"role": "assistant", "content": f"Stopped ({event.reason}): {event.error}"}
                    )
                elif event.reason == "aborted":
                    history.append({"role": "assistant", "content": "Aborted."})
                elif event.reason not in {"completed"} and event.text:
                    if not history or history[-1].get("content") != event.text:
                        history.append({"role": "assistant", "content": event.text})
                yield history, live_log, ""
    except asyncio.CancelledError:
        engine.session.aborted = True
        history.append({"role": "assistant", "content": "Aborted."})
        yield history, live_log, ""
        raise


async def agent_turn(user_msg: str, history: list, mode: str, log: str):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        yield history or [], log or "idle", ""
        return

    engine = await get_engine()
    if mode in {"plan", "default", "accept_edits", "dont_ask"}:
        if engine.session.permission_mode != mode:
            engine._system_prompt = None
            engine._prompt_sig = None
        engine.session.permission_mode = mode  # type: ignore[assignment]
    engine.session.aborted = False

    history = list(history or [])
    live_log = log or ""

    cmd = apply_command(engine, user_msg)
    if cmd is not None:
        if cmd.clear:
            reset_runtime(engine)
            yield [], "Session cleared.", ""
            return
        if cmd.abort:
            engine.session.aborted = True
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": cmd.message or "Aborted."})
            yield history, live_log, ""
            return
        if cmd.resume:
            history.append({"role": "user", "content": user_msg})
            async for state in _consume_query(engine, history, live_log, None, cmd.resume):
                yield state
            return
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": cmd.message})
        yield history, live_log, ""
        return

    history.append({"role": "user", "content": user_msg})
    if engine.app.pending_calls:
        goal, resume = None, user_msg
    else:
        goal, resume = user_msg, None

    async for state in _consume_query(engine, history, live_log, goal, resume):
        yield state


async def allow_turn(history: list, mode: str, log: str):
    async for state in agent_turn("yes", history, mode, log):
        yield state


async def deny_turn(history: list, mode: str, log: str):
    async for state in agent_turn("no", history, mode, log):
        yield state


async def abort_run(history: list, log: str):
    engine = await get_engine()
    engine.session.aborted = True
    history = list(history or [])
    history.append({"role": "assistant", "content": "Abort requested."})
    return history, log or "idle", ""


async def reset_session():
    engine = await get_engine()
    reset_runtime(engine)
    return [], "Session cleared.", ""


async def banner_text():
    engine = await get_engine()
    return engine.status_summary()


async def shutdown_engine():
    engine = _ENGINE
    if engine is None:
        return
    mcp = getattr(engine, "mcp", None)
    if mcp is not None and hasattr(mcp, "aclose"):
        try:
            await mcp.aclose()
        except Exception:
            pass


def main() -> None:
    workspace = os.getenv("AGENT_BASE_DIR") or os.getcwd()
    with gr.Blocks(title="General Local Agent", fill_height=True) as demo:
        gr.Markdown(
            f"# General Local Agent\n"
            f"Local tool-using loop · workspace `{workspace}` · "
            f"slash commands: `/help` · "
            f"[architecture](docs/ARCHITECTURE.md)"
        )
        banner = gr.Markdown("Starting…")
        with gr.Row():
            chatbot = gr.Chatbot(label="Agent", height=480, type="messages")
            trace = gr.Textbox(label="Trace", value="idle", lines=18, interactive=False)
        user_in = gr.Textbox(
            label="Request",
            placeholder="Ask something, or /help · /mode plan · /compact · /clear",
            lines=2,
        )
        mode = gr.Radio(
            choices=["plan", "default", "accept_edits", "dont_ask"],
            value=os.getenv("AGENT_PERMISSION_MODE", "default") or "default",
            label="Permission mode",
            info="plan = read-only · default = confirm writes/shell · accept_edits = auto file I/O · dont_ask = auto all (logged)",
        )
        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")
            allow_btn = gr.Button("Allow")
            deny_btn = gr.Button("Deny")
            abort_btn = gr.Button("Abort")
            reset_btn = gr.Button("New session")
        gr.Examples(
            examples=[
                ["Read AGENT.md and explain how the query loop works"],
                ["What is using CPU and RAM on this machine?"],
                ["Search the web for Model Context Protocol stdio transport and cite sources"],
                ["/help"],
            ],
            inputs=user_in,
        )

        outputs = [chatbot, trace, user_in]
        run_evt = run_btn.click(fn=agent_turn, inputs=[user_in, chatbot, mode, trace], outputs=outputs)
        submit_evt = user_in.submit(fn=agent_turn, inputs=[user_in, chatbot, mode, trace], outputs=outputs)
        allow_btn.click(fn=allow_turn, inputs=[chatbot, mode, trace], outputs=outputs)
        deny_btn.click(fn=deny_turn, inputs=[chatbot, mode, trace], outputs=outputs)
        abort_btn.click(fn=abort_run, inputs=[chatbot, trace], outputs=outputs, cancels=[run_evt, submit_evt])
        reset_btn.click(fn=reset_session, outputs=outputs)
        demo.load(banner_text, outputs=banner)

    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("AGENT_PORT", "7869"))
    try:
        demo.launch(server_name=host, server_port=port)
    finally:
        try:
            asyncio.run(shutdown_engine())
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
