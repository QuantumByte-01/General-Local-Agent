from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent.compact import autocompact, estimate_chars
from agent.events import (
    AssistantMessage,
    PermissionRequest,
    Status,
    Terminal,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from agent.permissions import is_affirmative
from agent.prompts import build_system_prompt, stop_continuation_prompt
from agent.state import ChatMessage
from agent.tools.pipeline import PreparedCall, permission_for, prepare_call
from agent.tools.executor import execute_batches


async def run_query(
    engine: Any,
    user_text: str | None,
    *,
    resume: str | None = None,
    is_subagent: bool = False,
) -> AsyncIterator[Any]:
    session = engine.session
    app = engine.app
    settings = engine.settings

    if resume and app.pending_calls:
        decision = is_affirmative(resume)
        if decision is None:
            yield PermissionRequest(
                call_id=app.pending_calls[0]["id"],
                name=app.pending_calls[0]["name"],
                input=app.pending_calls[0].get("arguments") or {},
                reason="Reply yes to allow these tool calls, or no to deny.",
            )
            return
        prepared = [
            prepare_call(engine.registry, c["id"], c["name"], c.get("arguments") or {})
            for c in app.pending_calls
        ]
        if decision:
            async for event in _run_tools(engine, prepared):
                yield event
        else:
            for call in prepared:
                app.messages.append(
                    ChatMessage(
                        role="tool",
                        content="User denied this tool call.",
                        tool_call_id=call.call_id,
                        tool_name=call.name,
                    )
                )
            yield Status(text="Denied pending tool calls.")
        app.pending_calls = []
        app.pending_reason = ""
    elif user_text:
        app.messages.append(ChatMessage(role="user", content=user_text))

    if resume and getattr(engine, "_system_prompt", None):
        system = engine._system_prompt
    else:
        recall_query = user_text or (app.messages[-1].content if app.messages else "")
        memory_excerpt = engine.memory.keyword_recall(recall_query, session.session_id)
        agent_md = ""
        agent_path = engine.workspace / "AGENT.md"
        if agent_path.exists():
            agent_md = agent_path.read_text(encoding="utf-8", errors="replace")[:6000]
        mcp_names = [t.name for t in engine.registry.all() if t.name.startswith("mcp_")]
        system = build_system_prompt(
            workspace=engine.workspace,
            permission_mode=session.permission_mode,
            skill_menu=engine.skills.menu(),
            memory_excerpt=memory_excerpt,
            mcp_tools=mcp_names,
            agent_md=agent_md,
            is_subagent=is_subagent,
        )
        engine._system_prompt = system

    tool_defs = engine.registry.schemas_for_llm()

    turns = 0
    output_tokens_cap = session.max_output_tokens
    while turns < session.max_turns:
        if session.aborted:
            yield Terminal(reason="aborted", text="Aborted.")
            return
        turns += 1
        if estimate_chars(app.messages) > settings.compact_chars:
            app.messages = autocompact(app.messages, settings.compact_chars)
            yield Status(text="Context compacted.")

        yield Status(text=f"Model turn {turns}…")
        try:
            llm_turn = None
            streamer = getattr(engine.llm, "stream_generate", None)
            use_stream = callable(streamer) and getattr(settings, "stream", True)
            if use_stream:
                async for kind, payload in streamer(
                    system=system,
                    messages=app.messages,
                    tool_defs=tool_defs,
                    max_output_tokens=output_tokens_cap,
                ):
                    if kind == "text" and payload:
                        yield TextDelta(text=str(payload))
                    elif kind == "turn":
                        llm_turn = payload
                    elif kind == "error":
                        raise payload
            if llm_turn is None:
                llm_turn = await engine.llm.generate(
                    system=system,
                    messages=app.messages,
                    tool_defs=tool_defs,
                    max_output_tokens=output_tokens_cap,
                )
        except Exception as exc:
            yield Terminal(reason="error", text="", error=str(exc))
            return

        if session.model is None:
            session.model = llm_turn.model  # sticky latch
        if llm_turn.truncated and output_tokens_cap < settings.escalated_output_tokens:
            output_tokens_cap = settings.escalated_output_tokens
            session.max_output_tokens = output_tokens_cap
            yield Status(text="Output slot escalated after truncation; retrying turn.")
            continue

        if llm_turn.text and not llm_turn.tool_calls:
            app.messages.append(ChatMessage(role="assistant", content=llm_turn.text))
            stop = engine.hooks.stop(llm_turn.text)
            if stop.block and turns < session.max_turns - 1:
                app.messages.append(ChatMessage(role="user", content=stop_continuation_prompt(stop.message)))
                yield Status(text="Stop hook requested continuation.")
                continue
            yield AssistantMessage(text=llm_turn.text)
            yield Terminal(reason="completed", text=llm_turn.text)
            return

        if not llm_turn.tool_calls:
            yield Terminal(reason="completed", text=llm_turn.text or "(no response)")
            return

        app.messages.append(
            ChatMessage(role="assistant", content=llm_turn.text, tool_calls=llm_turn.tool_calls)
        )
        if llm_turn.text:
            yield AssistantMessage(text=llm_turn.text)

        prepared = [
            prepare_call(engine.registry, c["id"], c["name"], c.get("arguments") or {})
            for c in llm_turn.tool_calls
        ]
        needing_ask: list[PreparedCall] = []
        auto: list[PreparedCall] = []
        denied: list[tuple[PreparedCall, str]] = []
        for call in prepared:
            hook = engine.hooks.pre_tool_use(call.name, call.arguments)
            if hook.block:
                denied.append((call, hook.message or "blocked by hook"))
                continue
            perm = permission_for(call, session.permission_mode, hook.decision)
            if perm.decision == "deny":
                denied.append((call, perm.reason))
            elif perm.decision == "ask":
                needing_ask.append(call)
            else:
                auto.append(call)

        for call, reason in denied:
            app.messages.append(
                ChatMessage(
                    role="tool",
                    content=f"Denied: {reason}",
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                )
            )
            yield ToolFinished(call_id=call.call_id, name=call.name, ok=False, output=reason)

        if needing_ask:
            app.pending_calls = [
                {"id": c.call_id, "name": c.name, "arguments": c.arguments} for c in needing_ask
            ]
            # If some calls were auto, run those first so work isn't lost.
            if auto:
                async for event in _run_tools(engine, auto):
                    yield event
            first = needing_ask[0]
            reason = (
                "These tool calls need confirmation:\n"
                + "\n".join(f"- {c.name} {c.arguments}" for c in needing_ask)
            )
            app.pending_reason = reason
            yield PermissionRequest(
                call_id=first.call_id,
                name=first.name,
                input=first.arguments,
                reason=reason,
            )
            return

        if auto:
            async for event in _run_tools(engine, auto):
                yield event
        elif not denied:
            yield Terminal(reason="error", text="", error="tool calls produced no work")
            return

    yield Terminal(reason="max_turns", text="Reached max turns without finishing.")


async def _run_tools(engine: Any, prepared: list[PreparedCall]) -> AsyncIterator[Any]:
    async def on_start(call: PreparedCall, speculative: bool):
        engine.session.tool_calls += 1
        await engine._sink(
            ToolStarted(
                call_id=call.call_id,
                name=call.name,
                input=call.arguments,
                speculative=speculative,
            )
        )

    async def on_done(call: PreparedCall, result, ms: int):
        note = engine.hooks.post_tool_use(call.name, result.output)
        output = result.output
        if note:
            output = output + "\n" + note
        engine.app.messages.append(
            ChatMessage(
                role="tool",
                content=output if result.ok else (result.error or output),
                tool_call_id=call.call_id,
                tool_name=call.name,
            )
        )
        await engine._sink(
            ToolFinished(
                call_id=call.call_id,
                name=call.name,
                ok=result.ok,
                output=(output if result.ok else (result.error or output))[:2000],
                duration_ms=ms,
            )
        )

    engine._batch_events = []

    async def sink(event):
        engine._batch_events.append(event)

    previous = engine._sink
    engine._sink = sink
    try:
        await execute_batches(
            prepared,
            engine,
            engine.settings.tool_result_chars,
            on_start=on_start,
            on_done=on_done,
        )
    finally:
        engine._sink = previous
    for event in engine._batch_events:
        yield event
