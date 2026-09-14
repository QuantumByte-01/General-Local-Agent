from __future__ import annotations

import json
from typing import Any

from agent.state import ChatMessage

SNIP_KEEP = 800
COLLAPSE_AFTER = 12


def estimate_chars(messages: list[ChatMessage]) -> int:
    total = 0
    for m in messages:
        total += len(m.content or "")
        if m.tool_calls:
            total += len(json.dumps(m.tool_calls, default=str))
    return total


def snip_old_tool_results(messages: list[ChatMessage], keep_last: int = 4) -> list[ChatMessage]:
    """Layer 1: truncate older tool outputs, keep recent ones intact."""
    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    stale = set(tool_indices[:-keep_last]) if len(tool_indices) > keep_last else set()
    out: list[ChatMessage] = []
    for i, m in enumerate(messages):
        if i in stale and m.content and len(m.content) > SNIP_KEEP:
            out.append(
                ChatMessage(
                    role=m.role,
                    content=m.content[:SNIP_KEEP] + "\n…[snipped old tool result]",
                    tool_call_id=m.tool_call_id,
                    tool_name=m.tool_name,
                    extra=m.extra,
                )
            )
        else:
            out.append(m)
    return out


def microcompact(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Layer 2: drop bulky extra metadata from older turns."""
    out: list[ChatMessage] = []
    for m in messages[:-6]:
        extra = {k: v for k, v in m.extra.items() if k in {"error"}}
        out.append(
            ChatMessage(
                role=m.role,
                content=m.content,
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
                tool_name=m.tool_name,
                extra=extra,
            )
        )
    out.extend(messages[-6:])
    return out


def collapse_tool_runs(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Layer 3: fold a long prefix of tool chatter into one summary message."""
    if len(messages) < COLLAPSE_AFTER + 4:
        return messages
    keep_tail = messages[-8:]
    head = messages[:-8]
    n_tools = sum(1 for m in head if m.role == "tool")
    n_calls = sum(1 for m in head if m.role == "assistant" and m.tool_calls)
    summary = ChatMessage(
        role="user",
        content=(
            f"[compacted earlier work: {n_calls} tool-call turns, {n_tools} results. "
            "Continue from the recent messages below.]"
        ),
    )
    preserved = [m for m in head if m.role == "user"][:2]
    return preserved + [summary] + keep_tail


def autocompact(messages: list[ChatMessage], budget: int) -> list[ChatMessage]:
    """Layer 4: apply 1–3 until under budget. LLM summary is optional in the loop."""
    current = list(messages)
    current = snip_old_tool_results(current)
    if estimate_chars(current) > budget:
        current = microcompact(current)
    if estimate_chars(current) > budget:
        current = collapse_tool_runs(current)
    return current
