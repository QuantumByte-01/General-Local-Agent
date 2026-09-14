from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.pipeline import PreparedCall, budget_output


def partition_calls(prepared: list[PreparedCall]) -> list[list[PreparedCall]]:
    """Group consecutive concurrency-safe calls; isolate the rest. Fail closed."""
    batches: list[list[PreparedCall]] = []
    parallel: list[PreparedCall] = []

    def flush_parallel() -> None:
        nonlocal parallel
        if parallel:
            batches.append(parallel)
            parallel = []

    for call in prepared:
        safe = False
        if call.tool is not None and call.error is None:
            try:
                safe = bool(call.tool.is_concurrency_safe(call.arguments))
            except Exception:
                safe = False
        if safe:
            parallel.append(call)
        else:
            flush_parallel()
            batches.append([call])
    flush_parallel()
    return batches


async def run_prepared(
    call: PreparedCall,
    ctx: ToolContext,
    limit: int,
) -> ToolResult:
    if call.tool is None or call.error:
        return ToolResult(ok=False, output="", error=call.error or "invalid tool call")
    try:
        result = await call.tool.run(call.arguments, ctx)
    except Exception as exc:
        result = ToolResult(ok=False, output="", error=f"{type(exc).__name__}: {exc}")
    return budget_output(result, limit)


async def execute_batches(
    prepared: list[PreparedCall],
    ctx: ToolContext,
    limit: int,
    on_start: Callable[[PreparedCall, bool], Awaitable[None] | None] | None = None,
    on_done: Callable[[PreparedCall, ToolResult, int], Awaitable[None] | None] | None = None,
) -> list[ToolResult]:
    """Run batches; buffer results and return them in submission order."""
    results: list[ToolResult | None] = [None] * len(prepared)
    index = {id(call): i for i, call in enumerate(prepared)}

    async def one(call: PreparedCall, speculative: bool) -> None:
        if on_start:
            maybe = on_start(call, speculative)
            if asyncio.iscoroutine(maybe):
                await maybe
        t0 = time.perf_counter()
        result = await run_prepared(call, ctx, limit)
        ms = int((time.perf_counter() - t0) * 1000)
        results[index[id(call)]] = result
        if on_done:
            maybe = on_done(call, result, ms)
            if asyncio.iscoroutine(maybe):
                await maybe

    for batch in partition_calls(prepared):
        if len(batch) == 1:
            await one(batch[0], speculative=False)
        else:
            await asyncio.gather(*(one(call, speculative=True) for call in batch))

    return [r or ToolResult(ok=False, output="", error="missing result") for r in results]
