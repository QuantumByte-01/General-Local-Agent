# Architecture

General Local Agent is a **tool-using loop**, not a one-shot planner. Patterns follow [Claude Code from Source](https://claude-code-from-source.com/) (educational). This tree is original Python.

Editable draw.io sources live next to the rendered SVGs in [`diagrams/`](diagrams/). Open any `.drawio` file at [diagrams.net](https://app.diagrams.net/).

## System map

![Architecture](diagrams/architecture.svg)

Six pieces:

| Piece | Role |
| --- | --- |
| Query loop | Async generator: model → tools → results until a typed `Terminal` |
| Tool system | Self-describing tools; reads may run in parallel; writes/shell stay serial |
| Permissions | Named modes, not scattered `if allowed` checks |
| Memory | Markdown files (user / project / session / feedback) plus `AGENT.md` |
| Skills | Frontmatter at boot; full body only on `load_skill` |
| Hooks | Config snapshotted at startup (no silent reread) |

MCP servers from `mcp.json` connect in parallel (8s timeout each, per-server session) and are wrapped as the same `Tool` interface. The harness can attach an in-process echo/add server without spawning a subprocess. Sub-agents keep read-only MCP tools (explore) or all MCP tools (worker).

## Bootstrap

![Bootstrap](diagrams/bootstrap.svg)

`agent/bootstrap.py` loads settings, then in parallel: local files (memory, skill names, frozen `hooks.json`), MCP connect, and Gemini HTTP-client warmup. MCP tools are then wrapped onto the same registry.

## Query loop

![Query loop](diagrams/query_loop.svg)

Implemented in `agent/loop.py`:

1. Append the user turn (or resume a pending permission).
2. Recall memory and build the system prompt.
3. Compact old tool output if the transcript is large.
4. Call Gemini with function declarations. Tokens stream into the UI (`text_delta`); the model is still in the loop. Tool schemas and the HTTP client are cached.
5. If the output was truncated, raise `max_output_tokens` (8K → 64K) and retry that turn (slot reservation).
6. If there are no tool calls, run Stop hooks; maybe continue, else `Terminal(completed)`.
7. Classify each call: hook deny, mode deny, ask, or allow.
8. Run allowed calls in safety batches; pause the UI on `ask`.

The first successful model id is **latched** for the rest of the session.

## Tool pipeline

![Tool pipeline](diagrams/tool_pipeline.svg)

`isConcurrencySafe(input)` is per-call. Fail closed: unknown tool, bad args, or an exception in the safety check → serial.

Permission modes (`agent/permissions.py`):

| Mode | Reads | File writes | Shell / network |
| --- | --- | --- | --- |
| `plan` | allow | deny | deny |
| `default` | allow | ask | ask |
| `accept_edits` | allow | allow | ask |
| `dont_ask` | allow | allow | allow (still logged) |

Hook `permissionBehavior` uses **deny > ask > allow**.

## Layout

```
app.py / agent/ui.py   Gradio REPL that consumes loop events
agent/cli.py           Headless `-p` print mode (same loop + LLM)
agent/loop.py          Query generator
agent/bootstrap.py     Process init
agent/prompts.py       System prompt
agent/tools/           Registry, pipeline, executor, built-ins
agent/mcp/             MCP client + Tool wrap
agent/memory/          File store
agent/skills/          Two-phase loader
agent/hooks/           Frozen snapshot
agent/tasks/           Sub-agent
skills/*/SKILL.md      Bundled skills
mcp.json / hooks.json  Connectors and lifecycle interceptors
```
