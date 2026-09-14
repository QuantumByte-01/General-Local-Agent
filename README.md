# General Local Agent

Local Windows agent that **uses tools in a loop**. Architecture follows transferable patterns from [Claude Code from Source](https://claude-code-from-source.com/) — original Python, not a port of Anthropic source.

**Docs:** [Architecture](docs/ARCHITECTURE.md) · [Usage](docs/USAGE.md) · [Testing](docs/TESTING.md) · [Diagrams](docs/diagrams/README.md)

## Architecture

![System architecture](docs/diagrams/architecture.svg)

The query loop, tool pipeline, and bootstrap:

| Workflow | Rendered | draw.io |
| --- | --- | --- |
| Query loop | [svg](docs/diagrams/query_loop.svg) | [drawio](docs/diagrams/query_loop.drawio) |
| Tool pipeline | [svg](docs/diagrams/tool_pipeline.svg) | [drawio](docs/diagrams/tool_pipeline.drawio) |
| Bootstrap | [svg](docs/diagrams/bootstrap.svg) | [drawio](docs/diagrams/bootstrap.drawio) |

## Setup

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY` (comma-separated keys allowed) and `AGENT_BASE_DIR`.
3. Optional: `TAVILY_API_KEY`, copy `mcp.example.json` to `mcp.json`. Flash is the default model for lower latency; the LLM is still used.

```powershell
uv sync
uv run app.py
```

UI: `http://127.0.0.1:7869`

Permission modes: `plan` (read-only) · `default` (confirm writes/shell) · `accept_edits` · `dont_ask`.

## Tests

```powershell
uv sync --extra dev
uv run pytest
uv run python -m agent.harness
```
