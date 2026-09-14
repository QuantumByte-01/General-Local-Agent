# Usage

## Install

```powershell
cd path\to\General-Local-Agent
copy .env.example .env
uv sync
```

Edit `.env`:

| Variable | Meaning |
| --- | --- |
| `GEMINI_API_KEY` | One key, or comma-separated keys |
| `GEMINI_MODEL_PREFERENCE` | Preferred models, first successful latched |
| `AGENT_BASE_DIR` | Workspace the tools may read/write |
| `AGENT_PERMISSION_MODE` | `plan` / `default` / `accept_edits` / `dont_ask` |
| `AGENT_LOW_LATENCY` | `1` (default): stream tokens, skip thinking, reuse HTTP client. LLM still runs. |
| `AGENT_THINKING_BUDGET` | Gemini thinking tokens. Default `0` when low-latency is on. |
| `AGENT_LLM_TIMEOUT_MS` | Per-attempt LLM timeout (default `30000` in low-latency). |
| `AGENT_MAX_TURNS` | Max model turns per query (default `24`). |
| `AGENT_COMPACT_CHARS` | Transcript size before autocompact (default `80000`). |
| `TAVILY_API_KEY` | Optional; otherwise DuckDuckGo |
| `AGENT_HOST` / `AGENT_PORT` | UI bind (default `127.0.0.1:7869`) |

```powershell
uv run app.py
uv run gla
uv run gla -p "Read AGENT.md and summarize it" --mode plan
uv run gla -p "Create notes.md" --mode default -y
```

Open `http://127.0.0.1:7869`. The banner lists tool count, skills, and MCP status.

Print mode (`gla -p`) uses the same query loop and the same LLM. `-y` auto-approves permission prompts.

## Permission modes

Use **plan** to inspect. **default** confirms writes and shell. **accept_edits** auto-applies file edits. **dont_ask** runs everything (still traced).

When the agent pauses, click **Allow** or **Deny**, reply `yes` / `no`, or send a **new request** (that cancels the pending calls). **Abort** stops the current turn.

Slash commands in the chat box: `/help`, `/status`, `/mode <name>`, `/compact`, `/clear`, `/abort`, `/allow`, `/deny`.

Paths cannot escape `AGENT_BASE_DIR`.

## Skills

Skills are `SKILL.md` files with YAML frontmatter. Only `name` / `description` / `when_to_use` go into the system prompt. The model calls `load_skill` for the body.

Search order: `skills/` in this repo, `skills/` in the workspace, `.gla/skills/`, `~/.gla/skills/`. First name wins.

Bundled: `pdf-summarize`, `web-research`, `system-audit`, `code-change`.

## Memory

Written under **the project root** `.gla/memory/` (the repo that contains `mcp.json` / this package), not necessarily `AGENT_BASE_DIR`:

- `user.md` — preferences
- `project.md` — repo conventions
- `session-*.md` — this session
- `feedback.md` — corrections

`AGENT.md` in the workspace is always injected (trimmed).

## MCP

`mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\path\\to\\workspace"]
    }
  }
}
```

A local echo server is in `mcp.example.json` (`python -m agent.mcp.echo_server`). Copy it to `mcp.json` to enable. Servers connect in parallel (20s timeout on Windows, 8s elsewhere). Duplicate tool names are renamed instead of overwriting builtins.

Copy `hooks.example.json` to `hooks.json` to start from a template. `hooks.json` is snapshotted once at process start.

## Latency

The LLM stays in the loop. Defaults favor flash models, reuse the Gemini HTTP client, stream tokens to the UI, skip thinking tokens (`AGENT_THINKING_BUDGET=0`), cache tool schemas, and skip prompt rebuild on permission resume. Set `AGENT_LOW_LATENCY=0` to turn streaming/thinking overrides off.

## Hooks

`hooks.json` is read **once** at startup.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "shell",
        "permissionBehavior": "ask",
        "message": "Confirm shell"
      }
    ],
    "Stop": [
      { "block": true, "message": "Verify the user’s goal is actually done." }
    ]
  }
}
```

`matcher` is `*` or `tool|tool`. `"block": true` on PreToolUse denies the call. On Stop it forces another model turn. `permissionBehavior`: `deny` > `ask` > `allow`.

## Tests

```powershell
uv sync --extra dev
uv run pytest
uv run python -m agent.harness
```

See [Testing](TESTING.md). The harness drives the real loop with a scripted model (no API key).
