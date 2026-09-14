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
| `TAVILY_API_KEY` | Optional; otherwise DuckDuckGo |
| `AGENT_HOST` / `AGENT_PORT` | UI bind (default `127.0.0.1:7869`) |

```powershell
uv run app.py
```

Open `http://127.0.0.1:7869`. The banner lists tool count, skills, and MCP status.

## Permission modes

Use **plan** to inspect. **default** confirms writes and shell. **accept_edits** auto-applies file edits. **dont_ask** runs everything (still traced).

When the agent pauses, reply `yes` or `no`.

Paths cannot escape `AGENT_BASE_DIR`.

## Skills

Skills are `SKILL.md` files with YAML frontmatter. Only `name` / `description` / `when_to_use` go into the system prompt. The model calls `load_skill` for the body.

Search order: `skills/` in this repo, `skills/` in the workspace, `.gla/skills/`, `~/.gla/skills/`. First name wins.

Bundled: `pdf-summarize`, `web-research`, `system-audit`, `code-change`.

## Memory

Written under `.gla/memory/`:

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

SSE: `{ "url": "http://127.0.0.1:3000/sse", "transport": "sse" }`.

Tools appear as `mcp_<server>_<tool>`. Failed servers show in the UI banner; the rest of the agent still starts.

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
```
