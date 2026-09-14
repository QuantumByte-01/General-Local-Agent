# Testing and harness

Unit tests cover permissions, compact, skills, hooks, and the tool executor. The **harness** runs the real query loop against a **scripted LLM** and real tools in a temp workspace. No Gemini key is required.

## Run

```powershell
uv sync --extra dev
uv run pytest
uv run python -m agent.harness tests/harness/cases
```

`pytest` picks up every YAML file in `tests/harness/cases/`. Set `mcp: true` on a case to attach the in-process echo/add MCP server (`mcp_harness_echo`, `mcp_harness_add`). Copy `mcp.example.json` to `mcp.json` to spawn the real stdio echo server in the live app.

A new user message while tools are waiting **cancels** those calls (see `pending_replaced_by_new_request`). `delegate_explore` covers sub-agents (child turns consume the next `script` items because they share the scripted LLM).

## Adding a case

```yaml
id: my_case
user: Do the thing
mode: dont_ask
mcp: false
workspace_files:
  hello.txt: hi
script:
  - tool_calls:
      - name: read_file
        arguments: {path: hello.txt}
  - text: Final answer
resumes: ["yes"]
expect:
  terminal: completed   # completed | max_turns | error | ...
  tools: [read_file]
  answer_contains: hi
  files:
    out.txt: body
  files_absent: [nope.txt]
  pending: false
```

The scripted model pops one `script` item per `generate()` call. `truncated: true` exercises output-slot escalation.

## What it does not do

Live-model evals. Point `ScriptedLLM` at recorded traces if you want that later; keep secrets out of git.
