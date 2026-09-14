# General Local Agent

Local computer agent for this workspace. Prefer tools over guesses. Keep secrets out of memory files.

## Conventions

- Python 3.12, `uv` for deps
- Stay inside `AGENT_BASE_DIR`
- Do not add AI tools as git co-authors
- Architecture: query loop + self-describing tools + named permission modes

## Pointers

- `docs/ARCHITECTURE.md` — loop, tools, MCP, diagrams
- `docs/USAGE.md` — env, skills, hooks
- `skills/` — two-phase skill packs
