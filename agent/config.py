from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]


@dataclass
class Settings:
    workspace: Path
    project_root: Path
    gemini_keys: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    tavily_key: str = ""
    permission_mode: str = "default"
    max_turns: int = 24
    max_output_tokens: int = 8192
    escalated_output_tokens: int = 65536
    compact_chars: int = 80_000
    tool_result_chars: int = 12_000
    mcp_config: Path | None = None
    hooks_config: Path | None = None
    low_latency: bool = True
    thinking_budget: int | None = 0
    stream: bool = True
    llm_timeout_ms: int = 30_000


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_csv(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def load_settings(project_root: Path | None = None) -> Settings:
    project_root = project_root or Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    workspace = Path(os.getenv("AGENT_BASE_DIR") or os.getcwd()).expanduser().resolve()
    keys = _split_csv(os.getenv("GEMINI_API_KEY", ""))
    models = _split_csv(os.getenv("GEMINI_MODEL_PREFERENCE", "") or os.getenv("GEMINI_MODELS", ""))
    mcp_path = project_root / "mcp.json"
    hooks_path = project_root / "hooks.json"
    low_latency = os.getenv("AGENT_LOW_LATENCY", "1").strip().lower() not in {"0", "false", "no"}
    thinking_raw = os.getenv("AGENT_THINKING_BUDGET")
    if thinking_raw is None or not thinking_raw.strip():
        thinking_budget: int | None = 0 if low_latency else None
    else:
        try:
            thinking_budget = int(thinking_raw.strip())
        except ValueError:
            thinking_budget = 0 if low_latency else None
    timeout_raw = os.getenv("AGENT_LLM_TIMEOUT_MS", "").strip()
    if timeout_raw:
        try:
            llm_timeout_ms = int(timeout_raw)
        except ValueError:
            llm_timeout_ms = 30_000 if low_latency else 60_000
    else:
        llm_timeout_ms = 30_000 if low_latency else 60_000
    return Settings(
        workspace=workspace,
        project_root=project_root,
        gemini_keys=keys,
        models=models or list(DEFAULT_MODELS),
        tavily_key=os.getenv("TAVILY_API_KEY", "").strip(),
        permission_mode=os.getenv("AGENT_PERMISSION_MODE", "default").strip() or "default",
        max_turns=_int_env("AGENT_MAX_TURNS", 24),
        compact_chars=_int_env("AGENT_COMPACT_CHARS", 80_000),
        tool_result_chars=_int_env("AGENT_TOOL_RESULT_CHARS", 12_000),
        mcp_config=mcp_path if mcp_path.exists() else None,
        hooks_config=hooks_path if hooks_path.exists() else None,
        low_latency=low_latency,
        thinking_budget=thinking_budget,
        stream=low_latency,
        llm_timeout_ms=llm_timeout_ms,
    )
