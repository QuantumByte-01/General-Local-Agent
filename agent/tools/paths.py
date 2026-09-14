from __future__ import annotations

from pathlib import Path


def resolve_in_workspace(workspace: Path, raw: str, must_exist: bool = False) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise PermissionError(f"path escapes workspace: {raw}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved
