from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


KINDS = ("user", "project", "session", "feedback")


class MemoryStore:
    """File-based memory. Four kinds, no database."""

    def __init__(self, project_root: Path, workspace: Path) -> None:
        self.project_root = project_root
        self.workspace = workspace
        self.dir = project_root / ".gla" / "memory"
        self.dir.mkdir(parents=True, exist_ok=True)
        for kind in ("user", "project", "feedback"):
            path = self._path(kind)
            if not path.exists():
                path.write_text(f"# {kind} memory\n", encoding="utf-8")

    def _path(self, kind: str, session_id: str | None = None) -> Path:
        if kind == "session":
            sid = (session_id or "default").replace("/", "_")[:12]
            return self.dir / f"session-{sid}.md"
        return self.dir / f"{kind}.md"

    def read_all(self, session_id: str) -> dict[str, str]:
        data: dict[str, str] = {}
        for kind in KINDS:
            path = self._path(kind, session_id)
            if path.exists():
                data[kind] = path.read_text(encoding="utf-8", errors="replace")
        agent_md = self.workspace / "AGENT.md"
        if agent_md.exists():
            data["agent_md"] = agent_md.read_text(encoding="utf-8", errors="replace")
        return data

    def append(self, kind: str, title: str, body: str, session_id: str) -> str:
        path = self._path(kind, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        block = f"\n## {title}\n_{stamp}_\n\n{body.strip()}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block)
        return str(path)

    def keyword_recall(self, query: str, session_id: str, budget: int = 4000) -> str:
        blob = self.read_all(session_id)
        terms = [t.lower() for t in query.split() if len(t) > 3]
        chunks: list[str] = []
        for kind, text in blob.items():
            parts = text.split("\n## ")
            for part in parts:
                hay = part.lower()
                if not terms or any(t in hay for t in terms):
                    chunks.append(f"[{kind}] {part.strip()[:1200]}")
        if not chunks:
            # fall back to project + agent.md heads
            for kind in ("agent_md", "project", "user"):
                if kind in blob:
                    chunks.append(f"[{kind}] {blob[kind][:800]}")
        joined = "\n\n".join(chunks)
        return joined[:budget]
