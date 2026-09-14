from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SkillMeta:
    name: str
    description: str
    when_to_use: str
    source: str
    path: Path
    body: str
    trusted: bool = True


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2].lstrip("\n")


class SkillLoader:
    """Two-phase skills: metadata at boot, body on load_skill."""

    def __init__(self, project_root: Path, workspace: Path) -> None:
        self.skills: dict[str, SkillMeta] = {}
        search_roots = [
            project_root / "skills",
            workspace / "skills",
            workspace / ".gla" / "skills",
            Path.home() / ".gla" / "skills",
        ]
        seen: set[str] = set()
        for root in search_roots:
            if not root.is_dir():
                continue
            for skill_md in root.glob("*/SKILL.md"):
                text = skill_md.read_text(encoding="utf-8", errors="replace")
                meta, body = _split_frontmatter(text)
                name = str(meta.get("name") or skill_md.parent.name)
                if name in seen:
                    continue
                seen.add(name)
                self.skills[name] = SkillMeta(
                    name=name,
                    description=str(meta.get("description") or ""),
                    when_to_use=str(meta.get("when_to_use") or meta.get("whenToUse") or ""),
                    source=str(root),
                    path=skill_md,
                    body=body,
                    trusted=True,
                )

    def menu(self) -> dict[str, str]:
        return {
            name: (s.description or s.when_to_use or "skill")
            for name, s in self.skills.items()
        }

    def render(self, name: str, extra: str = "", session_id: str = "") -> str | None:
        skill = self.skills.get(name)
        if skill is None:
            return None
        body = skill.body
        body = body.replace("$ARGUMENTS", extra)
        body = body.replace("${SESSION_ID}", session_id)
        body = body.replace("${SKILL_DIR}", str(skill.path.parent))
        header = (
            f"# Skill: {skill.name}\n{skill.description}\n\n"
            f"SKILL_DIR={skill.path.parent}\n"
        )
        if extra:
            header += f"ARGUMENTS={extra}\n"
        return header + "\n" + body
