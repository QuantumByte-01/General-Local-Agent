from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult
from agent.tools.files import record_file
from agent.tools.paths import resolve_in_workspace

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_PDF_EXT = {".pdf"}


def _with_line_numbers(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    width = len(str(start + len(lines)))
    return "\n".join(f"{i:>{width}}|{line}" for i, line in enumerate(lines, start=start))


class ReadTool(Tool):
    name = "read_file"
    description = (
        "Read a file in the workspace. Text files are returned with line numbers. "
        "PDFs extract text. Images return basic metadata. Use offset/limit for large files."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based start line"},
            "limit": {"type": "integer", "description": "max lines (default 400)"},
        },
        "required": ["path"],
    }
    risk = "read"
    max_result_chars = 80_000

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_in_workspace(Path(ctx.workspace), str(arguments["path"]), must_exist=True)
        if path.is_dir():
            names = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return ToolResult(ok=True, output="\n".join(names) or "(empty directory)")

        ext = path.suffix.lower()
        stat = path.stat()
        if ext in _IMAGE_EXT:
            info = f"image {path} size={stat.st_size} bytes"
            try:
                from PIL import Image

                with Image.open(path) as img:
                    info = f"image {path.name}: {img.format} {img.size[0]}x{img.size[1]} mode={img.mode}"
            except Exception:
                pass
            return ToolResult(ok=True, output=info)

        if ext in _PDF_EXT:
            try:
                import PyPDF2

                reader = PyPDF2.PdfReader(str(path))
                chunks = []
                for page in reader.pages[:8]:
                    chunks.append(page.extract_text() or "")
                text = "\n".join(chunks)[:20000]
            except Exception as exc:
                return ToolResult(ok=False, output="", error=str(exc))
            return ToolResult(ok=True, output=text or "(no extractable PDF text)")

        raw = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        offset = max(1, int(arguments.get("offset") or 1))
        limit = int(arguments.get("limit") or 400)
        slice_ = lines[offset - 1 : offset - 1 + limit]
        numbered = _with_line_numbers("\n".join(slice_), start=offset)
        suffix = ""
        if offset - 1 + limit < len(lines):
            suffix = f"\n… {len(lines) - (offset - 1 + limit)} more lines"
        record_file(path, raw, ctx.session)
        header = f"{path} ({len(lines)} lines)\n"
        return ToolResult(ok=True, output=header + numbered + suffix)
