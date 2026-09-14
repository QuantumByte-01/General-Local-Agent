from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult
from agent.tools.paths import resolve_in_workspace


class CsvTool(Tool):
    name = "analyze_csv"
    description = "Profile a CSV: shape, columns, dtypes, and a 5-row preview."
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    risk = "read"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_in_workspace(Path(ctx.workspace), str(arguments["path"]), must_exist=True)
        try:
            import pandas as pd
        except ImportError:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()[:8]
            return ToolResult(ok=True, output="pandas missing; head:\n" + "\n".join(lines))
        df = pd.read_csv(path)
        info = {
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "preview": df.head(5).to_dict(orient="records"),
        }
        return ToolResult(ok=True, output=json.dumps(info, indent=2, default=str))
