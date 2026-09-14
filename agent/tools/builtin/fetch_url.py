from __future__ import annotations

from typing import Any

import httpx

from agent.tools.base import Tool, ToolContext, ToolResult


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = "HTTP GET a URL and return text content (truncated). Use after web_search."
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer"},
        },
        "required": ["url"],
    }
    risk = "network"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = str(arguments["url"])
        if not url.startswith(("http://", "https://")):
            return ToolResult(ok=False, output="", error="url must be http(s)")
        max_chars = int(arguments.get("max_chars") or 8000)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
                resp = await client.get(url, headers={"User-Agent": "GeneralLocalAgent/2"})
                resp.raise_for_status()
                text = resp.text
        except Exception as exc:
            return ToolResult(ok=False, output="", error=str(exc))
        return ToolResult(ok=True, output=text[:max_chars])
