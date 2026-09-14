from __future__ import annotations

import os
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult


def _tavily(query: str, n: int) -> tuple[list[dict], str | None]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return [], "TAVILY_API_KEY not set"
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        data = client.search(query=query, max_results=n)
        return data.get("results", []) or [], None
    except Exception as exc:
        return [], str(exc)


def _ddg(query: str, n: int) -> tuple[list[dict], str | None]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [], "duckduckgo-search not installed"
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=n))
    except Exception as exc:
        return [], str(exc)
    out = []
    for row in raw:
        out.append(
            {
                "title": row.get("title", ""),
                "url": row.get("href") or row.get("url", ""),
                "content": row.get("body") or row.get("snippet", ""),
            }
        )
    return out, None


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the public web. Uses Tavily when TAVILY_API_KEY is set, "
        "otherwise DuckDuckGo. Return titles, URLs, and snippets."
    )
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer"},
        },
        "required": ["query"],
    }
    risk = "network"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(arguments["query"])
        n = int(arguments.get("num_results") or 5)
        results, err = _tavily(query, n)
        notes = []
        if err:
            notes.append(err)
        if not results:
            ddg, err2 = _ddg(query, n)
            if err2:
                notes.append(err2)
            results = ddg
        if not results:
            return ToolResult(ok=False, output="", error=" | ".join(notes) or "no results")
        blocks = []
        for row in results[:n]:
            blocks.append(
                f"# {row.get('title','')}\n{row.get('url','')}\n{(row.get('content') or '')[:700]}"
            )
        extra = ("\nnotes: " + " | ".join(notes)) if notes else ""
        return ToolResult(ok=True, output="\n\n".join(blocks) + extra)
