"""Minimal stdio MCP server: echo + add. Used by mcp.example.json and tests."""

from __future__ import annotations


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("mcp package is required to run the echo server") from exc

    server = FastMCP("gla-echo")

    @server.tool()
    def echo(text: str) -> str:
        """Return the same text."""
        return text

    @server.tool()
    def add(a: int, b: int) -> int:
        """Return a + b."""
        return a + b

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
