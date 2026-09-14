from __future__ import annotations

from agent.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("tool is missing a name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas_for_llm(self) -> list[dict]:
        out = []
        for tool in self._tools.values():
            out.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                }
            )
        return out
