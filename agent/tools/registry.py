from __future__ import annotations

from agent.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_cache: list[dict] | None = None

    def register(self, tool: Tool, *, replace: bool = False) -> str:
        if not tool.name:
            raise ValueError("tool is missing a name")
        name = tool.name
        if name in self._tools and not replace and self._tools[name] is not tool:
            suffix = 2
            while f"{name}_{suffix}" in self._tools:
                suffix += 1
            tool.name = f"{name}_{suffix}"
        self._tools[tool.name] = tool
        self._schema_cache = None
        return tool.name

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas_for_llm(self) -> list[dict]:
        if self._schema_cache is None:
            self._schema_cache = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                }
                for tool in self._tools.values()
            ]
        return self._schema_cache
