from agent.tools.builtin.csv import CsvTool
from agent.tools.builtin.delegate import DelegateTool
from agent.tools.builtin.edit import EditTool
from agent.tools.builtin.fetch_url import FetchUrlTool
from agent.tools.builtin.glob_tool import GlobTool
from agent.tools.builtin.grep import GrepTool
from agent.tools.builtin.memory import MemoryTool
from agent.tools.builtin.read import ReadTool
from agent.tools.builtin.shell import ShellTool
from agent.tools.builtin.skill import SkillTool
from agent.tools.builtin.system_info import SystemInfoTool
from agent.tools.builtin.web_search import WebSearchTool
from agent.tools.builtin.write import WriteTool
from agent.tools.registry import ToolRegistry


def register_builtin(registry: ToolRegistry) -> None:
    for tool in (
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
        ShellTool(),
        WebSearchTool(),
        FetchUrlTool(),
        SystemInfoTool(),
        CsvTool(),
        MemoryTool(),
        SkillTool(),
        DelegateTool(),
    ):
        registry.register(tool)
