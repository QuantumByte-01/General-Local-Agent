from __future__ import annotations

import json
from typing import Any

from agent.tools.base import Tool, ToolContext, ToolResult


def _human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit, div in (("KB", 1024), ("MB", 1024**2), ("GB", 1024**3)):
        if n < div * 1024 or unit == "GB":
            return f"{n / div:.1f} {unit}"
    return str(n)


class SystemInfoTool(Tool):
    name = "system_info"
    description = "CPU, RAM, disk, battery, and top processes on this machine."
    schema = {
        "type": "object",
        "properties": {
            "top_n": {"type": "integer", "description": "process rows, default 10"},
        },
        "required": [],
    }
    risk = "read"

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return True

    def is_read_only(self, arguments: dict[str, Any]) -> bool:
        return True

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import psutil
        except ImportError:
            return ToolResult(ok=False, output="", error="psutil not installed")
        mem = psutil.virtual_memory()
        disk_target = str(ctx.workspace)
        if len(disk_target) >= 2 and disk_target[1] == ":":
            disk_target = disk_target[:3]
        try:
            disk = psutil.disk_usage(disk_target)
        except Exception:
            disk = psutil.disk_usage("/")
        info: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=0.2),
            "ram_total": _human(mem.total),
            "ram_used_percent": mem.percent,
            "disk_total": _human(disk.total),
            "disk_free": _human(disk.free),
        }
        batt = psutil.sensors_battery()
        if batt:
            info["battery_percent"] = batt.percent
            info["charging"] = batt.power_plugged
        top_n = int(arguments.get("top_n") or 10)
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        procs.sort(key=lambda p: p.get("cpu_percent") or 0, reverse=True)
        info["top_processes"] = procs[:top_n]
        return ToolResult(ok=True, output=json.dumps(info, indent=2, default=str))
