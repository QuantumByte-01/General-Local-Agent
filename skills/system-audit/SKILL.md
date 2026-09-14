---
name: system-audit
description: Report CPU, RAM, disk, and heavy processes
when_to_use: User asks how the machine is doing, storage, or what is hogging resources
---

1. Call `system_info`.
2. Optionally `shell` for `Get-PSDrive` if disk detail is needed (ask first unless dont_ask).
3. Give a short report: pressure, top processes, what is safe to close. No destructive commands.
