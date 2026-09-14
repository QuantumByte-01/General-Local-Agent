---
name: code-change
description: Read, edit, and verify files in the workspace
when_to_use: Code edits, refactors, or adding files
---

1. `glob_files` / `grep` to find the right files.
2. `read_file` before every `edit_file`.
3. Small diffs. If uniqueness fails, re-read.
4. Summarize what changed and how to run it. Do not git commit unless asked.
