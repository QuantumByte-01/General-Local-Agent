---
name: pdf-summarize
description: Summarize local PDFs with page-aware extraction
when_to_use: User asks to summarize, extract, or Q&A a PDF in the workspace
---

1. `glob_files` for `**/*.pdf` if the path is unclear.
2. `read_file` on the PDF (text extraction is built in).
3. Write a short summary: purpose, key claims, numbers, open questions.
4. Cite the file path.
