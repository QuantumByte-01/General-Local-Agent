from __future__ import annotations

import hashlib

from agent.state import FileReadRecord


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def record_file(path, text: str, session) -> FileReadRecord:
    stat = path.stat()
    rec = FileReadRecord(mtime=stat.st_mtime, size=stat.st_size, excerpt_hash=content_hash(text))
    session.file_reads[str(path)] = rec
    return rec
