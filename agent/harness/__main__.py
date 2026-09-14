from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from agent.harness.cases import load_case, run_case


async def _run_all(paths: list[Path]) -> int:
    failed = 0
    for path in paths:
        case = load_case(path)
        with tempfile.TemporaryDirectory(prefix="gla-harness-") as tmp:
            workspace = Path(tmp)
            trace, failures = await run_case(case, workspace)
        name = case.get("id") or path.stem
        if failures:
            failed += 1
            print(f"FAIL {name}")
            for item in failures:
                print(f"  - {item}")
            if trace.terminal:
                print(f"  terminal={trace.terminal.reason} text={trace.terminal.text[:200]!r}")
            print(f"  tools={trace.tool_names}")
        else:
            print(f"PASS {name}")
    print(json.dumps({"ran": len(paths), "failed": failed}))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run General Local Agent harness scenarios")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("tests/harness/cases")],
        help="Scenario YAML file or directory",
    )
    args = parser.parse_args(argv)
    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.yaml")))
            files.extend(sorted(path.glob("*.yml")))
        elif path.exists():
            files.append(path)
        else:
            print(f"missing: {path}", file=sys.stderr)
            return 2
    if not files:
        print("no scenario files found", file=sys.stderr)
        return 2
    return asyncio.run(_run_all(files))


if __name__ == "__main__":
    raise SystemExit(main())
