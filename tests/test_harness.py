from pathlib import Path

import pytest

from agent.harness.cases import load_case, run_case

CASES = sorted((Path(__file__).parent / "harness" / "cases").glob("*.yaml"))


@pytest.mark.asyncio
@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
async def test_harness_case(path: Path, tmp_path: Path):
    case = load_case(path)
    trace, failures = await run_case(case, tmp_path)
    assert not failures, failures
    assert trace.events
