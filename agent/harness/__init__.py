"""Evaluation harness: scripted LLM + real tools, no live API required."""

from agent.harness.cases import load_case, run_case, score
from agent.harness.factory import make_engine
from agent.harness.runner import Trace, run_harness
from agent.harness.scripted import ScriptedLLM

__all__ = [
    "ScriptedLLM",
    "Trace",
    "load_case",
    "make_engine",
    "run_case",
    "run_harness",
    "score",
]
