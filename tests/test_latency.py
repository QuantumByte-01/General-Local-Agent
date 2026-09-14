from agent.config import DEFAULT_MODELS, load_settings
from agent.llm.client import GeminiClient


def test_client_is_reused():
    client = GeminiClient(["k1"], ["gemini-2.5-flash"], thinking_budget=0, stream=False)
    first = client._client(0)
    second = client._client(0)
    assert first is second


def test_tool_schema_cache():
    client = GeminiClient(["k1"], ["gemini-2.5-flash"])
    defs = [{"name": "read_file", "description": "r", "parameters": {"type": "object", "properties": {}}}]
    a = client._tool_payload(defs)
    b = client._tool_payload(defs)
    assert a is b
    assert hasattr(client, "stream_generate")
    assert hasattr(client, "warmup")


def test_low_latency_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LOW_LATENCY", "1")
    monkeypatch.delenv("AGENT_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("AGENT_LLM_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_PREFERENCE", raising=False)
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    monkeypatch.setenv("AGENT_MAX_TURNS", "12")
    settings = load_settings(tmp_path)
    assert settings.low_latency is True
    assert settings.stream is True
    assert settings.thinking_budget == 0
    assert settings.llm_timeout_ms == 30_000
    assert settings.max_turns == 12
    assert settings.models[0] == DEFAULT_MODELS[0]
