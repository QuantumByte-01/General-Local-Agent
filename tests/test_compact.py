from agent.compact import autocompact, estimate_chars, snip_old_tool_results
from agent.state import ChatMessage


def test_snip_old_results():
    messages = [ChatMessage(role="user", content="go")]
    for i in range(8):
        messages.append(ChatMessage(role="tool", content="x" * 2000, tool_name="read_file"))
    snipped = snip_old_tool_results(messages, keep_last=2)
    old = [m for m in snipped if m.role == "tool" and "snipped" in m.content]
    assert old
    assert estimate_chars(snipped) < estimate_chars(messages)


def test_autocompact_under_budget():
    messages = [ChatMessage(role="user", content="start")]
    for i in range(20):
        messages.append(ChatMessage(role="assistant", content="work", tool_calls=[{"name": "read_file"}]))
        messages.append(ChatMessage(role="tool", content="n" * 5000, tool_name="read_file"))
    compact = autocompact(messages, budget=8_000)
    assert estimate_chars(compact) <= estimate_chars(messages)
    assert compact
