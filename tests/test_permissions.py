from agent.permissions import is_affirmative, resolve_permission


def test_plan_blocks_writes():
    assert resolve_permission(mode="plan", risk="read").decision == "allow"
    assert resolve_permission(mode="plan", risk="write").decision == "deny"
    assert resolve_permission(mode="plan", risk="exec").decision == "deny"


def test_hook_deny_wins():
    result = resolve_permission(mode="dont_ask", risk="read", hook_decision="deny")
    assert result.decision == "deny"


def test_accept_edits_asks_for_shell():
    assert resolve_permission(mode="accept_edits", risk="write").decision == "allow"
    assert resolve_permission(mode="accept_edits", risk="exec").decision == "ask"


def test_default_reads_ok():
    assert resolve_permission(mode="default", risk="read").decision == "allow"
    assert resolve_permission(mode="default", risk="write").decision == "ask"


def test_yes_no():
    assert is_affirmative("yes") is True
    assert is_affirmative("no") is False
    assert is_affirmative("maybe") is None
    assert is_affirmative("yes please") is True
    assert is_affirmative("no thanks") is False
    assert is_affirmative("notes") is None
