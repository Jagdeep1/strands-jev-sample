import asyncio
from types import SimpleNamespace

from strands.interventions import Confirm, Deny, Proceed

from strands_jev import JevToolGate, Noul

POLICY = "Refunds only for late, damaged, or wrong items. Never refund digital goods."


def make_event(tool_name="issue_refund", tool_input=None, messages=None):
    """Minimal stand-in for BeforeToolCallEvent: the gate only reads these fields."""
    agent = SimpleNamespace(
        messages=messages
        or [
            {"role": "user", "content": [{"text": "My kettle ORD-1 arrived 9 days late, refund shipping."}]},
            {"role": "assistant", "content": [{"text": "Checking."}, {"toolUse": {"name": "issue_refund"}}]},
        ],
        system_prompt="You are a support agent.",
    )
    tool_use = {"toolUseId": "t1", "name": tool_name, "input": tool_input or {"order_id": "ORD-1", "amount_cents": 1200}}
    return SimpleNamespace(agent=agent, tool_use=tool_use, selected_tool=None, invocation_state={})


def answers(**probs):
    return {name: {"type": "noul", "noul": p} for name, p in probs.items()}


def run(gate, event):
    return asyncio.run(gate.before_tool_call(event))


def test_ungated_tool_proceeds_without_calling_jev(jev, fake):
    gate = JevToolGate(jev, policy=POLICY, tools=["issue_refund"])
    action = run(gate, make_event(tool_name="search_orders"))
    assert isinstance(action, Proceed)
    assert fake.requests == []


def test_all_checks_clear_approves(jev, fake):
    fake.answers = answers(user_requested=0.98, policy_allows=0.95, within_scope=0.97)
    gate = JevToolGate(jev, policy=POLICY, tools=["issue_refund"])
    action = run(gate, make_event())
    assert isinstance(action, Proceed)

    sent = fake.requests[0]
    assert sent["state"]["policy"] == POLICY
    assert sent["state"]["tool_call"] == {"name": "issue_refund", "input": {"order_id": "ORD-1", "amount_cents": 1200}}
    assert "refund shipping" in sent["state"]["conversation"][0]["text"]
    assert set(sent["questions"]) == {"user_requested", "policy_allows", "within_scope"}

    assert gate.decisions[-1].outcome == "approve"
    assert gate.decisions[-1].checks["policy_allows"] == 0.95


def test_any_check_clearly_false_denies_with_reason(jev, fake):
    fake.answers = answers(user_requested=0.97, policy_allows=0.03, within_scope=0.9)
    gate = JevToolGate(jev, policy=POLICY, tools=["issue_refund"])
    action = run(gate, make_event())
    assert isinstance(action, Deny)
    assert "policy_allows=0.03" in action.reason
    assert gate.decisions[-1].outcome == "block"


def test_ambiguous_asks_for_confirmation_by_default(jev, fake):
    fake.answers = answers(user_requested=0.98, policy_allows=0.45, within_scope=0.95)
    gate = JevToolGate(jev, policy=POLICY, tools=["issue_refund"])
    action = run(gate, make_event())
    assert isinstance(action, Confirm)
    assert action.response is None  # pauses the agent for external human review
    assert "issue_refund" in action.prompt
    assert gate.decisions[-1].outcome == "review"


def test_ambiguous_uses_reviewer_callback_when_given(jev, fake):
    fake.answers = answers(user_requested=0.98, policy_allows=0.45, within_scope=0.95)
    seen = []

    def reviewer(decision):
        seen.append(decision)
        return True

    gate = JevToolGate(jev, policy=POLICY, tools=["issue_refund"], reviewer=reviewer)
    action = run(gate, make_event())
    assert isinstance(action, Confirm)
    assert action.response is True
    assert seen[0].tool_name == "issue_refund"


def test_jev_failure_fails_closed(jev, fake):
    fake.status_code = 503
    gate = JevToolGate(jev, policy=POLICY, tools=["issue_refund"])
    action = run(gate, make_event())
    assert isinstance(action, Deny)
    assert "503" in action.reason
    assert gate.decisions[-1].outcome == "block"
    assert gate.decisions[-1].checks is None


def test_custom_checks_and_thresholds(jev, fake):
    fake.answers = answers(is_read_only=0.8)
    gate = JevToolGate(
        jev,
        policy=POLICY,
        checks={"is_read_only": Noul(instructions="`tool_call` only reads data.")},
        approve_at=0.75,
        block_at=0.2,
    )
    action = run(gate, make_event(tool_name="anything"))
    assert isinstance(action, Proceed)
    assert set(fake.requests[0]["questions"]) == {"is_read_only"}


def test_conversation_is_trimmed_to_recent_text_only(jev, fake):
    fake.answers = answers(user_requested=0.99, policy_allows=0.99, within_scope=0.99)
    messages = [{"role": "user", "content": [{"text": f"msg {i}"}]} for i in range(20)]
    gate = JevToolGate(jev, policy=POLICY, max_messages=4)
    run(gate, make_event(messages=messages))
    convo = fake.requests[0]["state"]["conversation"]
    assert [m["text"] for m in convo] == ["msg 16", "msg 17", "msg 18", "msg 19"]
    assert all(set(m) == {"role", "text"} for m in convo)
