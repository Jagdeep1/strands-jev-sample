"""End-to-end through the real Strands Agent, with a scripted model and the fake Decisions endpoint."""

import json

from helpers import ScriptedModel
from strands import Agent, tool
from strands.models.routing import ModelRouter, RoutingCandidate

from strands_jev import JevRoutingStrategy, JevToolGate


refunds: list[dict] = []


@tool
def issue_refund(order_id: str, amount_cents: int) -> dict:
    """Refund an order."""
    refunds.append({"order_id": order_id, "amount_cents": amount_cents})
    return {"status": "refunded"}


def gate_answers(fake, **probs):
    fake.answers = {k: {"type": "noul", "noul": v} for k, v in probs.items()}


def test_gate_lets_clear_call_run(jev, fake):
    refunds.clear()
    gate_answers(fake, user_requested=0.99, policy_allows=0.97, within_scope=0.98)
    model = ScriptedModel("m", [("tool", "issue_refund", {"order_id": "ORD-1", "amount_cents": 1200}), ("text", "Refunded.")])
    gate = JevToolGate(jev, policy="Refund late shipping.", tools=["issue_refund"])
    agent = Agent(model=model, tools=[issue_refund], interventions=[gate], callback_handler=None)

    result = agent("My order ORD-1 was late, refund the shipping")

    assert refunds == [{"order_id": "ORD-1", "amount_cents": 1200}]
    assert "Refunded" in str(result)
    assert gate.decisions[-1].outcome == "approve"
    # the state Jev saw came from the live agent
    sent = fake.requests[0]["state"]
    assert sent["tool_call"] == {"name": "issue_refund", "input": {"order_id": "ORD-1", "amount_cents": 1200}}
    assert sent["conversation"][0]["text"].startswith("My order ORD-1")


def test_gate_blocks_and_model_sees_reason(jev, fake):
    refunds.clear()
    gate_answers(fake, user_requested=0.95, policy_allows=0.02, within_scope=0.9)
    model = ScriptedModel("m", [("tool", "issue_refund", {"order_id": "ORD-2", "amount_cents": 40000}), ("text", "Sorry, no.")])
    gate = JevToolGate(jev, policy="Refund late shipping.", tools=["issue_refund"])
    agent = Agent(model=model, tools=[issue_refund], interventions=[gate], callback_handler=None)

    agent("Refund my espresso machine ORD-2 in full")

    assert refunds == []
    assert gate.decisions[-1].outcome == "block"
    # the denial was returned to the model as a tool result containing the reason
    tool_results = [
        b["toolResult"] for m in agent.messages for b in m["content"] if isinstance(b, dict) and "toolResult" in b
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "error"
    assert "policy_allows=0.02" in json.dumps(tool_results[0]["content"])


def test_gate_review_with_reviewer_denial(jev, fake):
    refunds.clear()
    gate_answers(fake, user_requested=0.95, policy_allows=0.5, within_scope=0.9)
    model = ScriptedModel("m", [("tool", "issue_refund", {"order_id": "ORD-3", "amount_cents": 8900}), ("text", "Escalated.")])
    gate = JevToolGate(jev, policy="p", tools=["issue_refund"], reviewer=lambda d: "n")
    agent = Agent(model=model, tools=[issue_refund], interventions=[gate], callback_handler=None)

    agent("Refund ORD-3, the box was crushed")

    assert refunds == []
    assert gate.decisions[-1].outcome == "review"


def test_router_uses_model_jev_picks(jev, fake):
    fake.answers = {"pick": {"type": "choice", "choice": "powerful", "confidence": 0.9, "probabilities": {"fast": 0.1, "powerful": 0.9}}}
    fast = ScriptedModel("fast", [("text", "fast answer")])
    powerful = ScriptedModel("powerful", [("text", "powerful answer")])
    router = ModelRouter(
        [
            RoutingCandidate(fast, name="fast", description="simple"),
            RoutingCandidate(powerful, name="powerful", description="hard"),
        ],
        strategy=JevRoutingStrategy(jev),
    )
    agent = Agent(model=router, system_prompt="assistant", callback_handler=None)

    result = agent("Design a distributed cache invalidation scheme")

    assert "powerful answer" in str(result)
    assert (fast.calls, powerful.calls) == (0, 1)
    assert fake.requests[0]["state"]["request"] == "Design a distributed cache invalidation scheme"


def test_router_falls_back_to_default_when_jev_fails(jev, fake):
    fake.status_code = 500
    fast = ScriptedModel("fast", [("text", "fast answer")])
    powerful = ScriptedModel("powerful", [("text", "powerful answer")])
    router = ModelRouter(
        [RoutingCandidate(fast, name="fast"), RoutingCandidate(powerful, name="powerful")],
        strategy=JevRoutingStrategy(jev),
    )
    agent = Agent(model=router, callback_handler=None)

    result = agent("hello")

    assert "fast answer" in str(result)
    assert (fast.calls, powerful.calls) == (1, 0)
