import asyncio

import pytest
from strands.models.routing import RoutingAttempt, RoutingCandidate, RoutingContext

from strands_jev import JevRoutingStrategy


class DummyModel:
    def __init__(self, name):
        self.name = name


def candidates():
    return (
        RoutingCandidate(DummyModel("fast"), name="fast", description="Lookups, formatting, short answers."),
        RoutingCandidate(
            DummyModel("powerful"),
            name="powerful",
            description="Multi-step reasoning, architecture, debugging.",
            metadata={"cost": "high"},
        ),
    )


def context(text="What's 2+2?", attempts=(), cands=None):
    return RoutingContext(
        messages=[
            {"role": "user", "content": [{"text": "earlier question"}]},
            {"role": "assistant", "content": [{"text": "earlier answer"}]},
            {"role": "user", "content": [{"text": text}]},
        ],
        system_prompt="You are a coding assistant.",
        tool_specs=(),
        candidates=cands or candidates(),
        invocation_state={},
        attempts=attempts,
    )


def choice(name, probs, confidence=0.9):
    return {"pick": {"type": "choice", "choice": name, "confidence": confidence, "probabilities": probs}}


def test_selects_candidate_named_by_jev(jev, fake):
    fake.answers = choice("powerful", {"fast": 0.2, "powerful": 0.8})
    strategy = JevRoutingStrategy(jev)
    ctx = context("Design a sharded Postgres schema for multi-tenant billing")
    picked = asyncio.run(strategy.select(ctx))
    assert picked is ctx.candidates[1]

    sent = fake.requests[0]
    assert sent["state"]["request"] == "Design a sharded Postgres schema for multi-tenant billing"
    assert sent["state"]["agent_instructions"] == "You are a coding assistant."
    q = sent["questions"]["pick"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"fast", "powerful"}
    assert q["criteria"]["powerful"]["description"].startswith("Multi-step")
    assert q["criteria"]["powerful"]["metadata"] == {"cost": "high"}

    assert strategy.last_decision.choice == "powerful"


def test_low_confidence_declines_so_router_uses_default(jev, fake):
    fake.answers = choice("powerful", {"fast": 0.48, "powerful": 0.52}, confidence=0.1)
    strategy = JevRoutingStrategy(jev, min_confidence=0.5)
    assert asyncio.run(strategy.select(context())) is None


def test_jev_error_declines_instead_of_raising(jev, fake):
    fake.status_code = 500
    strategy = JevRoutingStrategy(jev)
    assert asyncio.run(strategy.select(context())) is None


def test_single_candidate_skips_jev(jev, fake):
    strategy = JevRoutingStrategy(jev)
    ctx = context(cands=candidates()[:1])
    assert asyncio.run(strategy.select(ctx)) is ctx.candidates[0]
    assert fake.requests == []


def test_after_failure_falls_back_to_untried_candidate(jev, fake):
    strategy = JevRoutingStrategy(jev)
    cands = candidates()
    ctx = context(attempts=(RoutingAttempt(cands[1], exception=RuntimeError("boom")),), cands=cands)
    picked = asyncio.run(strategy.select(ctx))
    assert picked is cands[0]
    assert fake.requests == []


def test_unnamed_candidates_are_a_configuration_error(jev):
    strategy = JevRoutingStrategy(jev)
    cands = (RoutingCandidate(DummyModel("a")), RoutingCandidate(DummyModel("b"), name="b"))
    with pytest.raises(ValueError, match="name"):
        asyncio.run(strategy.select(context(cands=cands)))
