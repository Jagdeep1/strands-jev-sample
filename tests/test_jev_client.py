import asyncio

import pytest

from strands_jev import Choice, JevError, Noul, Score


def test_questions_serialize_to_decisions_payload():
    assert Noul(instructions="Is it urgent?").to_payload() == {"type": "noul", "instructions": "Is it urgent?"}
    assert Noul(instructions="q", criteria={"true": "yes", "false": "no"}).to_payload() == {
        "type": "noul",
        "instructions": "q",
        "criteria": {"true": "yes", "false": "no"},
    }
    assert Choice(instructions="Team?", criteria={"billing": "money", "tech": None}).to_payload() == {
        "type": "choice",
        "instructions": "Team?",
        "criteria": {"billing": "money", "tech": None},
    }
    assert Score(instructions="How bad?", criteria=["low", "mid", "high"]).to_payload() == {
        "type": "score",
        "instructions": "How bad?",
        "criteria": ["low", "mid", "high"],
    }


def test_decide_sends_state_and_questions_and_parses_typed_answers(jev, fake):
    fake.answers = {
        "urgent": {"type": "noul", "noul": 0.93},
        "team": {"type": "choice", "choice": "billing", "confidence": 0.8, "probabilities": {"billing": 0.9, "tech": 0.1}},
        "anger": {
            "type": "score",
            "score": 1.2,
            "confidence": 0.7,
            "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
            "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
        },
    }
    decision = asyncio.run(
        jev.decide(
            state={"ticket": "Charged twice, fix it now"},
            questions={
                "urgent": Noul(instructions="Needs attention now?"),
                "team": Choice(instructions="Owning team?", criteria={"billing": None, "tech": None}),
                "anger": Score(instructions="How angry?", criteria=["calm", "annoyed", "furious"]),
            },
            session_id="s-1",
        )
    )

    sent = fake.requests[0]
    assert sent["model"] == "typesafe/jev-1.13"
    assert sent["state"] == {"ticket": "Charged twice, fix it now"}
    assert sent["session_id"] == "s-1"
    assert set(sent["questions"]) == {"urgent", "team", "anger"}
    assert sent["questions"]["team"]["type"] == "choice"

    assert decision.noul("urgent") == 0.93
    assert decision.choice("team").choice == "billing"
    assert decision.choice("team").probabilities["tech"] == 0.1
    assert decision.score("anger").score == 1.2
    assert decision.score("anger").level == 1
    assert decision.score("anger").label == "annoyed"
    assert decision.usage.input_tokens == 10
    assert decision.model.startswith("typesafe/jev-1.13")


def test_decide_sync_wrapper(jev, fake):
    fake.answers = {"ok": {"type": "noul", "noul": 0.5}}
    decision = jev.decide_sync("hello", {"ok": Noul(instructions="fine?")})
    assert decision.noul("ok") == 0.5


def test_non_2xx_raises_jev_error(jev, fake):
    fake.status_code = 429
    with pytest.raises(JevError) as err:
        asyncio.run(jev.decide("x", {"q": Noul(instructions="?")}))
    assert err.value.status == 429


def test_missing_answer_raises(jev, fake):
    fake.answers = {"other": {"type": "noul", "noul": 0.2}}
    with pytest.raises(JevError, match="did not answer"):
        asyncio.run(jev.decide("x", {"q": Noul(instructions="?")}))


def test_noul_out_of_range_raises(jev, fake):
    fake.answers = {"q": {"type": "noul", "noul": 1.7}}
    with pytest.raises(JevError, match="probability"):
        asyncio.run(jev.decide("x", {"q": Noul(instructions="?")}))


def test_wrong_answer_type_accessor_raises(jev, fake):
    fake.answers = {"q": {"type": "noul", "noul": 0.2}}
    decision = asyncio.run(jev.decide("x", {"q": Noul(instructions="?")}))
    with pytest.raises(KeyError):
        decision.choice("q")
