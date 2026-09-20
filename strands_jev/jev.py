"""A small client for TypeSafe's Jev via OpenRouter's Decisions endpoint.

Jev is not a chat model. You hand it a *state* (any JSON) plus named *questions*, and it returns a
typed answer with a probability for each question. All questions are evaluated in one request.

Endpoint: ``POST https://openrouter.ai/api/alpha/decisions`` (alpha; the same OpenRouter key that
serves your chat models works here).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai"
DECISIONS_PATH = "/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"

JSON = str | int | float | bool | None | list["JSON"] | dict[str, "JSON"]


# --------------------------------------------------------------------------- questions


@dataclass(frozen=True)
class Noul:
    """A yes/no proposition. Jev returns the probability that it is true."""

    instructions: JSON
    criteria: Mapping[str, JSON] | None = None  # optional {"true": ..., "false": ...} guidance

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = dict(self.criteria)
        return payload


@dataclass(frozen=True)
class Choice:
    """Pick one option. ``criteria`` maps option name -> description (or None)."""

    instructions: JSON
    criteria: Mapping[str, JSON]

    def to_payload(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score:
    """Rate against ordered levels. ``criteria`` is the list of level descriptions, low to high."""

    instructions: JSON
    criteria: list[JSON]

    def to_payload(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score


# --------------------------------------------------------------------------- answers


@dataclass(frozen=True)
class NoulAnswer:
    noul: float  # probability the proposition is true, 0..1


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # expected level, e.g. 1.2 on a 0..2 scale
    confidence: float
    probabilities: dict[str, float]
    legend: dict[str, str]

    @property
    def level(self) -> int:
        """Nearest integer level."""
        return round(self.score)

    @property
    def label(self) -> str:
        return self.legend.get(str(self.level), str(self.level))


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost: float | None = None


@dataclass(frozen=True)
class Decision:
    """Parsed Decisions response."""

    id: str | None
    model: str
    answers: dict[str, Answer]
    usage: Usage
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def noul(self, key: str) -> float:
        return self._typed(key, NoulAnswer).noul

    def choice(self, key: str) -> ChoiceAnswer:
        return self._typed(key, ChoiceAnswer)

    def score(self, key: str) -> ScoreAnswer:
        return self._typed(key, ScoreAnswer)

    def _typed(self, key: str, kind: type) -> Any:
        answer = self.answers[key]
        if not isinstance(answer, kind):
            raise KeyError(f"answer {key!r} is a {type(answer).__name__}, not a {kind.__name__}")
        return answer


class JevError(RuntimeError):
    """Raised for transport errors, non-2xx responses, or malformed answers.

    Callers that gate actions on Jev should treat this as "unknown", never as approval.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- client


class JevClient:
    """Async-first client; ``decide_sync`` is provided for scripts and tests."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        app_title: str = "strands-jev-example",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY or pass api_key=")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": app_title,
        }

    async def decide(
        self,
        state: JSON,
        questions: Mapping[str, Question],
        *,
        session_id: str | None = None,
        user: str | None = None,
    ) -> Decision:
        if not questions:
            raise ValueError("at least one question is required")
        body: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {name: q.to_payload() for name, q in questions.items()},
        }
        if session_id:
            body["session_id"] = session_id
        if user:
            body["user"] = user

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, headers=self._headers, timeout=self.timeout, transport=self._transport
            ) as client:
                response = await client.post(DECISIONS_PATH, json=body)
        except httpx.HTTPError as error:
            raise JevError(f"Decisions request failed: {error}") from error

        if response.status_code // 100 != 2:
            raise JevError(f"Decisions {response.status_code}: {response.text[:500]}", status=response.status_code)

        try:
            data = response.json()
        except ValueError as error:
            raise JevError("Decisions returned non-JSON body") from error
        return _parse_decision(data, questions)

    def decide_sync(self, state: JSON, questions: Mapping[str, Question], **kwargs: Any) -> Decision:
        return asyncio.run(self.decide(state, questions, **kwargs))


def _parse_decision(data: dict[str, Any], questions: Mapping[str, Question]) -> Decision:
    raw_answers = data.get("answers")
    if not isinstance(raw_answers, dict):
        raise JevError("Decisions response has no answers")

    missing = [name for name in questions if name not in raw_answers]
    if missing:
        raise JevError(f"Jev did not answer: {', '.join(missing)}")

    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        answers[name] = _parse_answer(name, question, raw_answers[name])

    usage_raw = data.get("usage") or {}
    usage = Usage(
        input_tokens=int(usage_raw.get("input_tokens", 0)),
        output_tokens=int(usage_raw.get("output_tokens", 0)),
        cost=usage_raw.get("cost"),
    )
    return Decision(id=data.get("id"), model=str(data.get("model", "")), answers=answers, usage=usage, raw=data)


def _probability(name: str, value: Any) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError) as error:
        raise JevError(f"answer {name!r}: probability is not a number") from error
    if not 0.0 <= p <= 1.0:
        raise JevError(f"answer {name!r}: probability {p} outside [0, 1]")
    return p


def _parse_answer(name: str, question: Question, raw: Any) -> Answer:
    if not isinstance(raw, dict):
        raise JevError(f"answer {name!r} is malformed")
    if isinstance(question, Noul):
        return NoulAnswer(noul=_probability(name, raw.get("noul")))
    probabilities = {str(k): _probability(name, v) for k, v in (raw.get("probabilities") or {}).items()}
    confidence = _probability(name, raw.get("confidence", 0.0))
    if isinstance(question, Choice):
        choice = raw.get("choice")
        if choice not in question.criteria:
            raise JevError(f"answer {name!r}: choice {choice!r} is not one of the offered options")
        return ChoiceAnswer(choice=str(choice), confidence=confidence, probabilities=probabilities)
    try:
        score = float(raw.get("score"))
    except (TypeError, ValueError) as error:
        raise JevError(f"answer {name!r}: score is not a number") from error
    legend = {str(k): str(v) for k, v in (raw.get("legend") or {}).items()}
    if not legend:
        legend = {str(i): str(level) for i, level in enumerate(question.criteria)}
    return ScoreAnswer(score=score, confidence=confidence, probabilities=probabilities, legend=legend)
