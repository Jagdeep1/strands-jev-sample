"""JevRoutingStrategy: pick the agent's model per request with one Jev Choice question.

Strands' ``ModelRouter`` delegates every routing decision to a ``RoutingStrategy``. This strategy lets
Jev make that decision: it reads the latest user request plus each candidate's name/description and
returns a choice with probabilities, in a few
hundred milliseconds and for a fraction of a cent. Failures decline, so the router falls back to its
default (first) candidate rather than breaking the agent.

Usage::

    router = ModelRouter(
        [
            RoutingCandidate(fast_model, name="fast", description="Lookups, formatting, short answers."),
            RoutingCandidate(strong_model, name="powerful", description="Multi-step reasoning, debugging."),
        ],
        strategy=JevRoutingStrategy(jev),
    )
    agent = Agent(model=router, ...)
"""

from __future__ import annotations

import logging
from typing import Any

from strands.models.routing import FallbackStrategy, RoutingCandidate, RoutingContext

from .jev import Choice, ChoiceAnswer, JevClient, JevError

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = (
    "Choose the least capable candidate that can still fully and accurately handle `request`. Judge only the "
    "request's real requirements (reasoning depth, ambiguity, multi-step work, risk of error). Reserve more "
    "capable candidates for requests that need them. `agent_instructions` describe the agent the candidate "
    "will run as. Candidate order carries no meaning."
)


class JevRoutingStrategy:
    """Route with a Jev ``Choice`` over the candidates; fall back on failure."""

    def __init__(
        self,
        jev: JevClient,
        *,
        instructions: str = DEFAULT_INSTRUCTIONS,
        min_confidence: float = 0.0,
        max_request_chars: int = 4000,
        max_instruction_chars: int = 2000,
        fallback: FallbackStrategy | None = None,
    ):
        """
        Args:
            jev: Decisions client.
            instructions: Routing policy given to Jev as the Choice question.
            min_confidence: Decline (use the default candidate) when Jev's confidence is below this.
            max_request_chars: Cap on the request text sent to Jev.
            max_instruction_chars: Cap on the agent system prompt text sent to Jev.
            fallback: Strategy consulted after a failed model call. Defaults to ``FallbackStrategy``.
        """
        self._jev = jev
        self._instructions = instructions
        self._min_confidence = min_confidence
        self._max_request_chars = max_request_chars
        self._max_instruction_chars = max_instruction_chars
        self._fallback = fallback or FallbackStrategy()
        self.last_decision: ChoiceAnswer | None = None

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        if context.attempts:
            return await self._fallback.select(context)
        if len(context.candidates) == 1:
            return context.candidates[0]

        by_name = _index_by_name(context.candidates)
        state = {
            "request": _latest_user_text(context.messages)[: self._max_request_chars],
            "agent_instructions": _system_text(context.system_prompt)[: self._max_instruction_chars],
        }
        question = Choice(
            instructions=self._instructions,
            criteria={name: _profile(candidate) for name, candidate in by_name.items()},
        )

        try:
            decision = await self._jev.decide(state, {"pick": question})
        except JevError as error:
            logger.warning("jev routing failed, using default candidate: %s", error)
            return None

        answer = decision.choice("pick")
        self.last_decision = answer
        logger.info("jev routed to <%s> confidence=%.2f probabilities=%s", answer.choice, answer.confidence, answer.probabilities)
        if answer.confidence < self._min_confidence:
            logger.info("confidence below %.2f, using default candidate", self._min_confidence)
            return None
        return by_name[answer.choice]


def _index_by_name(candidates) -> dict[str, RoutingCandidate]:
    by_name: dict[str, RoutingCandidate] = {}
    for candidate in candidates:
        if not candidate.name:
            raise ValueError("JevRoutingStrategy requires every RoutingCandidate to have a name")
        by_name[candidate.name] = candidate
    return by_name


def _profile(candidate: RoutingCandidate) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if candidate.description:
        profile["description"] = candidate.description
    if candidate.metadata:
        profile["metadata"] = dict(candidate.metadata)
    return profile or {"description": candidate.name}


def _latest_user_text(messages) -> str:
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        text = " ".join(b["text"] for b in message.get("content", []) if isinstance(b, dict) and "text" in b)
        if text.strip():
            return text
    return "[no user text]"


def _system_text(system_prompt) -> str:
    if system_prompt is None:
        return ""
    if isinstance(system_prompt, str):
        return system_prompt
    return " ".join(b.get("text", "") for b in system_prompt if isinstance(b, dict))
