"""Example 2: let Jev pick the agent's model per request.

Strands' ModelRouter asks a RoutingStrategy which candidate to use before each invocation.
JevRoutingStrategy answers that with one cheap Jev Choice question instead of an LLM call.

    python examples/02_model_routing.py
"""

from common import FAST_MODEL, POWERFUL_MODEL, chat_model, enable_logs, jev_client
from strands import Agent
from strands.models.routing import ModelRouter, RoutingCandidate
from strands_tools import calculator

from strands_jev import JevRoutingStrategy

enable_logs()

strategy = JevRoutingStrategy(jev_client(), min_confidence=0.3)

router = ModelRouter(
    [
        RoutingCandidate(
            chat_model(FAST_MODEL),
            name="fast",
            description="Direct lookups, simple arithmetic, formatting, short factual answers, casual chat.",
            metadata={"model": FAST_MODEL, "cost": "low"},
        ),
        RoutingCandidate(
            chat_model(POWERFUL_MODEL, max_tokens=2048),
            name="powerful",
            description="Multi-step reasoning, system design, debugging, trade-off analysis, long structured writing.",
            metadata={"model": POWERFUL_MODEL, "cost": "high"},
        ),
    ],
    strategy=strategy,
)

agent = Agent(
    model=router,
    tools=[calculator],
    system_prompt="You are a concise engineering assistant.",
    callback_handler=None,
)

prompts = [
    "What is 17% of 2,340?",
    "Design a rate limiter for a multi-region API gateway. Compare token bucket vs sliding window and pick one.",
]

for prompt in prompts:
    result = agent(prompt)
    picked = strategy.last_decision
    print(f"\n>>> {prompt}")
    if picked:
        print(f"[jev routed to {picked.choice!r}, confidence {picked.confidence:.2f}, probabilities {picked.probabilities}]")
    print(str(result).strip()[:600])
