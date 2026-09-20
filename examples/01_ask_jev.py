"""Example 1: call Jev directly. State in, typed answers with probabilities out.

    python examples/01_ask_jev.py
"""

from common import jev_client

from strands_jev import Choice, Noul, Score

ticket = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. "
    "I'm losing sales. Please help ASAP."
)

decision = jev_client().decide_sync(
    state=ticket,
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated does the customer appear?",
            criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
        ),
        "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity."),
    },
)

dept = decision.choice("department")
frustration = decision.score("frustration")
print(f"department : {dept.choice}  (confidence {dept.confidence:.2f}, probabilities {dept.probabilities})")
print(f"frustration: level {frustration.level} = {frustration.label!r}  (score {frustration.score:.2f})")
print(f"is_urgent  : p={decision.noul('is_urgent'):.2f}")
print(f"usage      : {decision.usage.input_tokens} in / {decision.usage.output_tokens} out, cost ${decision.usage.cost}")
