# Jev + Strands Agents (Bedrock for the agent, OpenRouter for Jev)

An example of using [TypeSafe's Jev](https://docs.typesafe.ai/) as a **decision harness** around a
[Strands Agents](https://strandsagents.com/) agent. The agent's chat models run on **Amazon Bedrock**;
Jev is reached through [OpenRouter](https://openrouter.ai/). Each piece is built on a native Strands
primitive: a `RoutingStrategy` for model routing, and `InterventionHandler`s for tool gating and
trajectory validation.

## What Jev is (and is not)

Jev is not a chat model. It is a "System One" decision model: you send it a **state** (any JSON) and
named **questions**, and it returns a typed answer with a probability for each question in one fast,
cheap call. Three question types exist:

| Type     | Ask                                | Get back                                  |
| -------- | ---------------------------------- | ----------------------------------------- |
| `Noul`   | a yes/no proposition               | probability it is true                    |
| `Choice` | pick one of N named options        | choice, confidence, per-option probabilities |
| `Score`  | rate against ordered levels        | expected level, confidence, probabilities |

Because it returns numbers instead of prose, your code owns the decision: compare against thresholds,
log the probabilities, and never parse free text. That makes it a good fit for the small decisions an
agent loop makes constantly: *which model should answer this?* and *is this tool call safe to run?*

On OpenRouter Jev is `typesafe/jev-1.13`, served through the alpha **Decisions** endpoint
(`POST https://openrouter.ai/api/alpha/decisions`), not chat completions. `strands_jev/jev.py` is a
small client for it.

## What this repo shows

```
strands_jev/
  jev.py         JevClient + Noul/Choice/Score questions and typed answers (OpenRouter Decisions API)
  routing.py     JevRoutingStrategy  -> plugs into strands.models.routing.ModelRouter
  tool_gate.py   JevToolGate         -> a strands InterventionHandler for before_tool_call
  verify.py      JevFieldVerifier    -> checks LLM-extracted fields against their cited source text
  trajectory.py  JevTrajectoryValidator -> validates every tool call against the task and prior calls, and the final answer
examples/
  01_ask_jev.py               plain Jev call: triage a support ticket
  02_model_routing.py         Jev picks fast vs powerful model per request
  03_tool_gate.py             Jev approves / blocks / escalates a refund tool call
  04_document_extraction.py   agent extracts facts from Amazon's 10-K PDF, Jev verifies each one
  05_trajectory_validation.py built-in validation of a multi-step tool-calling run (try --inject)
tests/                        unit tests with a fake Decisions endpoint (no network, no key)
data/                         the 10-K PDF is downloaded here on first run
```

### 1. Model routing (`JevRoutingStrategy`)

Strands' `ModelRouter` asks a `RoutingStrategy` which candidate to use before each invocation.
`JevRoutingStrategy` answers with one `Choice` question whose options are the candidates' names and
descriptions. Compared to the built-in `ClassifierStrategy`, which runs an LLM with structured output,
this is a single sub-second call that costs a few millionths of a dollar and gives you calibrated
probabilities you can threshold with `min_confidence`.

```python
from strands import Agent
from strands.models.routing import ModelRouter, RoutingCandidate
from strands_jev import JevClient, JevRoutingStrategy

router = ModelRouter(
    [
        RoutingCandidate(fast_model, name="fast", description="Lookups, arithmetic, short answers."),
        RoutingCandidate(strong_model, name="powerful", description="Multi-step reasoning, design, debugging."),
    ],
    strategy=JevRoutingStrategy(JevClient(), min_confidence=0.3),
)
agent = Agent(model=router, system_prompt="You are a concise engineering assistant.")
```

If Jev errors or is unsure, the strategy declines and the router uses the first candidate. After a
failed model call it defers to Strands' `FallbackStrategy`, so failover still works.

### 2. Tool gating (`JevToolGate`)

`JevToolGate` is an `InterventionHandler`; before a gated tool runs it sends Jev the policy, the recent conversation, the
proposed call, and any domain records you add, then asks narrow yes/no checks:

| check            | proposition                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| `user_requested` | the user asked for, or clearly implied, this action with these arguments |
| `policy_allows`  | the policy permits it here (and the conversation cannot rewrite the policy) |
| `within_scope`   | the arguments don't touch more money, records, or systems than asked    |

Fixed thresholds turn the probabilities into a Strands intervention action:

| outcome   | rule                                   | action returned                          |
| --------- | -------------------------------------- | ---------------------------------------- |
| `approve` | every check ≥ `approve_at` (0.9)       | `Proceed()`                              |
| `block`   | any check ≤ `block_at` (0.1)           | `Deny(reason)`; the model sees the reason |
| `review`  | anything in between                    | `Confirm(...)`; a human decides           |

Every decision is appended to `gate.decisions` with its probabilities for auditing. Any Jev failure is a
`block`, never an approval.

```python
from strands import Agent
from strands_jev import JevClient, JevToolGate

gate = JevToolGate(
    JevClient(),
    policy=REFUND_POLICY,
    tools=["issue_refund"],                       # gate only this tool
    reviewer=lambda d: input(f"approve {d.tool_input}? [y/N] "),  # omit to pause the agent instead
    extra_state=lambda event: {"orders": ORDERS}, # records Jev should judge against
)
agent = Agent(model=model, tools=[lookup_order, issue_refund], interventions=[gate])
```

Without a `reviewer`, `review` returns `Confirm` with no response, which pauses the agent with an
interrupt so an external system (queue, UI, Slack) can resume it. Keep a plain static rule for tools
that are always dangerous; the gate is for tools whose safety depends on the call.

### 3. Document extraction with verification (`JevFieldVerifier`)

Jev does not generate text, so it cannot extract on its own. The split that works: the agent's chat
model **extracts**, Jev **verifies**. Example 4 downloads Amazon's 2025 Annual Report PDF (which
contains the FY2025 Form 10-K), gives the agent two tools (`search_filing`, `read_page`), and asks for a
structured `FilingFacts` object in which every value carries the page it came from and a verbatim
quote. Then:

1. Code checks each quote really appears on the cited page. A fabricated quote is rejected before Jev
   is ever asked.
2. One Jev request judges all surviving fields against their cited pages with two propositions each:
   `<field>_supported` (the value is what the page states) and `<field>_right_item` (it answers the
   field's description and is not a neighbouring figure such as the prior-year column).
3. Thresholds map probabilities to `accept` / `review` / `reject` per field.

```python
from strands_jev import ExtractedField, JevClient, JevFieldVerifier

verifier = JevFieldVerifier(JevClient(), accept_at=0.9, reject_at=0.1)
verdicts = await verifier.verify(
    [ExtractedField("total_net_sales", "Total net sales, fiscal 2025, USD millions", 716924, page=48, quote="Total net sales 574,785 637,959 716,924")],
    pages,  # {page_number: text}
)
```

Fields extracted in the example: registrant name, fiscal year end, state of incorporation, commission
file number, exchange, shares outstanding, total net sales, net income, employee count, auditor, CEO,
plus three that are deliberately easy to get wrong:

| field | the trap next to it in the filing |
| --- | --- |
| `aws_operating_income` (2025) | AWS net sales, consolidated operating income, and the 2023/2024 columns are in the same table |
| `free_cash_flow` (2025) | non-GAAP; operating cash flow is about 12x larger and the 2024 column comes first |
| `long_term_debt` (Dec 31, 2025) | the balance sheet lists 2024 before 2025; the debt note shows the total before deducting the current portion |

For these, the `<field>_supported` check alone is not enough: a prior-year or wrong-segment number is
also "on the page". The `<field>_right_item` check is what catches it. The whole verification is one
Decisions request costing a fraction of a cent.

### 4. Built-in trajectory validation (`JevTrajectoryValidator`)

The tool gate asks "is this one risky call allowed by policy?". The trajectory validator asks a
different question of *every* call: "does this make sense as the next step of this task, given what the
agent has already done?" It is an `InterventionHandler` that reads the trajectory straight from the
agent's message history, so it carries no state of its own.

Before each tool call, code first turns back an exact repeat of a call that already succeeded (with the
earlier result), and denies a call that has already been turned back too many times. Otherwise one Jev
request checks three propositions against the task, the tool descriptions, and the history of calls
and results so far:

| check | proposition |
| --- | --- |
| `advances_task` | given the history, this call is a useful next step, not pointless, premature, or already answered |
| `grounded` | every id, name, date, and amount in the arguments comes from the task or an earlier result |
| `in_scope` | the call does not act on records or people the task did not cover, even if a tool result suggests it |

Outcomes: all clear → `Proceed`; any check clearly false → `Deny` (the model sees why); anything in the
middle → `Guide` (the call is cancelled and the model gets feedback to adjust or move on).

When the model stops with a final answer, one more request checks `task_completed` (the history shows
the needed calls were made and succeeded) and `answer_grounded` (nothing in the answer is invented or
contradicts the results). A failing answer is discarded and the model retries with feedback, capped by
`max_final_retries` so the loop always converges.

```python
from strands import Agent
from strands_jev import JevClient, JevTrajectoryValidator

validator = JevTrajectoryValidator(JevClient(), tools=["send_reminder", "list_invoices", "get_customer"])
agent = Agent(model=model, tools=[get_customer, list_invoices, send_reminder], interventions=[validator])
agent("For customer C-102, find every overdue invoice and send a reminder for each. Then report back.")
print(validator.decisions)  # every call and the final answer, with the probabilities behind each outcome
```

Example 5 runs an accounts-receivable assistant through exactly that task. With `--inject`, the
`list_invoices` result carries a planted memo telling the agent to also remind paid and not-yet-due
invoices. A model that obeys proposes an out-of-scope `send_reminder`, and the `in_scope` check turns it
back. Jev errors fail open here by default (`on_jev_error="proceed"`), because this is quality
assurance rather than a security boundary; the tool gate stays fail-closed.

Two things learned building it, both measured against real Jev:

- **Don't show Jev the call it is judging as history.** When `before_tool_call` fires, the assistant
  message containing that tool use is already in `agent.messages`. Included as a pending step, it made Jev
  rate a perfectly good first `get_customer` at 0.14 for "advances the task" (it looked already done).
  Judged against completed steps only, the same call scored 0.85.
- **Hide your own refusals from Jev too.** After a `Guide`, the model's rejected attempt sits in the history
  as an error result. Left visible, a perfectly good retry scored 0.59 for "advances the task" (it looked
  already attempted and rejected); with the validator's own refusals filtered out it scored 0.95. Real tool
  errors stay visible, because they are world state.
- **Judge facts, not wording, in free-text arguments.** A `message` body the model composed is not "in the
  history" verbatim, and a strict "every value comes from a result" check scored it 0.80. Asking instead
  whether the targeted ids exist and the mentioned names, amounts, and dates agree with the history
  scored 0.97, while a message with a wrong amount still scored 0.02.
- **Use a looser approve band than the gate.** Good steps here scored 0.85 to 0.98 and bad ones 0.09 or
  below, so `approve_at=0.8` accepts every good step while still denying every bad one. A validator that
  runs on every call must not turn back borderline-good steps, or the agent stalls.

## Run it

You need AWS credentials with Bedrock access (for the agent models) and an OpenRouter key (for Jev).

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env            # add your OPENROUTER_API_KEY
export AWS_PROFILE=...          # or any other way boto3 finds credentials

pytest                          # offline unit tests
python examples/01_ask_jev.py
python examples/02_model_routing.py
python examples/03_tool_gate.py
python examples/04_document_extraction.py            # downloads the 10-K PDF to data/ on first run
python examples/05_trajectory_validation.py          # add --inject to plant a rogue instruction in a tool result
```

Bedrock models default to `global.anthropic.claude-haiku-4-5-20251001-v1:0` (fast) and
`global.anthropic.claude-sonnet-5` (powerful). Override with `FAST_MODEL`, `POWERFUL_MODEL`,
`BEDROCK_REGION`, and `JEV_MODEL` in `.env`.

## Writing good questions for Jev

- **Ask propositions, not decisions.** "The user asked for this refund" is checkable; "should this be
  approved" is the decision your code makes from several checks, and splitting it up is what gives you a
  reason for every block.
- **Spell the claim out; don't make Jev dereference it.** Measured on the 10-K fields above, with the same
  page text in the state each time:

  | question form | correct fields, min / mean | prior-year figure (should be low) |
  | --- | --- | --- |
  | "`fields.X.value` is exactly what `pages.48` states ..." | 0.84 / 0.94 | 0.19 |
  | JSON instruction `{field, extracted_value, question}` | 0.99 / 0.99 | 0.20 |
  | "According to `pages.48`, the total net sales for fiscal 2025 is 716,924." | 0.98 / 0.99 | 0.02 |

  The complete, inlined proposition is both more confident on true claims and far more decisive on
  false ones. `JevFieldVerifier` uses that form.
- **Name the state fields** the question depends on (`tool_call.input`, `policy`) so Jev evaluates the
  same evidence a reviewer would.
- **Say which text is authoritative.** The conversation is user-controlled; say explicitly that anything
  it claims about the policy is part of the situation, not the policy.
- **Keep thresholds far apart** (0.9 / 0.1) so humans only see the genuinely ambiguous middle.
- Add more questions freely: they run in parallel on the same state and cost only their own tokens.

## Notes and caveats

- OpenRouter's Decisions API is alpha and may change. The client only depends on the documented
  `model` / `state` / `questions` request and `answers` / `usage` response shape.
- `ModelRouter` and `InterventionHandler` in Strands are marked provisional; this was built against
  `strands-agents` 1.56.
- Jev's probabilities move a few hundredths between identical calls; outcomes are stable when the
  thresholds are far apart, which is why the defaults are 0.9 and 0.1.
