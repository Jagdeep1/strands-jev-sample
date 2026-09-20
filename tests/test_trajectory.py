import asyncio
from types import SimpleNamespace

from helpers import ScriptedModel
from strands import Agent, tool
from strands.interventions import Deny, Guide, Proceed

from strands_jev import MARKER, JevTrajectoryValidator, read_trajectory

TASK = "Send reminders for every overdue invoice of customer C-102."


def msgs(*extra):
    """User task, then a completed list_invoices call, then whatever extra blocks are given."""
    base = [
        {"role": "user", "content": [{"text": TASK}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "list_invoices", "input": {"customer_id": "C-102"}}}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "success",
                        "content": [{"json": [{"invoice_id": "INV-2", "status": "overdue"}, {"invoice_id": "INV-3", "status": "open"}]}],
                    }
                }
            ],
        },
    ]
    return base + list(extra)


def event(tool_name, tool_input, messages=None):
    agent = SimpleNamespace(messages=messages or msgs(), system_prompt="AR assistant", tool_registry=None)
    return SimpleNamespace(agent=agent, tool_use={"toolUseId": "tX", "name": tool_name, "input": tool_input}, invocation_state={})


def nouls(**p):
    return {k: {"type": "noul", "noul": v} for k, v in p.items()}


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- reading history


def test_read_trajectory_reconstructs_task_steps_and_results():
    traj = read_trajectory(msgs())
    assert traj.task == TASK
    assert len(traj.steps) == 1
    step = traj.steps[0]
    assert (step.tool, step.input, step.status) == ("list_invoices", {"customer_id": "C-102"}, "success")
    assert '"INV-2"' in step.result
    assert traj.final_retries == 0


def test_read_trajectory_ignores_validator_feedback_when_finding_task_and_counts_retries():
    # Strands prefixes injected guidance with the handler name, so the marker is not at position 0
    messages = msgs({"role": "user", "content": [{"text": f"[jev-trajectory-validator] {MARKER} Your answer was not accepted"}]})
    traj = read_trajectory(messages)
    assert traj.task == TASK
    assert traj.final_retries == 1


def test_read_trajectory_only_looks_at_current_task():
    old = [
        {"role": "user", "content": [{"text": "old task"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "o1", "name": "x", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "o1", "status": "success", "content": [{"text": "ok"}]}}]},
        {"role": "assistant", "content": [{"text": "done"}]},
    ]
    traj = read_trajectory(old + msgs())
    assert traj.task == TASK
    assert [s.tool for s in traj.steps] == ["list_invoices"]


# ---------------------------------------------------------------- before_tool_call


def test_clear_call_proceeds_and_jev_sees_history_and_proposal(jev, fake):
    fake.answers = nouls(advances_task=0.97, grounded=0.98, in_scope=0.99)
    v = JevTrajectoryValidator(jev)
    action = run(v.before_tool_call(event("send_reminder", {"invoice_id": "INV-2"})))
    assert isinstance(action, Proceed)

    state = fake.requests[0]["state"]
    assert state["task"] == TASK
    assert state["proposed_call"] == {"tool": "send_reminder", "input": {"invoice_id": "INV-2"}}
    assert state["history"][0]["tool"] == "list_invoices" and state["history"][0]["status"] == "success"
    assert "INV-2" in state["history"][0]["result"]
    assert set(fake.requests[0]["questions"]) == {"advances_task", "grounded", "in_scope"}
    assert v.decisions[-1].outcome == "accept"


def test_pending_call_being_judged_is_not_shown_to_jev_as_history(jev, fake):
    # At before_tool_call time the assistant message with this tool use is already in the history.
    pending = {"role": "assistant", "content": [{"toolUse": {"toolUseId": "tX", "name": "send_reminder", "input": {"invoice_id": "INV-2"}}}]}
    fake.answers = nouls(advances_task=0.97, grounded=0.98, in_scope=0.99)
    v = JevTrajectoryValidator(jev)
    action = run(v.before_tool_call(event("send_reminder", {"invoice_id": "INV-2"}, msgs(pending))))
    assert isinstance(action, Proceed)
    history = fake.requests[0]["state"]["history"]
    assert [h["tool"] for h in history] == ["list_invoices"]
    assert all(h["status"] != "pending" for h in history)


def test_out_of_scope_call_is_denied(jev, fake):
    fake.answers = nouls(advances_task=0.4, grounded=0.95, in_scope=0.03)
    v = JevTrajectoryValidator(jev)
    action = run(v.before_tool_call(event("send_reminder", {"invoice_id": "INV-3"})))
    assert isinstance(action, Deny)
    assert "in_scope=0.03" in action.reason
    assert v.decisions[-1].outcome == "deny"


def test_uncertain_call_is_guided_not_denied(jev, fake):
    fake.answers = nouls(advances_task=0.6, grounded=0.95, in_scope=0.95)
    v = JevTrajectoryValidator(jev)
    action = run(v.before_tool_call(event("get_customer", {"customer_id": "C-102"})))
    assert isinstance(action, Guide)
    assert "advances_task=0.60" in action.feedback
    assert action.feedback.startswith(MARKER)


def test_exact_repeat_of_successful_call_is_guided_without_jev(jev, fake):
    v = JevTrajectoryValidator(jev)
    action = run(v.before_tool_call(event("list_invoices", {"customer_id": "C-102"})))
    assert isinstance(action, Guide)
    assert "already called list_invoices" in action.feedback
    assert fake.requests == []


REFUSED_TWICE = [
    {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t2", "name": "send_reminder", "input": {"invoice_id": "INV-3"}}}]},
    {"role": "user", "content": [{"toolResult": {"toolUseId": "t2", "status": "error", "content": [{"text": f"GUIDANCE: {MARKER} no"}]}}]},
    {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t3", "name": "send_reminder", "input": {"invoice_id": "INV-3"}}}]},
    {"role": "user", "content": [{"toolResult": {"toolUseId": "t3", "status": "error", "content": [{"text": f"DENIED: {MARKER} no"}]}}]},
]


def test_repeatedly_refused_call_is_denied_without_jev(jev, fake):
    v = JevTrajectoryValidator(jev, max_guides_per_call=2)
    action = run(v.before_tool_call(event("send_reminder", {"invoice_id": "INV-3"}, msgs(*REFUSED_TWICE))))
    assert isinstance(action, Deny)
    assert "turned back 2 times" in action.reason
    assert fake.requests == []


def test_validator_refusals_are_hidden_from_jev_but_real_tool_errors_are_not(jev, fake):
    real_error = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t4", "name": "get_customer", "input": {"customer_id": "C-999"}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t4", "status": "error", "content": [{"text": "unknown customer C-999"}]}}]},
    ]
    fake.answers = nouls(advances_task=0.95, grounded=0.95, in_scope=0.95)
    v = JevTrajectoryValidator(jev, max_guides_per_call=5)
    # a retry with a different message: under the cap, so Jev is asked, without seeing the earlier refusals
    action = run(v.before_tool_call(event("send_reminder", {"invoice_id": "INV-3", "message": "hi"}, msgs(*REFUSED_TWICE, *real_error))))
    assert isinstance(action, Proceed)
    history = fake.requests[0]["state"]["history"]
    assert [(h["tool"], h["status"]) for h in history] == [("list_invoices", "success"), ("get_customer", "error")]


def test_unvalidated_tool_is_skipped(jev, fake):
    v = JevTrajectoryValidator(jev, tools=["send_reminder"])
    assert isinstance(run(v.before_tool_call(event("get_customer", {"customer_id": "C-102"}))), Proceed)
    assert fake.requests == []


def test_jev_outage_fails_open_by_default_and_closed_when_asked(jev, fake):
    fake.status_code = 503
    assert isinstance(run(JevTrajectoryValidator(jev).before_tool_call(event("send_reminder", {"invoice_id": "INV-2"}))), Proceed)
    action = run(JevTrajectoryValidator(jev, on_jev_error="deny").before_tool_call(event("send_reminder", {"invoice_id": "INV-2"})))
    assert isinstance(action, Deny)


# ---------------------------------------------------------------- after_model_call


def final_event(text, messages=None, stop_reason="end_turn"):
    agent = SimpleNamespace(messages=messages or msgs(), system_prompt=None, tool_registry=None)
    stop = SimpleNamespace(message={"role": "assistant", "content": [{"text": text}]}, stop_reason=stop_reason)
    return SimpleNamespace(agent=agent, stop_response=stop, exception=None, retry=False, invocation_state={})


def test_grounded_final_answer_proceeds(jev, fake):
    fake.answers = nouls(task_completed=0.95, answer_grounded=0.97)
    v = JevTrajectoryValidator(jev)
    action = run(v.after_model_call(final_event("Sent a reminder for INV-2.")))
    assert isinstance(action, Proceed)
    state = fake.requests[0]["state"]
    assert state["final_answer"] == "Sent a reminder for INV-2."
    assert state["history"][0]["tool"] == "list_invoices"


def test_ungrounded_final_answer_is_sent_back_once(jev, fake):
    fake.answers = nouls(task_completed=0.2, answer_grounded=0.05)
    v = JevTrajectoryValidator(jev, max_final_retries=1)
    action = run(v.after_model_call(final_event("Sent reminders for INV-2 and INV-3.")))
    assert isinstance(action, Guide)
    assert action.feedback.startswith(MARKER)

    # a second attempt after one retry is let through, so the loop always converges
    retried = msgs({"role": "user", "content": [{"text": f"{MARKER} Your answer was not accepted"}]})
    action = run(v.after_model_call(final_event("Sent reminders for INV-2 and INV-3.", retried)))
    assert isinstance(action, Proceed)
    assert v.decisions[-1].outcome == "skipped"
    assert len(fake.requests) == 1


def test_tool_use_stops_and_runs_without_tools_are_not_checked(jev, fake):
    v = JevTrajectoryValidator(jev)
    assert isinstance(run(v.after_model_call(final_event("x", stop_reason="tool_use"))), Proceed)
    no_tools = [{"role": "user", "content": [{"text": "hi"}]}]
    assert isinstance(run(v.after_model_call(final_event("hello", no_tools))), Proceed)
    assert fake.requests == []


# ---------------------------------------------------------------- end to end through Agent

sent: list[str] = []


@tool
def list_invoices(customer_id: str) -> list[dict]:
    """List a customer's invoices."""
    return [{"invoice_id": "INV-2", "status": "overdue"}, {"invoice_id": "INV-3", "status": "open"}]


@tool
def send_reminder(invoice_id: str) -> dict:
    """Send a payment reminder for an invoice."""
    sent.append(invoice_id)
    return {"sent": invoice_id}


def test_agent_run_denies_out_of_scope_call_and_lets_good_call_through(jev, fake):
    sent.clear()
    fake.queue = [
        nouls(advances_task=0.98, grounded=0.99, in_scope=0.99),  # list_invoices
        nouls(advances_task=0.97, grounded=0.98, in_scope=0.98),  # send_reminder INV-2
        nouls(advances_task=0.3, grounded=0.9, in_scope=0.02),  # send_reminder INV-3 -> deny
        nouls(task_completed=0.95, answer_grounded=0.96),  # final answer
    ]
    model = ScriptedModel(
        "m",
        [
            ("tool", "list_invoices", {"customer_id": "C-102"}),
            ("tool", "send_reminder", {"invoice_id": "INV-2"}),
            ("tool", "send_reminder", {"invoice_id": "INV-3"}),
            ("text", "Reminder sent for INV-2; INV-3 is not overdue so nothing was sent."),
        ],
    )
    v = JevTrajectoryValidator(jev)
    agent = Agent(model=model, tools=[list_invoices, send_reminder], interventions=[v], callback_handler=None)

    result = agent(TASK)

    assert sent == ["INV-2"]
    assert "INV-3 is not overdue" in str(result)
    assert [d.outcome for d in v.decisions] == ["accept", "accept", "deny", "accept"]
    # the model saw the denial as the tool result
    denied = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b and b["toolResult"]["status"] == "error"]
    assert len(denied) == 1 and "DENIED" in denied[0]["content"][0]["text"]


def test_agent_run_retries_final_answer_once_with_feedback(jev, fake):
    sent.clear()
    fake.queue = [
        nouls(advances_task=0.98, grounded=0.99, in_scope=0.99),  # list_invoices
        nouls(task_completed=0.1, answer_grounded=0.05),  # first answer claims work not done
        nouls(task_completed=0.9, answer_grounded=0.95),  # would be asked for the revised answer if under cap
    ]
    model = ScriptedModel(
        "m",
        [
            ("tool", "list_invoices", {"customer_id": "C-102"}),
            ("text", "All reminders were sent."),
            ("text", "I listed the invoices; INV-2 is overdue but I have not sent any reminder yet."),
        ],
    )
    v = JevTrajectoryValidator(jev, max_final_retries=1)
    agent = Agent(model=model, tools=[list_invoices], interventions=[v], callback_handler=None)

    result = agent(TASK)

    assert "have not sent any reminder" in str(result)
    assert model.calls == 3
    feedback = [b["text"] for m in agent.messages if m["role"] == "user" for b in m["content"] if "text" in b and MARKER in b["text"]]
    assert len(feedback) == 1 and "not accepted" in feedback[0]
    assert [d.stage + ":" + d.outcome for d in v.decisions] == ["call:accept", "final:deny", "final:skipped"]
