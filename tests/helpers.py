"""Test helpers shared across integration tests."""

import json
from typing import Any

from strands.models import Model


class ScriptedModel(Model):
    """Replays scripted turns: each turn is ("text", str) or ("tool", name, input_dict).

    When the script runs out it answers ("text", "done").
    """

    def __init__(self, name: str, turns: list[tuple]):
        self.model_name = name
        self.turns = list(turns)
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    def update_config(self, **model_config: Any) -> None:  # pragma: no cover
        pass

    def get_config(self) -> Any:
        return {"model_id": self.model_name}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.seen_messages.append([dict(m) for m in messages])
        turn = self.turns.pop(0) if self.turns else ("text", "done")
        yield {"messageStart": {"role": "assistant"}}
        if turn[0] == "tool":
            _, name, tool_input = turn
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.calls}", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_input)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": turn[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 0}}}
