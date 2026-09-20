"""Shared fixtures: a fake OpenRouter Decisions endpoint backed by httpx.MockTransport."""

import json

import httpx
import pytest

from strands_jev import JevClient

DECISIONS_PATH = "/api/alpha/decisions"


class FakeDecisions:
    """Records requests and returns scripted answers."""

    def __init__(self):
        self.requests: list[dict] = []
        self.answers: dict = {}
        self.queue: list[dict] = []  # per-request answers, consumed in order; falls back to `answers`
        self.status_code = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == DECISIONS_PATH
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status_code != 200:
            return httpx.Response(self.status_code, json={"error": {"code": self.status_code, "message": "nope"}})
        return httpx.Response(
            200,
            json={
                "id": "gen-dec-1",
                "model": "typesafe/jev-1.13-20260917",
                "provider": "TypeSafe",
                "answers": self.queue.pop(0) if self.queue else self.answers,
                "usage": {"input_tokens": 10, "output_tokens": 3, "cost": 0.0000004},
            },
        )


@pytest.fixture
def fake():
    return FakeDecisions()


@pytest.fixture
def jev(fake):
    return JevClient(api_key="test-key", transport=httpx.MockTransport(fake.handler))
