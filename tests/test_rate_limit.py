"""Per-IP rate limit on /api/v1/ask (Adım 5.4).

The decorator is applied at module import time, so the limit is shared
across the test process. We override the rate via the ``ACU_ASK_RATE`` env
var BEFORE the view module imports, which means the rate Django sees in
this file is whatever the test runner started with — instead, we rely on
``django_ratelimit`` reading ``request.META["REMOTE_ADDR"]`` and use
unique fake IPs to reset the bucket between scenarios.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from django.core.cache import cache
from django.test import Client

from chatbot.services.ask_orchestrator import ANSWER_SOURCE_RAG_LLM, AskMeta
from chatbot.services.query_parser import QueryFilters


@pytest.fixture(autouse=True)
def _reset_ratelimit_cache():
    """Each test starts from an empty bucket so order does not cross-contaminate."""
    cache.clear()
    yield
    cache.clear()


def _stub_run_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make /ask succeed instantly — we only care about the ratelimit envelope."""
    def _fake(question: str, conv) -> tuple[dict[str, Any], int, AskMeta]:
        from chatbot.models import Message

        Message.objects.create(conversation=conv, role=Message.ROLE_ASSISTANT, content="ok")
        return (
            {"conversation_id": int(conv.pk), "answer": "ok"},
            200,
            AskMeta(answer_source=ANSWER_SOURCE_RAG_LLM, filters=QueryFilters()),
        )

    monkeypatch.setattr("chatbot.api.v1.views.run_ask", _fake)


def _post_ask(client: Client, *, ip: str | None = None) -> int:
    headers: dict[str, str] = {}
    if ip is not None:
        # Django's test client expects WSGI-style META keys via the ``HTTP_``
        # prefix for headers, plus REMOTE_ADDR for client IP.
        headers["REMOTE_ADDR"] = ip
    response = client.post(
        "/api/v1/ask",
        data=json.dumps({"question": "hi"}),
        content_type="application/json",
        **headers,
    )
    return response.status_code


@pytest.mark.django_db
def test_ask_v1_blocks_after_rate_limit_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Within the same minute, a single IP must hit 429 once the bucket empties.

    The default rate is ``30/m``; we hammer well past it from one fake IP and
    expect at least one 429 in the burst.
    """
    _stub_run_ask(monkeypatch)
    client = Client()

    fake_ip = "203.0.113." + str(uuid.uuid4().int % 250 + 1)  # RFC5737 TEST-NET-3

    statuses = [_post_ask(client, ip=fake_ip) for _ in range(60)]
    assert 200 in statuses, "expected at least one allowed request before the bucket empties"
    assert 429 in statuses, f"never hit 429 in 60 requests: {statuses[-10:]!r}"

    # When 429 fires, the body must follow the v1 error envelope.
    response = client.post(
        "/api/v1/ask",
        data=json.dumps({"question": "hi"}),
        content_type="application/json",
        REMOTE_ADDR=fake_ip,
    )
    assert response.status_code == 429
    body = json.loads(response.content.decode("utf-8"))
    assert body == {
        "error": {
            "code": "rate_limited",
            "message": "Çok sık soru soruyorsunuz, lütfen biraz bekleyin.",
        }
    }


@pytest.mark.django_db
def test_ask_v1_separate_ips_have_independent_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """One spammy IP must not get a different IP throttled (`key='ip'` invariant)."""
    _stub_run_ask(monkeypatch)
    client = Client()

    spam_ip = "203.0.113.10"
    quiet_ip = "203.0.113.20"

    spam_results = [_post_ask(client, ip=spam_ip) for _ in range(60)]
    assert 429 in spam_results, "spam IP should have hit the limit"

    # Quiet IP, fresh bucket: first request must succeed.
    assert _post_ask(client, ip=quiet_ip) == 200
