"""Wire-format regression tests for /api/v1/ask (Adım 5.1).

The chatbot's HTTP contract is consumed by an external Next.js client;
breaking the field set silently is far worse than a noisy CI failure here.
These tests pin the v1 shape by monkey-patching the heavy collaborators
(retriever + LLM) so we can exercise the HTTP boundary without booting
Ollama or sentence-transformers in CI.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from django.test import Client

from chatbot.services.ask_orchestrator import (
    ANSWER_SOURCE_RAG_LLM,
    AskMeta,
)
from chatbot.services.query_parser import QueryFilters


def _stub_run_ask(monkeypatch: pytest.MonkeyPatch, *, answer: str = "stub answer") -> None:
    """Replace ``chatbot.api.v1.views.run_ask`` with a deterministic stub.

    The stub builds a realistic AskMeta (one chunk + one matched filter +
    plausible timings) so assertions cover the full happy-path payload.
    """
    fake_chunk = {
        "chunk_id": 4242,
        "url": "https://obs.example.org/cse101",
        "title": "CSE101 — Introduction",
        "text": "CSE101 is a 6 ECTS course taught in the first semester.",
        "score": 0.812345,
        "metadata": {
            "content_type": "bologna_course",
            "department": "Bilgisayar Mühendisliği",
            "faculty": "Mühendislik ve Doğa Bilimleri Fakültesi",
            "course_code": "CSE101",
            "semester": 1,
        },
    }
    fake_filters = QueryFilters(
        faculty="Mühendislik ve Doğa Bilimleri Fakültesi",
        department="Bilgisayar Mühendisliği",
        course_code="CSE101",
        semester=1,
        content_types=("bologna_course",),
        matched_terms=("dept:Bilgisayar Mühendisliği", "course_code:CSE101"),
    )

    def _fake_run_ask(question: str, conv) -> tuple[dict[str, Any], int, AskMeta]:
        from chatbot.models import Message

        Message.objects.create(conversation=conv, role=Message.ROLE_ASSISTANT, content=answer)
        meta = AskMeta(
            answer_source=ANSWER_SOURCE_RAG_LLM,
            retrieved_chunks=[fake_chunk],
            filters=fake_filters,
            latency_ms={"retrieve": 12, "llm": 540, "total": 600},
        )
        return {"conversation_id": int(conv.pk), "answer": answer}, 200, meta

    monkeypatch.setattr("chatbot.api.v1.views.run_ask", _fake_run_ask)


@pytest.mark.django_db
def test_ask_v1_returns_canonical_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_ask(monkeypatch, answer="CSE101 is 6 ECTS.")
    client = Client()

    r = client.post(
        "/api/v1/ask",
        data=json.dumps({"question": "CSE101 kaç AKTS?"}),
        content_type="application/json",
    )

    assert r.status_code == 200, r.content
    body = json.loads(r.content.decode("utf-8"))

    # Top-level keys
    assert set(body.keys()) == {
        "conversation_id",
        "answer",
        "answer_source",
        "retrieved_chunks",
        "filters",
        "latency_ms",
    }
    assert body["answer"] == "CSE101 is 6 ECTS."
    assert body["answer_source"] == "RAG_LLM"
    assert isinstance(body["conversation_id"], int)

    # retrieved_chunks: text MUST be absent in default (debug=false) mode.
    assert len(body["retrieved_chunks"]) == 1
    chunk = body["retrieved_chunks"][0]
    assert "text" not in chunk
    for required in ("chunk_id", "score", "title", "url", "metadata"):
        assert required in chunk
    # Promoted metadata fields surface as first-class keys.
    assert chunk["course_code"] == "CSE101"
    assert chunk["department"] == "Bilgisayar Mühendisliği"

    # filters: every key present, even when null.
    expected_filter_keys = {
        "faculty",
        "department",
        "course_code",
        "semester",
        "content_types",
        "matched_terms",
    }
    assert set(body["filters"].keys()) == expected_filter_keys
    assert body["filters"]["course_code"] == "CSE101"

    # Timings
    assert body["latency_ms"] == {"retrieve": 12, "llm": 540, "total": 600}


@pytest.mark.django_db
def test_ask_v1_debug_includes_chunk_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_ask(monkeypatch)
    client = Client()

    r = client.post(
        "/api/v1/ask",
        data=json.dumps({"question": "CSE101?", "debug": True}),
        content_type="application/json",
    )
    assert r.status_code == 200
    body = json.loads(r.content.decode("utf-8"))
    chunks = body["retrieved_chunks"]
    assert len(chunks) == 1
    assert "text" in chunks[0]
    assert "CSE101 is a 6 ECTS course" in chunks[0]["text"]


@pytest.mark.django_db
def test_ask_v1_empty_question_returns_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_ask(monkeypatch)
    client = Client()

    r = client.post(
        "/api/v1/ask",
        data=json.dumps({"question": "   "}),
        content_type="application/json",
    )
    assert r.status_code == 400
    body = json.loads(r.content.decode("utf-8"))
    assert body == {"error": {"code": "empty_question", "message": "Question cannot be empty."}}


@pytest.mark.django_db
def test_ask_v1_invalid_json_returns_error_envelope() -> None:
    client = Client()

    r = client.post(
        "/api/v1/ask",
        data="not json",
        content_type="application/json",
    )
    assert r.status_code == 400
    body = json.loads(r.content.decode("utf-8"))
    assert body["error"]["code"] == "invalid_json"


@pytest.mark.django_db
def test_ask_v1_unknown_conversation_returns_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_ask(monkeypatch)
    client = Client()

    # Pick an id that is guaranteed not to exist for this test session.
    r = client.post(
        "/api/v1/ask",
        data=json.dumps({"question": "hi", "conversation_id": 99_999_999}),
        content_type="application/json",
    )
    assert r.status_code == 404
    body = json.loads(r.content.decode("utf-8"))
    assert body["error"]["code"] == "conversation_not_found"


@pytest.mark.django_db
def test_ask_v1_is_csrf_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proxy strategy assumes /api/v1/ask never sees a browser CSRF token."""
    _stub_run_ask(monkeypatch)
    client = Client(enforce_csrf_checks=True)

    r = client.post(
        "/api/v1/ask",
        data=json.dumps({"question": "CSE101?"}),
        content_type="application/json",
    )
    # Legacy /ask (with enforce_csrf_checks=True and no token) returns 403; the v1
    # endpoint must accept the same call because it is meant to be called server-side.
    assert r.status_code == 200, r.content


@pytest.mark.django_db
def test_ask_v1_health_endpoint() -> None:
    client = Client()
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert json.loads(r.content.decode("utf-8")) == {"status": "ok"}
