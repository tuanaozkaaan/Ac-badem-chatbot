"""Orchestrator-level guard for the Adım 5.3 retrieval-width fix.

The unit test in ``test_query_parser.py`` proves that the parser flags
"dersler" / "courses" with BOLOGNA_COURSE; this file proves that
``run_ask`` actually reacts to that flag by bumping ``k`` on the call to
``_retrieve_top_chunks_by_embedding``.

We monkey-patch the heavy collaborators (retriever + LLM + extractive
helper) so the test runs without Postgres data or Ollama and stays a
fast, deterministic regression check.
"""
from __future__ import annotations

from typing import Any

import pytest

import chatbot.services.ask_orchestrator as ask_orchestrator
from chatbot.services.query_parser import QueryFilters


@pytest.fixture
def captured_k_ctx(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub ``run_ask``'s collaborators and capture the ``k`` argument."""
    captured: dict[str, Any] = {"k": None, "filters": None}

    def fake_retrieve(question: str, k: int, *, source_type=None, filters=None):
        captured["k"] = int(k)
        captured["filters"] = filters
        # Return a single non-empty chunk so the orchestrator goes down the
        # "I have context" branch instead of the early FALLBACK.
        return [
            {
                "chunk_id": 1,
                "url": "https://obs.example/p",
                "title": "stub",
                "text": "stub course chunk content for the LLM",
                "score": 0.9,
                "metadata": {"content_type": "bologna_course"},
            }
        ]

    monkeypatch.setattr(
        ask_orchestrator, "_retrieve_top_chunks_by_embedding", fake_retrieve
    )
    monkeypatch.setattr(
        ask_orchestrator, "ask_gemma", lambda prompt: "stub answer from LLM"
    )
    # Skip the extractive fast path so we always exercise the LLM branch.
    monkeypatch.setattr(
        ask_orchestrator, "_is_extractive_question", lambda question: False
    )
    return captured


@pytest.mark.django_db
def test_course_catalog_question_bumps_k_ctx(
    captured_k_ctx: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Department + bare 'dersler' must lift k_ctx to at least 14 (Adım 5.3)."""
    from chatbot.models import Conversation

    conv = Conversation.objects.create(title="", session_key="t-5.3")

    payload, status, meta = ask_orchestrator.run_ask(
        "Bilgisayar mühendisliği dersleri nelerdir?", conv
    )

    assert status == 200, payload
    assert captured_k_ctx["k"] is not None
    assert captured_k_ctx["k"] >= 14, (
        f"k_ctx was not bumped for course-catalog query: got {captured_k_ctx['k']}"
    )
    parsed: QueryFilters = captured_k_ctx["filters"]
    assert parsed.department == "Bilgisayar Mühendisliği"
    assert "bologna_course" in parsed.content_types
    # Side-channel meta should also reflect the parser output.
    assert meta.filters.department == "Bilgisayar Mühendisliği"


@pytest.mark.django_db
def test_specific_course_code_does_not_bump_k_ctx(
    captured_k_ctx: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A precise CSE101-style query must NOT trigger the wide retrieval path."""
    from chatbot.models import Conversation

    conv = Conversation.objects.create(title="", session_key="t-5.3-precise")

    payload, status, _meta = ask_orchestrator.run_ask("CSE101 dersi kaç AKTS?", conv)

    assert status == 200, payload
    assert captured_k_ctx["k"] is not None
    # Default k_ctx is 5; with a course_code we expect the parser-narrow path.
    assert captured_k_ctx["k"] < 14, (
        f"k_ctx unexpectedly widened for course_code-specific query: got {captured_k_ctx['k']}"
    )


@pytest.mark.django_db
def test_unrelated_question_uses_default_k_ctx(
    captured_k_ctx: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No department + no curriculum cue → default narrow retrieval."""
    from chatbot.models import Conversation

    conv = Conversation.objects.create(title="", session_key="t-5.3-unrelated")

    payload, status, _meta = ask_orchestrator.run_ask("Saat kaç?", conv)

    assert status == 200, payload
    assert captured_k_ctx["k"] == 5, f"baseline k_ctx changed: got {captured_k_ctx['k']}"
