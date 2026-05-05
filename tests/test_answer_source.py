"""Adım 5.4 — answer_source classification.

Pin the contract: ``run_ask`` must surface ``LLM_TIMEOUT`` (separate from
``FALLBACK``) when Ollama times out, and must downgrade ``RAG_LLM`` to
``NO_INFO`` when the LLM emits the stock "I don't know" reply despite
having context. Both signals drive specific UX in the Next.js client.
"""
from __future__ import annotations

import pytest

import chatbot.services.ask_orchestrator as ask_orchestrator
from chatbot.services.ask_orchestrator import (
    ANSWER_SOURCE_EXTRACTIVE,
    ANSWER_SOURCE_LLM_TIMEOUT,
    ANSWER_SOURCE_NO_INFO,
    ANSWER_SOURCE_RAG_LLM,
)
from chatbot.services.llm_client import OLLAMA_TIMEOUT_SENTINEL


def _stub_retrieval(monkeypatch: pytest.MonkeyPatch, *, with_context: bool = True) -> None:
    chunks = (
        [
            {
                "chunk_id": 1,
                "url": "https://obs.example/p",
                "title": "stub program page",
                # A long enough body so ``len(context.strip()) > 180`` and the
                # orchestrator stays on the LLM branch instead of an early
                # FALLBACK due to empty context.
                "text": (
                    "Bu chunk, Bilgisayar Mühendisliği programının genel "
                    "özeti hakkında uzunca bir açıklamadır. Ders, kredi ve "
                    "müfredat detayları dahil pek çok bilgiyi içerir, böylece "
                    "_select_context_for_llm aşaması bu bloğu eler değil tutar."
                ),
                "score": 0.85,
                "metadata": {"content_type": "bologna_program"},
            }
        ]
        if with_context
        else []
    )
    monkeypatch.setattr(
        ask_orchestrator,
        "_retrieve_top_chunks_by_embedding",
        lambda question, k, *, source_type=None, filters=None: list(chunks),
    )
    monkeypatch.setattr(ask_orchestrator, "_is_extractive_question", lambda question: False)


@pytest.mark.django_db
def test_campus_postal_and_transport_skips_llm_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Postal campus questions must not depend on irrelevant OBS chunks or Gemma."""
    from chatbot.models import Conversation

    def _should_not_retrieve(*_a, **_kw):
        raise AssertionError("retrieval must be skipped for canonical campus address")

    monkeypatch.setattr(ask_orchestrator, "_retrieve_top_chunks_by_embedding", _should_not_retrieve)
    monkeypatch.setattr(
        ask_orchestrator,
        "ask_gemma",
        lambda prompt: (_ for _ in ()).throw(AssertionError("LLM must not run")),
    )

    conv = Conversation.objects.create(title="", session_key="t-canonical-address")
    q = "Acıbadem Üniversitesi kampüs adresi ve ulaşım bilgisi nedir?"
    payload, status, meta = ask_orchestrator.run_ask(q, conv)

    assert status == 200, payload
    assert meta.answer_source == ANSWER_SOURCE_EXTRACTIVE
    assert "Kerem Aydınlar" in (payload.get("answer") or "")
    assert "Kayışdağı" in (payload.get("answer") or "")
    assert "ulaşım" in (payload.get("answer") or "").lower()


@pytest.mark.django_db
def test_llm_timeout_emits_dedicated_answer_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama timeout must surface as LLM_TIMEOUT, not the legacy FALLBACK."""
    from chatbot.models import Conversation

    _stub_retrieval(monkeypatch)
    monkeypatch.setattr(ask_orchestrator, "ask_gemma", lambda prompt: OLLAMA_TIMEOUT_SENTINEL)

    conv = Conversation.objects.create(title="", session_key="t-5.4-timeout")
    payload, status, meta = ask_orchestrator.run_ask("Bilgisayar mühendisliği nedir?", conv)

    assert status == 200, payload
    assert meta.answer_source == ANSWER_SOURCE_LLM_TIMEOUT


@pytest.mark.django_db
def test_stock_no_info_reply_downgrades_to_no_info_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the LLM says 'Bu konuda elimde net bir bilgi bulunamadı' we tag NO_INFO."""
    from chatbot.models import Conversation

    _stub_retrieval(monkeypatch)
    monkeypatch.setattr(
        ask_orchestrator,
        "ask_gemma",
        lambda prompt: "Bu konuda elimde net bir bilgi bulunamadı.",
    )

    conv = Conversation.objects.create(title="", session_key="t-5.4-stock")
    # Question keywords overlap the stub context (bilgisayar / müfredat) so we
    # pass the orchestrator's _context_likely_relevant gate and reach the LLM
    # branch — that is exactly where the stock-no-info → NO_INFO override fires.
    payload, status, meta = ask_orchestrator.run_ask(
        "Bilgisayar mühendisliği müfredatı nedir?", conv
    )

    assert status == 200, payload
    assert meta.answer_source == ANSWER_SOURCE_NO_INFO


@pytest.mark.django_db
def test_normal_answer_keeps_rag_llm_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: a real-looking answer stays RAG_LLM (regression guard for the override)."""
    from chatbot.models import Conversation

    _stub_retrieval(monkeypatch)
    monkeypatch.setattr(
        ask_orchestrator,
        "ask_gemma",
        lambda prompt: "Acıbadem Üniversitesi Türkiye'de yer alır ve birden çok fakültesi vardır.",
    )

    conv = Conversation.objects.create(title="", session_key="t-5.4-rag")
    payload, status, meta = ask_orchestrator.run_ask("Acıbadem nerede?", conv)

    assert status == 200, payload
    assert meta.answer_source == ANSWER_SOURCE_RAG_LLM
