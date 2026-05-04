"""Version 1 of the chatbot HTTP API.

This module is the only entry point Django URL routing should bind to.
Domain logic lives under ``chatbot.services``; views here parse requests,
dispatch to the orchestrator, and serialize responses.

Endpoints
---------
* :func:`ask` — legacy SPA + JSON view bound to ``/`` and ``/ask``. Renders
  the bundled HTML on GET, dispatches POST through the orchestrator. CSRF is
  enforced (cookie + ``X-CSRFToken`` header) because the SPA shares the same
  origin.
* :func:`ask_v1` — API-only view bound to ``/api/v1/ask`` (Adım 5.1). Always
  JSON, never renders a template. CSRF-exempt by design: the intended caller
  is a server-side proxy (Next.js Route Handler) which does not propagate
  browser cookies. Cross-origin browsers are blocked by ``CORS_ALLOWED_ORIGINS``
  when ``DEBUG=0``.
"""
from __future__ import annotations

import json
import os
import unicodedata

from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from django_ratelimit.decorators import ratelimit

from chatbot.api.v1.permissions import (
    assert_conversation_owned,
    filter_conversations_by_session,
    get_or_create_session_key,
)
from chatbot.api.v1.serializers import (
    serialize_ask_response,
    serialize_conversation_detail,
    serialize_conversation_summary,
    serialize_error,
)
from chatbot.services.ask_orchestrator import run_ask
from chatbot.services.conversation_repo import resolve_conversation
from chatbot.services.embedding import (
    _STRICT_RAG_NOT_FOUND,
    _retrieve_top_chunks_by_embedding,
)

# Per-IP request budget for /api/v1/ask. The LLM round-trip is slow (Gemma 7B
# ~5-30s), so a healthy human user produces at most ~5 questions/min; ``30/m``
# leaves headroom for legit users on shared NAT while still capping a runaway
# script. Override at deploy time with ``ACU_ASK_RATE`` (django-ratelimit
# rate string format).
_ASK_RATE_LIMIT = os.environ.get("ACU_ASK_RATE", "30/m")


def _probe_database() -> tuple[str, str | None]:
    """Cheap liveness ping against the configured Postgres connection.

    Returns ``("up", None)`` on success and ``("down", message)`` otherwise.
    The probe deliberately runs on the ``default`` connection (the one the
    chatbot reads/writes) — verifying a different alias would not catch the
    real failure modes (psycopg2 cannot resolve host, credentials wrong,
    pgbouncer down, ...).
    """
    from django.db import DatabaseError, connection

    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()
        if row and row[0] == 1:
            return ("up", None)
        return ("down", "unexpected SELECT 1 result")
    except DatabaseError as exc:
        return ("down", str(exc))
    except Exception as exc:  # noqa: BLE001 — surface any backend failure
        return ("down", f"{type(exc).__name__}: {exc}")


def _probe_llm() -> tuple[str, str | None]:
    """HTTP HEAD-ish probe against ``OLLAMA_BASE_URL``.

    Uses ``/api/tags`` (Ollama's "list local models" endpoint, GET, cheap)
    rather than ``/api/generate`` so we never trigger a model load. A short
    timeout keeps the health endpoint snappy under back-pressure.
    """
    import requests

    base = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    timeout_raw = os.environ.get("ACU_HEALTH_LLM_TIMEOUT_S", "2.0")
    try:
        timeout = max(0.5, float(timeout_raw))
    except ValueError:
        timeout = 2.0
    try:
        response = requests.get(f"{base}/api/tags", timeout=timeout)
    except requests.RequestException as exc:
        return ("down", f"{type(exc).__name__}: {exc}")
    if 200 <= response.status_code < 500:
        # 200 is ideal; 4xx still proves the server is up and answering. We
        # only treat 5xx and connection errors as "down" so a stricter Ollama
        # build that 401s on /api/tags does not fail the deploy gate.
        return ("up", None)
    return ("down", f"HTTP {response.status_code}")


def health(_request):
    """Liveness + dependency probe (Adım 5.5).

    Aggregate status is ``ok`` only when both the DB and the LLM probe are up;
    any failure flips the top-level status to ``degraded`` and the response
    code to 503 so a Compose / k8s healthcheck can react immediately. Probe
    detail messages are kept on the wire to make on-call triage one step
    faster.
    """
    db_status, db_detail = _probe_database()
    llm_status, llm_detail = _probe_llm()

    everything_up = db_status == "up" and llm_status == "up"
    body: dict[str, object] = {
        "status": "ok" if everything_up else "degraded",
        "db": db_status,
        "llm": llm_status,
    }
    if db_detail:
        body["db_detail"] = db_detail
    if llm_detail:
        body["llm_detail"] = llm_detail
    return JsonResponse(body, status=200 if everything_up else 503)


def _strict_rag_verify_response(question: str, body: dict) -> JsonResponse:
    """
    Verification-only path: embedding retrieval from DB, no LLM, no Message/Conversation writes.
    """
    raw_k = body.get("strict_rag_top_k") or os.environ.get("ACU_STRICT_RAG_TOP_K") or 8
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        k = 8
    k = max(1, min(k, 50))

    chunks = _retrieve_top_chunks_by_embedding(question, k=k)
    if not chunks:
        return JsonResponse(
            {
                "strict_rag_verify": True,
                "answer": _STRICT_RAG_NOT_FOUND,
                "conversation_id": None,
                "retrieved_chunks": [],
            },
            status=200,
        )

    retrieved_lines: list[str] = ["[RETRIEVED CHUNKS]"]
    for i, c in enumerate(chunks, start=1):
        retrieved_lines.append(f"<chunk {i}>")
        retrieved_lines.append(f"score={c['score']:.4f}")
        retrieved_lines.append(f"url={c['url']}")
        retrieved_lines.append(f"title={c['title']}")
        retrieved_lines.append(c["text"])
        retrieved_lines.append("")

    max_chars = int(os.environ.get("ACU_STRICT_RAG_ANSWER_MAX_CHARS", "6000"))
    buf: list[str] = []
    rem = max_chars
    for c in chunks:
        if rem <= 0:
            break
        t = c["text"].strip()
        piece = t[:rem]
        buf.append(piece)
        rem -= len(piece)

    answer_only = "\n\n".join(buf)
    full_answer = "\n".join(retrieved_lines + ["[ANSWER BASED ON CONTEXT ONLY]", answer_only])

    slim = [
        {
            "chunk_id": c["chunk_id"],
            "score": c["score"],
            "url": c["url"],
            "title": c["title"],
            "text": c["text"],
        }
        for c in chunks
    ]

    return JsonResponse(
        {
            "strict_rag_verify": True,
            "answer": full_answer,
            "conversation_id": None,
            "retrieved_chunks": slim,
        },
        status=200,
    )


def _resolve_conversation(body: dict, request):
    """Thin HTTP wrapper around services.conversation_repo.resolve_conversation.

    Binds the resolved conversation to the current browser session so that
    subsequent reads (list, detail) are filtered by ownership.
    """
    session_key = get_or_create_session_key(request)
    conv, err = resolve_conversation(body, session_key=session_key)
    if err is not None:
        err_payload, err_status = err
        return None, JsonResponse(err_payload, status=err_status)
    return conv, None


@require_http_methods(["GET", "POST"])
@ensure_csrf_cookie
def ask(request):
    # Router: serves the SPA on GET, dispatches /ask POSTs to the orchestrator.
    # GET responses force the csrftoken cookie so subsequent POSTs can carry the
    # X-CSRFToken header validated by Django's CsrfViewMiddleware.
    if request.method == "GET":
        return render(
            request,
            "index.html",
            {"csrf_token_value": get_token(request)},
        )

    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON."}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"detail": "Question cannot be empty."}, status=400)
    question = unicodedata.normalize("NFC", question)

    strict_rag = bool(body.get("strict_rag_verify")) or (
        (os.environ.get("ACU_STRICT_RAG_VERIFY") or "").strip().lower() in ("1", "true", "yes")
    )
    if strict_rag:
        return _strict_rag_verify_response(question, body)

    conv, conv_err = _resolve_conversation(body, request)
    if conv_err is not None:
        return conv_err

    payload, status, _meta = run_ask(question, conv)
    return JsonResponse(payload, status=status)


# ---------------------------------------------------------------------------
# /api/v1/ask  (Adım 5.1)
# ---------------------------------------------------------------------------
def _truthy(value) -> bool:
    """Tolerant boolean for query params and JSON booleans (``true``/``1``/``yes``)."""
    if isinstance(value, bool):
        return value
    return (str(value or "")).strip().lower() in ("1", "true", "yes")


def _resolve_conversation_v1(body: dict, request) -> tuple[object | None, JsonResponse | None]:
    """Same as :func:`_resolve_conversation` but emits the canonical error envelope."""
    session_key = get_or_create_session_key(request)
    conv, err = resolve_conversation(body, session_key=session_key)
    if err is None:
        return conv, None
    err_payload, err_status = err
    code = "invalid_conversation_id" if err_status == 400 else "conversation_not_found"
    msg = err_payload.get("detail") or "Conversation lookup failed."
    return None, JsonResponse(serialize_error(code, msg), status=err_status)


@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key="ip", rate=_ASK_RATE_LIMIT, block=False)
def ask_v1(request):
    """Stable JSON contract for the chatbot. See ``docs/openapi.yaml`` for the schema.

    Request body::

        {
          "question": "string (required, non-empty)",
          "conversation_id": int | null,
          "debug": bool                 // include retrieved_chunks[].text
        }

    Success (HTTP 200) returns the shape produced by :func:`serialize_ask_response`.
    Errors return ``{"error": {"code", "message"}}`` with an appropriate 4xx/5xx.

    Rate limiting (Adım 5.4)
    ------------------------
    ``django-ratelimit`` decorates this view with ``key="ip", rate=ACU_ASK_RATE``
    and ``block=False``. We do not let the library raise; instead we inspect
    ``request.limited`` here so a 429 follows the canonical error envelope.
    """
    if getattr(request, "limited", False):
        return JsonResponse(
            serialize_error(
                "rate_limited",
                "Çok sık soru soruyorsunuz, lütfen biraz bekleyin.",
            ),
            status=429,
        )

    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse(
            serialize_error("invalid_json", "Request body is not valid JSON."),
            status=400,
        )

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse(
            serialize_error("empty_question", "Question cannot be empty."),
            status=400,
        )
    question = unicodedata.normalize("NFC", question)

    debug = _truthy(body.get("debug")) or _truthy(request.GET.get("debug"))

    if _truthy(body.get("strict_rag_verify")) or _truthy(os.environ.get("ACU_STRICT_RAG_VERIFY")):
        # Strict-RAG verify still uses the legacy compact payload (chunks include
        # text by design). v1 callers asking for ``debug`` get the same data via
        # the regular branch below; the verify mode stays as an operator tool.
        return _strict_rag_verify_response(question, body)

    conv, conv_err = _resolve_conversation_v1(body, request)
    if conv_err is not None:
        return conv_err

    payload, status, meta = run_ask(question, conv)

    # run_ask returns the legacy ``{detail: ...}`` shape for hard errors; translate
    # those into the v1 error envelope so the wire stays consistent.
    if status >= 400:
        code = "llm_error" if status == 502 else "internal_error"
        msg = payload.get("detail") or payload.get("answer") or "Internal error."
        return JsonResponse(
            serialize_error(code, msg, conversation_id=payload.get("conversation_id")),
            status=status,
        )

    response_payload = serialize_ask_response(
        conversation_id=payload.get("conversation_id"),
        answer=payload.get("answer", ""),
        retrieved_chunks=meta.retrieved_chunks,
        filters=meta.filters,
        latency_ms=meta.latency_ms,
        answer_source=meta.answer_source,
        include_chunk_text=debug,
    )
    return JsonResponse(response_payload, status=status)


@require_http_methods(["GET", "POST"])
def conversations_root(request):
    from chatbot.models import Conversation

    if request.method == "GET":
        qs = filter_conversations_by_session(Conversation.objects.all(), request)
        qs = qs.order_by("-updated_at")[:200]
        results = [serialize_conversation_summary(c) for c in qs]
        return JsonResponse({"results": results})
    session_key = get_or_create_session_key(request)
    conv = Conversation.objects.create(title="", session_key=session_key)
    return JsonResponse(serialize_conversation_summary(conv), status=201)


@require_http_methods(["GET"])
def conversations_detail(request, pk):
    from chatbot.models import Conversation

    conv = Conversation.objects.filter(pk=pk).first()
    if not conv:
        return JsonResponse({"detail": "Not found."}, status=404)
    forbidden = assert_conversation_owned(conv, request)
    if forbidden is not None:
        return forbidden
    return JsonResponse(serialize_conversation_detail(conv))


__all__ = [
    "_resolve_conversation",
    "_strict_rag_verify_response",
    "ask",
    "ask_v1",
    "conversations_detail",
    "conversations_root",
    "health",
]
