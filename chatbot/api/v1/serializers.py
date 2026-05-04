"""Wire-format serializers for v1 of the chatbot API.

Plain functions returning ``dict`` are intentional: no DRF dependency yet, and
the project's external wire format is a small, stable surface. A future move to
``rest_framework.serializers.Serializer`` only needs to touch this module.

The shapes produced here are the source of truth for ``docs/openapi.yaml``;
when you add a field here, update the spec in the same patch.
"""
from __future__ import annotations

from typing import Any

from chatbot.services.query_parser import QueryFilters

# ---------------------------------------------------------------------------
# Conversations / messages
# ---------------------------------------------------------------------------


def serialize_conversation_summary(conv) -> dict:
    """Compact representation used by the conversation list endpoint."""
    return {
        "id": conv.id,
        "title": conv.title or "",
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
    }


def serialize_message(msg) -> dict:
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "created_at": msg.created_at.isoformat(),
    }


def serialize_conversation_detail(conv) -> dict:
    """Full representation including all messages on the conversation."""
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "messages": [serialize_message(m) for m in conv.messages.all()],
    }


# ---------------------------------------------------------------------------
# /api/v1/ask response shape (Adım 5.1)
# ---------------------------------------------------------------------------
# Subset of PageChunk.metadata we promote to first-class fields on the wire.
# Everything else stays under ``metadata`` so a frontend can opt in without
# the backend having to enumerate keys ahead of time.
_PROMOTED_METADATA_KEYS: tuple[str, ...] = (
    "content_type",
    "department",
    "faculty",
    "course_code",
    "semester",
)


def serialize_retrieved_chunk(chunk: dict, *, include_text: bool = False) -> dict:
    """Serialize a single hit from ``_retrieve_top_chunks_by_embedding``.

    ``include_text`` is opt-in: the default response keeps ``text`` out of the
    payload to avoid leaking prompt-sized strings to every client. Pass
    ``debug=true`` to /api/v1/ask to flip it on.
    """
    metadata = dict(chunk.get("metadata") or {})

    out: dict[str, Any] = {
        "chunk_id": int(chunk["chunk_id"]),
        "score": float(chunk.get("score") or 0.0),
        "title": chunk.get("title") or "",
        "url": chunk.get("url") or "",
    }
    for key in _PROMOTED_METADATA_KEYS:
        if key in metadata:
            out[key] = metadata[key]
    out["metadata"] = metadata
    if include_text:
        out["text"] = chunk.get("text") or ""
    return out


def serialize_query_filters(filters: QueryFilters) -> dict:
    """Mirror :class:`QueryFilters` onto the wire as a plain dict.

    Always returns the full set of keys so clients can rely on field presence;
    unmatched fields are ``null`` / ``[]`` rather than missing.
    """
    return {
        "faculty": filters.faculty,
        "department": filters.department,
        "course_code": filters.course_code,
        "semester": filters.semester,
        "content_types": list(filters.content_types),
        "matched_terms": list(filters.matched_terms),
    }


def serialize_ask_response(
    *,
    conversation_id: int | None,
    answer: str,
    retrieved_chunks: list[dict],
    filters: QueryFilters,
    latency_ms: dict[str, int] | None = None,
    answer_source: str | None = None,
    include_chunk_text: bool = False,
) -> dict:
    """Build the canonical /api/v1/ask success payload.

    ``latency_ms`` keys (when present): ``retrieve``, ``llm``, ``total``.
    ``answer_source`` is one of: ``RAG_LLM``, ``EXTRACTIVE``, ``FALLBACK``.
    """
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "answer_source": answer_source,
        "retrieved_chunks": [
            serialize_retrieved_chunk(c, include_text=include_chunk_text)
            for c in (retrieved_chunks or [])
        ],
        "filters": serialize_query_filters(filters),
        "latency_ms": latency_ms or {},
    }


def serialize_error(code: str, message: str, *, conversation_id: int | None = None) -> dict:
    """Standard error envelope for /api/v1/* endpoints.

    Includes ``conversation_id`` when known so a client can correlate the failure
    with a half-written thread (the user message has already been persisted by
    the time most LLM-side failures surface).
    """
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    return payload


__all__ = [
    "serialize_conversation_summary",
    "serialize_message",
    "serialize_conversation_detail",
    "serialize_retrieved_chunk",
    "serialize_query_filters",
    "serialize_ask_response",
    "serialize_error",
]
