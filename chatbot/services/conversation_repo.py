"""Conversation/message persistence helpers.

The functions here are HTTP-agnostic by design: they return plain dataclasses,
tuples, or model instances. The HTTP adapter (``chatbot.api.v1.views`` after F7)
is responsible for translating outcomes into ``JsonResponse``.

For the duration of the strangler-fig refactor, ``chatbot/views.py`` keeps thin
wrappers that adapt the pure helpers below to the legacy ``JsonResponse`` calling
convention so existing call sites in the ``ask`` view do not change in F5.
"""
from __future__ import annotations

import hashlib

from django.utils import timezone


def conversation_title_from_question(question: str, max_len: int = 80) -> str:
    t = " ".join((question or "").split())
    if not t:
        return "Yeni sohbet"
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def touch_conversation_updated_at(conv) -> None:
    from chatbot.models import Conversation

    Conversation.objects.filter(pk=conv.pk).update(updated_at=timezone.now())


def resolve_conversation(
    body: dict,
    *,
    session_key: str = "",
) -> tuple[object | None, tuple[dict, int] | None]:
    """Return ``(conversation, error)`` where exactly one is ``None``.

    Errors are returned as ``({"detail": str}, status_code)`` so the HTTP adapter
    can wrap them in a ``JsonResponse`` without leaking framework types here.

    ``session_key`` ties new rows to a single browser session and gates lookups
    of existing rows. Mismatches return 404 (rather than 403) so a malicious
    caller cannot probe for the existence of conversations they do not own.
    """
    from chatbot.models import Conversation

    raw = body.get("conversation_id")
    if raw in (None, "", False):
        return Conversation.objects.create(title="", session_key=session_key), None
    try:
        cid = int(raw)
    except (TypeError, ValueError):
        return None, ({"detail": "Invalid conversation_id."}, 400)
    qs = Conversation.objects.filter(pk=cid)
    if session_key:
        qs = qs.filter(session_key=session_key)
    conv = qs.first()
    if not conv:
        return None, ({"detail": "Conversation not found."}, 404)
    return conv, None


def append_followup_invite(text: str, *, is_tr: bool, conv_id: int, question: str) -> str:
    """Short closing line suggesting what the user might ask next (rotates per thread/question)."""
    t = (text or "").strip()
    if not t:
        return t
    salt = f"{conv_id}|{question[:96]}|{len(t)}"
    h = int(hashlib.sha256(salt.encode("utf-8", errors="replace")).hexdigest(), 16)
    if is_tr:
        variants = [
            "\n\n— Başvuru, burs veya kampüs için de sorabilirsin.",
            "\n\n— Program, staj veya iletişim hakkında devam edebilirsin.",
            "\n\n— Akademik takvim veya ulaşım için de sorabilirsin.",
            "\n\n— Kayıt veya yurt konularında da sorabilirsin.",
        ]
    else:
        variants = [
            "\n\n— Ask about admissions, scholarships, or campus if you like.",
            "\n\n— Programs, internships, or contact — happy to help.",
            "\n\n— Academic calendar or transport — just ask.",
        ]
    return t + variants[h % len(variants)]


def build_assistant_reply(
    conv,
    text: str,
    *,
    status: int = 200,
    as_detail: bool = False,
    attach_followup: bool = False,
    is_tr: bool = True,
    question: str = "",
) -> tuple[dict, int]:
    """Persist the assistant message and return the JSON-serializable payload + status."""
    from chatbot.models import Message

    if attach_followup and status == 200 and not as_detail:
        text = append_followup_invite(text, is_tr=is_tr, conv_id=int(conv.pk), question=question or "")
    Message.objects.create(conversation=conv, role=Message.ROLE_ASSISTANT, content=text)
    touch_conversation_updated_at(conv)
    payload: dict = {"conversation_id": conv.pk}
    if as_detail:
        payload["detail"] = text
    else:
        payload["answer"] = text
    return payload, status


__all__ = [
    "conversation_title_from_question",
    "touch_conversation_updated_at",
    "resolve_conversation",
    "append_followup_invite",
    "build_assistant_reply",
]
