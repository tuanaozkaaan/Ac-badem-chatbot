"""Wire-format serializers for v1 of the chatbot API.

Plain functions returning ``dict`` are intentional: no DRF dependency yet, and
the project's external wire format is a small, stable surface. A future move to
``rest_framework.serializers.Serializer`` only needs to touch this module.
"""
from __future__ import annotations


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


__all__ = [
    "serialize_conversation_summary",
    "serialize_message",
    "serialize_conversation_detail",
]
