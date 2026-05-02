"""Authorization hooks for v1 of the chatbot API.

Implements session-bound conversation ownership: a conversation belongs to the
browser session that created it, and other sessions must not be able to read,
list, or probe its existence.

Mismatches return 404 (not 403) on purpose: returning 403 would leak that the
``conversation_id`` is real but unauthorized, which is itself an information
disclosure (CWE-204). 404 keeps the existence of the row private.
"""
from __future__ import annotations

from django.http import HttpRequest, JsonResponse


def get_or_create_session_key(request: HttpRequest) -> str:
    """Return the session key for the current request, creating one if absent.

    Django only allocates a key when the session is touched; for first-time
    visitors we trigger the allocation explicitly so newly-created conversations
    can be tagged with a stable owner identifier.
    """
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key or ""


def filter_conversations_by_session(qs, request: HttpRequest):
    """Restrict a Conversation queryset to rows owned by this browser session."""
    key = get_or_create_session_key(request)
    if not key:
        # Defensive: should not happen because get_or_create_session_key always
        # produces a key, but never expose unowned legacy rows if it does.
        return qs.none()
    return qs.filter(session_key=key)


def assert_conversation_owned(conv, request: HttpRequest) -> JsonResponse | None:
    """Return a 404 response if ``conv`` is not owned by the current session.

    Returning ``None`` means "ok, proceed". A non-None return value should be
    bubbled up as the view's response without further processing.
    """
    key = get_or_create_session_key(request)
    if not key or (conv.session_key or "") != key:
        return JsonResponse({"detail": "Not found."}, status=404)
    return None


__all__ = [
    "get_or_create_session_key",
    "filter_conversations_by_session",
    "assert_conversation_owned",
]
