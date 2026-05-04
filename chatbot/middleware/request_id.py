"""Per-request correlation IDs for structured logging (Adım 5.5).

Why a context variable instead of ``threading.local`` or a logging adapter
-------------------------------------------------------------------------
Django supports both WSGI (sync) and ASGI (async) handlers. ``contextvars``
is the only primitive that propagates correctly across both: it is bound
to the asyncio Task in async views and to the request-handling thread in
sync views without any extra glue. ``threading.local`` would silently
leak ids between coroutines on the same thread; a logging adapter would
require every call site to pass ``extra={"request_id": ...}`` by hand,
which we explicitly want to avoid.

Surface
-------
* :class:`RequestIdMiddleware` — generates one UUID per inbound request,
  exposes it on ``request.request_id``, sets the ``X-Request-Id`` response
  header, and clears the contextvar after the response is built.
* :class:`RequestIdFilter` — logging filter that injects the contextvar
  value (or ``"-"`` when no request is active) into every log record so
  the formatter can render ``[%(request_id)s]`` uniformly.
* :func:`get_request_id` — read-only helper for code that wants to log
  the id manually (e.g. inside Celery tasks triggered from a view).

Inbound ``X-Request-Id`` headers from a trusted proxy are honored, with
basic shape validation so a hostile client cannot inject control
characters or arbitrarily long strings into the log lines.
"""
from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar
from typing import Callable

from django.http import HttpRequest, HttpResponse

# Inline so callers do not need to import ContextVar themselves.
_request_id_var: ContextVar[str] = ContextVar("acu_request_id", default="-")

# Trust an upstream-supplied id only if it looks like a sane correlation token:
# letters, digits, dash, underscore, max 64 chars. Reject anything longer so a
# rogue proxy cannot enlarge the log payload, and anything with newlines /
# control characters that could fake new log lines (CWE-117).
_INBOUND_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def get_request_id() -> str:
    """Current request id, or ``"-"`` if called outside the request lifecycle."""
    return _request_id_var.get()


def _new_request_id(inbound: str | None) -> str:
    if inbound:
        candidate = inbound.strip()
        if _INBOUND_ID_RE.match(candidate):
            return candidate
    # Hex form (no dashes) keeps the log column tight without losing entropy.
    return uuid.uuid4().hex


class RequestIdMiddleware:
    """Attach a request id to every request/response pair."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        rid = _new_request_id(request.headers.get("X-Request-Id"))
        token = _request_id_var.set(rid)
        request.request_id = rid  # type: ignore[attr-defined]
        try:
            response = self.get_response(request)
        finally:
            _request_id_var.reset(token)
        response["X-Request-Id"] = rid
        return response


class RequestIdFilter(logging.Filter):
    """Logging filter that copies the contextvar onto every record.

    Wired in via Django's ``LOGGING["filters"]`` so the standard formatter
    can reference ``%(request_id)s`` without each log call passing it by
    hand. Outside an HTTP request the filter emits ``"-"`` so the column
    width stays stable.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


__all__ = [
    "RequestIdMiddleware",
    "RequestIdFilter",
    "get_request_id",
]
