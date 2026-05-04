"""Adım 5.5 — per-request UUID + structured logging.

The middleware generates a token that travels with the request through:
  * ``request.request_id`` (Python attribute, available to views),
  * the ``X-Request-Id`` response header (visible to the client / proxy),
  * the ``request_id`` logging field (rendered by the default formatter).

Inbound ``X-Request-Id`` headers are honored only when they look benign
(letters / digits / ``-``/``_``, ≤64 chars). Anything else is replaced
with a fresh UUID so a hostile client cannot poison log lines (CWE-117).
"""
from __future__ import annotations

import logging
import re

import pytest
from django.test import Client

from chatbot.middleware.request_id import (
    RequestIdFilter,
    _request_id_var,
    get_request_id,
)


_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


@pytest.mark.django_db
def test_response_carries_x_request_id_header() -> None:
    client = Client()
    r = client.get("/api/v1/health")
    rid = r.headers.get("X-Request-Id") or r.headers.get("x-request-id") or r.get("X-Request-Id")
    assert rid, "X-Request-Id header missing on response"
    assert _HEX_RE.match(rid), f"expected 32-char hex UUID, got {rid!r}"


@pytest.mark.django_db
def test_inbound_request_id_is_propagated_when_safe() -> None:
    client = Client()
    safe_id = "trace-abc_123"
    r = client.get("/api/v1/health", HTTP_X_REQUEST_ID=safe_id)
    assert (r.headers.get("X-Request-Id") or r.get("X-Request-Id")) == safe_id


@pytest.mark.django_db
@pytest.mark.parametrize(
    "hostile",
    [
        "this is too long " * 10,                # > 64 chars
        "trace\nINJECTED LOG LINE",              # control character
        "trace; rm -rf /",                       # punctuation outside allow set
        "id with spaces",                        # space disallowed
        "a" * 100,                               # length cap
    ],
)
def test_hostile_request_id_is_replaced_with_fresh_uuid(hostile: str) -> None:
    client = Client()
    r = client.get("/api/v1/health", HTTP_X_REQUEST_ID=hostile)
    rid = r.headers.get("X-Request-Id") or r.get("X-Request-Id")
    assert rid != hostile, f"hostile id leaked through: {hostile!r}"
    assert _HEX_RE.match(rid or ""), f"expected fresh hex UUID, got {rid!r}"


def test_request_id_filter_attaches_default_when_outside_request() -> None:
    """Logging from a non-request context (Celery, management cmd, ...) gets ``-``."""
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="x", args=None, exc_info=None,
    )
    flt = RequestIdFilter()
    assert flt.filter(record) is True
    assert record.request_id == "-"


def test_request_id_filter_uses_contextvar_when_set() -> None:
    """When the middleware bound a value, the filter copies it onto the record."""
    token = _request_id_var.set("manual-id")
    try:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="x", args=None, exc_info=None,
        )
        RequestIdFilter().filter(record)
        assert record.request_id == "manual-id"
        assert get_request_id() == "manual-id"
    finally:
        _request_id_var.reset(token)
    # Reset must restore the default sentinel for the next request.
    assert get_request_id() == "-"
