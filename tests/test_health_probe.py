"""Adım 5.5 — health endpoint must probe DB and Ollama.

The legacy implementation always returned ``{"status": "ok"}`` whether or
not the dependencies it claimed to front were actually reachable. That is
worse than no health check at all because Compose / k8s liveness gates
trust the response. These tests pin the new behaviour:

* both probes up   → HTTP 200, ``status=ok``,        ``db=up``,   ``llm=up``
* one probe down   → HTTP 503, ``status=degraded``,  flag ``up``/``down`` per probe
* probe details propagate so on-call triage can see the underlying error
"""
from __future__ import annotations

import json

import pytest
from django.test import Client

from chatbot.api.v1 import views as v1_views


@pytest.mark.django_db
def test_health_returns_ok_when_both_probes_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_views, "_probe_database", lambda: ("up", None))
    monkeypatch.setattr(v1_views, "_probe_llm", lambda: ("up", None))

    client = Client()
    r = client.get("/api/v1/health")

    assert r.status_code == 200
    body = json.loads(r.content.decode("utf-8"))
    assert body == {"status": "ok", "db": "up", "llm": "up"}


@pytest.mark.django_db
def test_health_returns_503_when_db_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        v1_views, "_probe_database", lambda: ("down", "could not connect to server")
    )
    monkeypatch.setattr(v1_views, "_probe_llm", lambda: ("up", None))

    client = Client()
    r = client.get("/api/v1/health")

    assert r.status_code == 503
    body = json.loads(r.content.decode("utf-8"))
    assert body["status"] == "degraded"
    assert body["db"] == "down"
    assert body["llm"] == "up"
    assert "could not connect to server" in body.get("db_detail", "")


@pytest.mark.django_db
def test_health_returns_503_when_llm_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_views, "_probe_database", lambda: ("up", None))
    monkeypatch.setattr(
        v1_views, "_probe_llm", lambda: ("down", "ConnectionError: refused")
    )

    client = Client()
    r = client.get("/api/v1/health")

    assert r.status_code == 503
    body = json.loads(r.content.decode("utf-8"))
    assert body["status"] == "degraded"
    assert body["db"] == "up"
    assert body["llm"] == "down"
    assert "refused" in body.get("llm_detail", "")


@pytest.mark.django_db
def test_health_legacy_path_also_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The /health and /api/health legacy paths reach the same view (Adım 5.1 split)."""
    monkeypatch.setattr(v1_views, "_probe_database", lambda: ("up", None))
    monkeypatch.setattr(v1_views, "_probe_llm", lambda: ("up", None))

    client = Client()
    for path in ("/health", "/health/", "/api/health", "/api/health/"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert json.loads(r.content.decode("utf-8")).get("status") == "ok", path
