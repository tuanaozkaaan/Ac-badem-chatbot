"""Smoke tests: application boots and critical routes respond."""
from __future__ import annotations

import json

import pytest


@pytest.mark.django_db
def test_health_json_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adım 5.5: /health now performs DB+Ollama probes; mock both as up."""
    from django.test import Client

    from chatbot.api.v1 import views as v1_views

    monkeypatch.setattr(v1_views, "_probe_database", lambda: ("up", None))
    monkeypatch.setattr(v1_views, "_probe_llm", lambda: ("up", None))

    client = Client()
    for path in ("/health", "/health/", "/api/health", "/api/health/"):
        r = client.get(path)
        assert r.status_code == 200, path
        body = json.loads(r.content.decode("utf-8"))
        assert body.get("status") == "ok", path
        assert body.get("db") == "up", path
        assert body.get("llm") == "up", path


@pytest.mark.django_db
def test_ask_get_returns_ui() -> None:
    from django.test import Client

    client = Client()
    r = client.get("/ask/")
    assert r.status_code == 200
    assert "text/html" in (r.get("Content-Type") or "")


@pytest.mark.django_db
def test_conversations_list_ok() -> None:
    from django.test import Client

    client = Client()
    r = client.get("/conversations/")
    assert r.status_code == 200
    body = json.loads(r.content.decode("utf-8"))
    assert "results" in body
    assert isinstance(body["results"], list)
