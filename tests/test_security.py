"""Security-focused regression tests (CSRF, IDOR, fail-closed settings)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest
from django.contrib.sessions.backends.db import SessionStore
from django.test import Client, RequestFactory

from chatbot.api.v1.permissions import assert_conversation_owned
from chatbot.models import Conversation


@pytest.mark.django_db
def test_post_ask_rejects_missing_csrf(api_client: Client) -> None:
    """Mutating POST /ask must require a valid CSRF token when checks are enforced."""
    api_client.get("/ask/")
    r = api_client.post(
        "/ask/",
        data=json.dumps({"question": "hello", "conversation_id": None}),
        content_type="application/json",
    )
    assert r.status_code == 403


@pytest.mark.django_db
def test_conversation_detail_idor_returns_404() -> None:
    """Another browser session must not read someone else's conversation (404, not 403)."""
    a = Client(enforce_csrf_checks=False)
    b = Client(enforce_csrf_checks=False)
    created = a.post("/conversations/", data=json.dumps({}), content_type="application/json")
    assert created.status_code == 201
    conv_id = json.loads(created.content.decode("utf-8"))["id"]

    ok = a.get(f"/conversations/{conv_id}/")
    assert ok.status_code == 200

    leaked = b.get(f"/conversations/{conv_id}/")
    assert leaked.status_code == 404
    payload = json.loads(leaked.content.decode("utf-8"))
    assert payload.get("detail") == "Not found."


@pytest.mark.django_db
def test_assert_conversation_owned_mismatch_is_404() -> None:
    """Unit-level guardrail: wrong session_key yields a 404-shaped JsonResponse."""
    conv = Conversation.objects.create(title="t", session_key="owner-session-a")

    factory = RequestFactory()
    request = factory.get("/conversations/1/")
    session = SessionStore()
    session.create()
    request.session = session

    resp = assert_conversation_owned(conv, request)
    assert resp is not None
    assert resp.status_code == 404
    body: dict[str, Any] = json.loads(resp.content.decode("utf-8"))
    assert body.get("detail") == "Not found."


def test_production_rejects_default_insecure_secret() -> None:
    """Fail-closed: DEBUG=0 must not boot with the django-insecure placeholder key."""
    env = os.environ.copy()
    env.update(
        {
            "DJANGO_SETTINGS_MODULE": "acu_chatbot.settings",
            "PYTHONPATH": os.pathsep.join([os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]),
            "DJANGO_DEBUG": "0",
            "DJANGO_SECRET_KEY": "django-insecure-dev-key-change-in-production",
            "DJANGO_ALLOWED_HOSTS": "example.com",
            "DJANGO_CORS_ALLOWED_ORIGINS": "https://example.com",
            "POSTGRES_DB": "x",
            "POSTGRES_USER": "x",
            "POSTGRES_PASSWORD": "x",
        }
    )

    code = r"""
from django.core.exceptions import ImproperlyConfigured

import django

try:
    django.setup()
except ImproperlyConfigured:
    raise SystemExit(0)
raise SystemExit(1)
"""
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
