"""URL routing for the API-only v1 surface.

These routes are mounted under ``/api/v1/`` (see ``acu_chatbot.urls``). Unlike
the legacy ``chatbot.urls`` module, nothing here returns the SPA template;
clients are expected to be JSON-speaking (the Next.js Route Handler proxy
or external integrations).

Pairing
-------
* ``/api/v1/ask``                — Adım 5.1 canonical chat endpoint.
* ``/api/v1/health``             — same shape as the legacy /health probe.
* ``/api/v1/conversations/``     — list/create.
* ``/api/v1/conversations/<int:pk>/`` — detail.

The conversation routes intentionally re-use the legacy view callables — they
already return JSON and their wire shape is fixed by ``serializers.py``. Only
``/api/v1/ask`` is rewritten to honor the new contract.
"""
from __future__ import annotations

from django.urls import path

from chatbot.api.v1 import views

urlpatterns = [
    path("ask", views.ask_v1),
    path("ask/", views.ask_v1),
    path("health", views.health),
    path("health/", views.health),
    path("conversations/", views.conversations_root),
    path("conversations/<int:pk>/", views.conversations_detail),
]
