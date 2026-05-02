"""Pytest fixtures shared across tests."""
from __future__ import annotations

import pytest
from django.test import Client


@pytest.fixture
def api_client() -> Client:
    """Django test client with CSRF enforcement (matches production middleware)."""
    return Client(enforce_csrf_checks=True)


@pytest.fixture
def api_client_no_csrf() -> Client:
    """Client without CSRF enforcement — use only when simulating token-aware clients."""
    return Client(enforce_csrf_checks=False)
