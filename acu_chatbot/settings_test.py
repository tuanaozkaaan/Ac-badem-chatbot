"""Django settings for automated tests only.

``ChunkEmbedding`` uses ``django.contrib.postgres.fields.ArrayField``, so the test
suite runs against PostgreSQL (not SQLite). Point ``DATABASES`` at a disposable
database (e.g. ``acibadem_test``) using the same host/user/password as your dev
instance, or start the service defined in ``.github/workflows/main.yml``.

This module must not be used in production.
"""
from __future__ import annotations

import os
from pathlib import Path

# Satisfy fail-closed base imports; real credentials should come from the
# environment (CI services or local `.env` via the parent settings module).
os.environ.setdefault("DJANGO_DEBUG", "1")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-not-for-production-please-change")
os.environ.setdefault("POSTGRES_DB", "acibadem_test")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:9")

from acu_chatbot.settings import *  # noqa: E402, F403

# WhiteNoise warns if STATIC_ROOT is missing; keep a disposable directory for tests.
_static_root = Path(__file__).resolve().parent.parent / ".pytest_staticfiles"
_static_root.mkdir(exist_ok=True)
STATIC_ROOT = _static_root

_test_db_name = os.environ.get("POSTGRES_TEST_DB") or os.environ.get("POSTGRES_DB", "acibadem_test")
_db = DATABASES["default"].copy()
_db["NAME"] = _test_db_name
DATABASES = {"default": _db}

STATICFILES_STORAGE = "django.contrib.staticfiles.storage.StaticFilesStorage"

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]
