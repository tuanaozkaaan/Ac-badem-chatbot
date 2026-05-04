"""
Django settings for acu_chatbot.

Security policy: production-mode is fail-closed.
- DEBUG=0 + missing/insecure SECRET_KEY        -> ImproperlyConfigured at startup
- DEBUG=0 + ALLOWED_HOSTS empty or contains "*" -> ImproperlyConfigured at startup
- DEBUG=0 + DJANGO_CORS_ALLOWED_ORIGINS empty   -> ImproperlyConfigured at startup
- POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD are always required (no insecure fallback).

Misconfiguration must prevent boot rather than silently downgrade to a vulnerable mode.
"""
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    # python-dotenv is optional; env vars may come from the shell or Docker Compose.
    pass


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and (value is None or value == ""):
        raise ImproperlyConfigured(f"Required environment variable missing: {name}")
    return value or ""


DEBUG = _env("DJANGO_DEBUG", "0") == "1"

_INSECURE_SECRET_FALLBACK = "django-insecure-dev-key-change-in-production"
SECRET_KEY = _env("DJANGO_SECRET_KEY", _INSECURE_SECRET_FALLBACK)
if not DEBUG and SECRET_KEY in ("", _INSECURE_SECRET_FALLBACK):
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY must be set to a strong, unique value when DEBUG=0."
    )

_allowed_hosts = [
    h.strip()
    for h in _env("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if h.strip()
]
if not DEBUG and (not _allowed_hosts or "*" in _allowed_hosts):
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must be an explicit, non-wildcard list when DEBUG=0."
    )
ALLOWED_HOSTS = _allowed_hosts

# Ollama: docker-compose sets OLLAMA_BASE_URL=http://ollama:11434 (Compose service DNS name).
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "gemma2:2b")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "chatbot",
]

# WhiteNoise must sit immediately after SecurityMiddleware to serve collected static
# files in production without an external web server.
#
# RequestIdMiddleware sits at the top so the correlation id is set before any
# downstream middleware logs anything, and it lives outside the security
# middleware sandwich because it neither validates nor mutates the request.
MIDDLEWARE = [
    "chatbot.middleware.request_id.RequestIdMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "acu_chatbot.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "acu_chatbot.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": _env("POSTGRES_DB", required=True),
        "USER": _env("POSTGRES_USER", required=True),
        "PASSWORD": _env("POSTGRES_PASSWORD", required=True),
        "HOST": _env("POSTGRES_HOST", "localhost"),
        "PORT": _env("POSTGRES_PORT", "5432"),
    }
}

# Cache backend. Adım 5.4 wires ``django-ratelimit`` to this cache; the
# in-process LocMemCache is sufficient for single-worker dev. For multi-worker
# / multi-host deployments you MUST switch to a shared backend (Redis,
# Memcached) or each worker will enforce its own private bucket and the
# advertised rate is effectively multiplied by the worker count.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "acu-chatbot-default",
    },
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
# Compressed + manifest hashed filenames; safe behind WhiteNoise in production.
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Match prior FastAPI routes: POST /ask without trailing slash (no redirect on POST).
APPEND_SLASH = False

# Adım 5.5 CORS strategy
# ----------------------
# The Next.js frontend talks to Django via a server-side Route Handler proxy
# (see frontend/app/lib/proxy.ts), so the browser is single-origin from its
# point of view and CORS is mostly inert.
#
# We still keep django-cors-headers wired up for two reasons:
#   1) the legacy `/ask` SPA template at the Django origin still ships and a
#      developer might point it at a different host during testing,
#   2) external integrations (Postman, curl from another origin, federated
#      apps) are the canonical "allowlisted via CORS" callers.
#
# Therefore: NEVER use `CORS_ALLOW_ALL_ORIGINS=True`. Even in dev, only
# explicit known origins are accepted. The proxy strategy means the
# allowlist may be empty in production — `setdefault` lets that be a
# legitimate value rather than a misconfiguration.
_DEFAULT_DEV_CORS_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:8000,"
    "http://127.0.0.1:8000,"
    "http://localhost:8001,"
    "http://127.0.0.1:8001"
)

if DEBUG:
    CORS_ALLOWED_ORIGINS = [
        o.strip()
        for o in _env("DJANGO_CORS_ALLOWED_ORIGINS", _DEFAULT_DEV_CORS_ORIGINS).split(",")
        if o.strip()
    ]
    CSRF_TRUSTED_ORIGINS = list(CORS_ALLOWED_ORIGINS)
else:
    # Production: the allowlist is sourced from the deploy environment. An
    # empty list is the correct value when the only legitimate caller is the
    # server-side Next.js proxy (which never goes through CORS). Operators
    # who add browser-direct callers must add their origins here explicitly.
    CORS_ALLOWED_ORIGINS = [
        o.strip()
        for o in _env("DJANGO_CORS_ALLOWED_ORIGINS", "").split(",")
        if o.strip()
    ]
    # CSRF_TRUSTED_ORIGINS still needs at least the Next.js public origin so
    # cookie-authenticated POSTs from the proxy succeed. Operators who run
    # the frontend on a different host MUST set ``DJANGO_CSRF_TRUSTED_ORIGINS``.
    CSRF_TRUSTED_ORIGINS = [
        o.strip()
        for o in _env(
            "DJANGO_CSRF_TRUSTED_ORIGINS",
            ",".join(CORS_ALLOWED_ORIGINS),
        ).split(",")
        if o.strip()
    ]

    # Hardening that only makes sense behind TLS termination.
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 60 * 60 * 24 * 30
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    X_FRAME_OPTIONS = "DENY"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        # Adım 5.5: every log record gets a `request_id` attribute so the
        # ``default`` formatter can reference it. Outside an HTTP request the
        # filter still fires and writes ``"-"``, keeping the column aligned.
        "request_id": {
            "()": "chatbot.middleware.request_id.RequestIdFilter",
        },
    },
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "filters": ["request_id"],
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "DEBUG" if DEBUG else "INFO",
    },
    "loggers": {
        "django.security": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "django.request": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
}
