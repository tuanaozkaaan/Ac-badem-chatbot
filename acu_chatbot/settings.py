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
MIDDLEWARE = [
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

if DEBUG:
    # Local dev: allow same-origin pages on common dev ports without explicit configuration.
    CORS_ALLOW_ALL_ORIGINS = True
    CSRF_TRUSTED_ORIGINS = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]
else:
    CORS_ALLOWED_ORIGINS = [
        o.strip()
        for o in _env("DJANGO_CORS_ALLOWED_ORIGINS", "").split(",")
        if o.strip()
    ]
    if not CORS_ALLOWED_ORIGINS:
        raise ImproperlyConfigured(
            "DJANGO_CORS_ALLOWED_ORIGINS must list explicit origins when DEBUG=0."
        )
    # CSRF_TRUSTED_ORIGINS mirrors CORS allow-list so cookie-authenticated POSTs work behind a proxy.
    CSRF_TRUSTED_ORIGINS = list(CORS_ALLOWED_ORIGINS)

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
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
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
