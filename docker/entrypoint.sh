#!/bin/sh
# Container entrypoint for the Django web service.
# - Applies migrations and collects static assets on every boot (idempotent).
# - Runs Django's dev server when DJANGO_DEBUG=1, gunicorn otherwise.
# - Gunicorn timeout is sized for local LLM latency (Gemma 7B first-token cold start).
set -eu

echo "[entrypoint] applying database migrations"
python manage.py migrate --noinput

echo "[entrypoint] collecting static files"
python manage.py collectstatic --noinput

if [ "${DJANGO_DEBUG:-0}" = "1" ]; then
    echo "[entrypoint] starting Django dev server (DJANGO_DEBUG=1)"
    exec python manage.py runserver 0.0.0.0:8000
else
    echo "[entrypoint] starting gunicorn (workers=${GUNICORN_WORKERS:-2}, timeout=${GUNICORN_TIMEOUT:-300})"
    exec gunicorn acu_chatbot.wsgi:application \
        --bind 0.0.0.0:8000 \
        --workers "${GUNICORN_WORKERS:-2}" \
        --timeout "${GUNICORN_TIMEOUT:-300}" \
        --access-logfile - \
        --error-logfile -
fi
