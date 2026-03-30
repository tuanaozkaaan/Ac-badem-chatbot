#!/bin/sh
# Apply database schema, then run Django (assignment: PostgreSQL + runserver on all interfaces).
set -e
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-acu_chatbot.settings}"
python manage.py migrate --noinput
exec python manage.py runserver 0.0.0.0:8000
