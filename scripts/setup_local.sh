#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo "Missing .env in project root. Create it (see .env.example) and re-run."
  exit 1
fi

python3 -m venv .venv || true
source .venv/bin/activate

pip install -r requirements.txt

docker compose up -d db

python manage.py migrate
python manage.py create_embeddings

echo "Done. You can now run: python manage.py runserver"
