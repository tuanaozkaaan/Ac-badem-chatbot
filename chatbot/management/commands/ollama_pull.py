"""Idempotently ensure the configured Ollama model is downloaded.

Invoked from ``docker/entrypoint.sh`` after migrations and ``collectstatic`` so
that ``docker compose up`` can produce a working /ask endpoint without manual
intervention. Safe to run on every boot: a quick HEAD-style probe short-circuits
when the model is already present.
"""
from __future__ import annotations

import json
import os

import requests
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Ensure the configured Ollama model is downloaded (idempotent)."

    def handle(self, *args, **options) -> None:
        base_url = (os.environ.get("OLLAMA_BASE_URL") or "http://ollama:11434").rstrip("/")
        model = (os.environ.get("OLLAMA_MODEL") or "gemma:7b").strip()

        if self._model_exists(base_url, model):
            self.stdout.write(self.style.SUCCESS(f"ollama_pull: {model} already present"))
            return

        self.stdout.write(f"ollama_pull: downloading {model} (this can take several minutes)…")
        self._pull(base_url, model)
        self.stdout.write(self.style.SUCCESS(f"ollama_pull: {model} ready"))

    # ------------------------------------------------------------------ helpers

    def _model_exists(self, base_url: str, model: str) -> bool:
        """Return True iff Ollama already has the model locally.

        Network/parse errors are treated as 'unknown' and we fall through to a
        pull attempt; the pull itself is idempotent on the Ollama side.
        """
        try:
            resp = requests.post(
                f"{base_url}/api/show",
                json={"name": model},
                timeout=(5, 30),
            )
            return resp.status_code == 200
        except requests.RequestException as e:
            self.stdout.write(
                self.style.WARNING(f"ollama_pull: probe failed ({e}); proceeding to pull")
            )
            return False

    def _pull(self, base_url: str, model: str) -> None:
        """Stream the pull, log status transitions, raise on errors."""
        try:
            with requests.post(
                f"{base_url}/api/pull",
                json={"name": model, "stream": True},
                stream=True,
                # Read timeout is generous: large model layers on slow networks may take a long time.
                timeout=(10, 1800),
            ) as resp:
                resp.raise_for_status()
                last_status = ""
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    try:
                        line = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    err = line.get("error")
                    if err:
                        raise RuntimeError(err)
                    status = line.get("status") or ""
                    # Only log transitions to keep output compact across the very chatty stream.
                    if status and status != last_status:
                        self.stdout.write(f"  · {status}")
                        last_status = status
        except (requests.RequestException, RuntimeError) as e:
            self.stderr.write(self.style.ERROR(f"ollama_pull: failed to pull {model}: {e}"))
            raise SystemExit(1)
