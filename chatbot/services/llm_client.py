"""Ollama HTTP client and the translation helper that rides on top of it.

Both functions are deliberately small wrappers around a single ``requests.post`` call.
They have no Django imports so the orchestrator can compose them freely.
"""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

# Public sentinel returned by ask_gemma() when the upstream model exceeds OLLAMA_HTTP_TIMEOUT.
# Kept as the literal string so existing callers using `== "__OLLAMA_TIMEOUT__"` keep working.
OLLAMA_TIMEOUT_SENTINEL = "__OLLAMA_TIMEOUT__"


def ask_gemma(prompt: str) -> str:
    # CPU'da ilk üretim + uzun prompt 120s'yi aşabiliyor; varsayılanı yükselt (env ile düşürülebilir).
    ollama_http_timeout = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "90"))
    ollama_http_timeout = max(45, min(ollama_http_timeout, 900))
    try:
        base_url = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        model = (os.environ.get("OLLAMA_MODEL") or "gemma2:2b").strip()
        # Uzun liste/müfredat için yüksek tutulabilir; .env: OLLAMA_NUM_PREDICT (üst sınır 2048).
        raw_np = int(os.environ.get("OLLAMA_NUM_PREDICT", "140"))
        num_predict = max(64, min(raw_np, 2048))
        temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
        top_p = float(os.environ.get("OLLAMA_TOP_P", "0.8"))
        # Keep model loaded in VRAM/RAM between requests (e.g. "10m", "0" to unload). Empty = omit.
        keep_alive = (os.environ.get("OLLAMA_KEEP_ALIVE") or "").strip()
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive
        # (bağlantı, okuma): yavaş üretimde okuma süresi env ile kontrol edilir.
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=(min(30, ollama_http_timeout // 3), ollama_http_timeout),
        )
        if response.status_code == 404:
            body = (response.text or "").strip()[:400]
            return (
                "Gemma error: Ollama returned 404 (usually the model is not downloaded yet). "
                f"Run once: docker compose exec ollama ollama pull {model} "
                f"(or locally: ollama pull {model}). "
                f"Ollama said: {body or 'empty body'}"
            )
        response.raise_for_status()
        data = response.json()
        return (data["response"] or "").strip()
    except KeyError:
        return "Gemma error: Missing 'response' field in Ollama JSON."
    except requests.Timeout:
        return OLLAMA_TIMEOUT_SENTINEL
    except Exception as e:
        return f"Gemma error: {str(e)}"


def translate_answer(answer: str, target_lang: str) -> str:
    """
    Translate an already-produced answer to the target language ("tr" or "en").
    Must preserve meaning and avoid adding new facts.
    """
    if target_lang not in ("tr", "en"):
        return answer
    if not (answer or "").strip():
        return answer
    to_label = "Turkish" if target_lang == "tr" else "English"
    prompt = f"""Translate the text below to {to_label}.
Rules:
- Preserve meaning exactly.
- Do not add any new information.
- Output only the translation (no quotes, no extra text).

Text:
{answer}
"""
    translated = ask_gemma(prompt)
    return (translated or "").strip() or answer


__all__ = [
    "OLLAMA_TIMEOUT_SENTINEL",
    "ask_gemma",
    "translate_answer",
]
