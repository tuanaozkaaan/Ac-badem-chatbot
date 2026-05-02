"""Ollama HTTP client and the translation helper that rides on top of it.

Tuned for Gemma 7B as the deployment target:
- Generation defaults (num_predict=256, temperature=0.15, top_p=0.9) match the
  values supplied via docker-compose so behaviour stays consistent when the
  service runs outside of Compose.
- Default OLLAMA_HTTP_TIMEOUT raised to 240s to accommodate 7B latency.
- One transparent retry on transient network errors (ConnectionError or HTTP 503,
  which Ollama returns while a model is being loaded into RAM). Hard timeouts
  are NOT retried — the user already waited; a second blocking call is worse UX.
"""
from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# Public sentinel returned by ask_gemma() when the upstream model exceeds OLLAMA_HTTP_TIMEOUT.
# Kept as the literal string so existing callers using `== "__OLLAMA_TIMEOUT__"` keep working.
OLLAMA_TIMEOUT_SENTINEL = "__OLLAMA_TIMEOUT__"

_RETRY_DELAY_SECONDS = 2.0


def _post_generate(
    *,
    base_url: str,
    payload: dict,
    timeout: tuple[int, int],
) -> requests.Response:
    """Single POST against /api/generate; isolated for the retry helper."""
    return requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)


def ask_gemma(prompt: str) -> str:
    # CPU'da ilk üretim + uzun prompt 120s'yi aşabiliyor; varsayılanı yükselt (env ile düşürülebilir).
    ollama_http_timeout = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "240"))
    ollama_http_timeout = max(45, min(ollama_http_timeout, 900))
    try:
        base_url = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        model = (os.environ.get("OLLAMA_MODEL") or "gemma:7b").strip()
        # Uzun liste/müfredat için yüksek tutulabilir; .env: OLLAMA_NUM_PREDICT (üst sınır 2048).
        raw_np = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))
        num_predict = max(64, min(raw_np, 2048))
        temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.15"))
        top_p = float(os.environ.get("OLLAMA_TOP_P", "0.9"))
        # Keep model loaded in VRAM/RAM between requests (e.g. "30m", "0" to unload). Empty = omit.
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
        timeout = (min(30, ollama_http_timeout // 3), ollama_http_timeout)

        # One retry policy: transient network glitches + Ollama 503 (model warming).
        # requests.Timeout is intentionally NOT retried.
        attempt = 1
        started = time.perf_counter()
        while True:
            try:
                response = _post_generate(base_url=base_url, payload=payload, timeout=timeout)
            except requests.ConnectionError as conn_err:
                if attempt == 1:
                    logger.info("ask_gemma: connection error (%s); retrying once after %.1fs", conn_err, _RETRY_DELAY_SECONDS)
                    attempt += 1
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise
            if response.status_code == 503 and attempt == 1:
                logger.info("ask_gemma: 503 from Ollama (model warming); retrying once after %.1fs", _RETRY_DELAY_SECONDS)
                attempt += 1
                time.sleep(_RETRY_DELAY_SECONDS)
                continue
            break

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
        answer = (data["response"] or "").strip()
        logger.info(
            "ask_gemma: model=%s prompt_chars=%d response_chars=%d elapsed_ms=%d attempts=%d",
            model,
            len(prompt),
            len(answer),
            int((time.perf_counter() - started) * 1000),
            attempt,
        )
        return answer
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
