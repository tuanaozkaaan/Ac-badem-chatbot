import logging
import os
from pathlib import Path

DEFAULT_OLLAMA_MODEL = "gemma2:2b"

logger = logging.getLogger(__name__)


def _ollama_configured() -> bool:
    base = os.environ.get("OLLAMA_BASE_URL", "").strip()
    return bool(base)


def _get_ollama_model() -> str:
    model = os.environ.get("OLLAMA_MODEL", "").strip()
    return model or DEFAULT_OLLAMA_MODEL


def _get_ollama_timeout_seconds() -> int:
    raw_value = os.environ.get("OLLAMA_HTTP_TIMEOUT", "360").strip()
    try:
        timeout = int(raw_value)
    except ValueError:
        timeout = 360
    return max(90, min(timeout, 900))


class LocalLLM:
    """
    Thin wrapper around llama.cpp OR Ollama HTTP API for local inference.

    Important: llama-cpp-python is lazily imported only when Ollama is not used.
    This allows Docker builds to succeed even if llama-cpp-python is not installed.
    """

    def __init__(self, model_path: str | None = None, n_ctx: int = 2048):
        self._use_ollama = _ollama_configured()
        if self._use_ollama:
            self._ollama_base = os.environ["OLLAMA_BASE_URL"].strip().rstrip("/")
            self._ollama_model = _get_ollama_model()
            self._ollama_timeout_seconds = _get_ollama_timeout_seconds()
            self.llm = None
            logger.info(
                "LocalLLM: Ollama mode — base_url=%r, model=%r, timeout=%ss",
                self._ollama_base,
                self._ollama_model,
                self._ollama_timeout_seconds,
            )
            # Ensures docker logs show the resolved URL even if logging is not configured.
            _gen = f"{self._ollama_base}/api/generate"
            print(
                f"DEBUG: LocalLLM: Ollama OLLAMA_BASE_URL={self._ollama_base!r}, "
                f"OLLAMA_MODEL={self._ollama_model!r}, timeout={self._ollama_timeout_seconds}s, "
                f"generate endpoint={_gen!r}"
            )
            return

        if not model_path:
            raise ValueError(
                "model_path is required unless OLLAMA_BASE_URL is set."
            )

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Local model file not found: {model_path}\n"
                "Download a GGUF model and set --model-path to that file."
            )

        # Lazy import to avoid hard dependency at Docker build time.
        # If GGUF mode is used without llama-cpp-python installed, provide a clear error.
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it to use GGUF local inference, "
                "or configure OLLAMA_BASE_URL to use Ollama."
            ) from exc

        self.llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=4,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        if self._use_ollama:
            import httpx

            url = f"{self._ollama_base}/api/generate"
            payload = {
                "model": self._ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            try:
                timeout = httpx.Timeout(
                    connect=min(30.0, self._ollama_timeout_seconds / 3),
                    read=float(self._ollama_timeout_seconds),
                    write=min(30.0, self._ollama_timeout_seconds / 3),
                    pool=min(30.0, self._ollama_timeout_seconds / 3),
                )
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                full_body = exc.response.text or ""
                err_text = (
                    f"Ollama HTTP {exc.response.status_code} for POST {url!r} "
                    f"(OLLAMA_BASE_URL={self._ollama_base!r}, model={self._ollama_model!r}). "
                    f"Response body:\n{full_body}"
                )
                logger.error("%s", err_text, exc_info=True)
                print(f"ERROR: LocalLLM Ollama: {err_text}", flush=True)
                return ""
            except httpx.RequestError as exc:
                err_text = (
                    f"Ollama request failed for {url!r} (OLLAMA_BASE_URL={self._ollama_base!r}, "
                    f"model={self._ollama_model!r}): {type(exc).__name__}: {exc!r}"
                )
                logger.error("%s", err_text, exc_info=True)
                print(f"ERROR: LocalLLM Ollama: {err_text}", flush=True)
                return ""
            try:
                data = response.json()
            except ValueError as exc:
                raw = response.text
                err_text = (
                    f"Ollama response was not valid JSON for {url!r}. Parse error: {exc!r}. "
                    f"Raw body (first 8000 chars): {raw[:8000]!r}"
                )
                logger.error("%s", err_text, exc_info=True)
                print(f"ERROR: LocalLLM Ollama: {err_text}", flush=True)
                return ""
            text = (data.get("response") or "").strip()
            if not text:
                err_text = (
                    f"Ollama returned empty 'response' from {url!r} "
                    f"(OLLAMA_BASE_URL={self._ollama_base!r}, model={self._ollama_model!r}). "
                    f"Full JSON: {data!r}"
                )
                logger.error("%s", err_text)
                print(f"ERROR: LocalLLM Ollama: {err_text}", flush=True)
                return ""
            return text

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "User:", "Question:"],
        )
        return output["choices"][0]["text"].strip()
