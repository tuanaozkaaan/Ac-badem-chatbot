import os
from pathlib import Path


def _ollama_configured() -> bool:
    base = os.environ.get("OLLAMA_BASE_URL", "").strip()
    model = os.environ.get("OLLAMA_MODEL", "").strip()
    return bool(base and model)


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
            self._ollama_model = os.environ["OLLAMA_MODEL"].strip()
            self.llm = None
            return

        if not model_path:
            raise ValueError(
                "model_path is required unless OLLAMA_BASE_URL and OLLAMA_MODEL are set."
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
                "or configure OLLAMA_BASE_URL and OLLAMA_MODEL to use Ollama."
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
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
            data = response.json()
            return (data.get("response") or "").strip()

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "User:", "Question:"],
        )
        return output["choices"][0]["text"].strip()
