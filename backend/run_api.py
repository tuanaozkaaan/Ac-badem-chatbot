import argparse

import uvicorn

from backend.api import init_rag
from model.local_llm import _ollama_configured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Acibadem RAG backend API")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local GGUF model file (not needed if Ollama env vars are set).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _ollama_configured() and not args.model_path:
        raise SystemExit(
            "Provide --model-path to a GGUF file, or set OLLAMA_BASE_URL and OLLAMA_MODEL."
        )
    init_rag(model_path=args.model_path)
    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
