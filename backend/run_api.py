import argparse

import uvicorn

from backend.api import init_rag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Acibadem RAG backend API")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local GGUF model file.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_rag(model_path=args.model_path)
    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
