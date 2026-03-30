import argparse

from model.local_llm import LocalLLM, _ollama_configured
from rag.pipeline import RAGConfig, RAGSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local RAG demo for Acibadem University")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local GGUF model file (not needed if Ollama env vars are set).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Ask a single question and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not _ollama_configured() and not args.model_path:
        raise SystemExit(
            "Provide --model-path to a GGUF file, or set OLLAMA_BASE_URL and OLLAMA_MODEL."
        )

    llm = LocalLLM(model_path=args.model_path)
    config = RAGConfig()
    rag = RAGSystem(llm=llm, config=config)
    rag.build_knowledge_base()

    if args.question:
        answer = rag.answer(args.question)
        print(f"\nQ: {args.question}\nA: {answer}\n")
        return

    print("Local RAG demo is ready. Type 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = rag.answer(question)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
