from pathlib import Path

from llama_cpp import Llama


class LocalLLM:
    """
    Thin wrapper around llama.cpp for local open-source inference.
    Expects a local GGUF model file.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Local model file not found: {model_path}\n"
                "Download a GGUF model and set --model-path to that file."
            )

        self.llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=4,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "User:", "Question:"],
        )
        return output["choices"][0]["text"].strip()
