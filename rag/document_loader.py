from pathlib import Path
from typing import List


def load_text_documents(data_dir: str) -> List[str]:
    """Load all .txt documents from a local directory."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = []
    for file_path in sorted(path.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            documents.append(content)

    if not documents:
        raise ValueError(f"No .txt documents found in: {data_dir}")
    return documents
