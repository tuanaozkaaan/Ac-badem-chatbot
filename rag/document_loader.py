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


def load_chunks_from_db() -> List[str]:
    """
    Load chunked knowledge directly from PostgreSQL (PageChunk table).
    Returns empty list if Django ORM is unavailable or no chunks exist.
    """
    try:
        from chatbot.models import PageChunk
    except Exception:
        return []

    rows = PageChunk.objects.order_by("id").values_list("chunk_text", flat=True)
    chunks = [text.strip() for text in rows if text and text.strip()]
    return chunks
