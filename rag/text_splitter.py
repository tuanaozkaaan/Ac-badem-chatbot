from typing import List


def split_into_chunks(
    documents: List[str], chunk_size: int = 450, chunk_overlap: int = 80
) -> List[str]:
    """
    Split documents into overlapping character-based chunks.
    Keeps implementation simple and dependency-free for the demo.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    step = chunk_size - chunk_overlap

    for doc in documents:
        start = 0
        while start < len(doc):
            chunk = doc[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step

    if not chunks:
        raise ValueError("No chunks generated from documents.")
    return chunks
