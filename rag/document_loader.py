from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Single source of truth for the embedding model used across the project.
#
# ``paraphrase-multilingual-MiniLM-L12-v2`` is preferred over
# ``all-MiniLM-L6-v2`` because Acıbadem University content is primarily
# Turkish — the multilingual checkpoint understands Turkish morphology and
# inflection (öğrenci / öğrencinin / öğrenciye), Turkish-specific terms
# (Bologna, ders kataloğu, fakülte), and cross-lingual queries (TR question
# vs EN syllabus text in OBS) far better than the English-only L6 model.
# Both checkpoints emit 384-dimensional vectors so the DB schema does not
# need to change, but the *semantic* space is completely different — every
# stored vector MUST be regenerated when this constant moves. The
# ``rebuild_embeddings`` / ``create_embeddings --force`` commands enforce
# that contract.
EXPECTED_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class LoadedDbChunk:
    """One knowledge chunk from DB with optional precomputed embedding (ChunkEmbedding)."""

    chunk_text: str
    vector: np.ndarray | None  # 1D float32 when present; matches JSON-stored list from create_embeddings


def load_text_documents(data_dir: str) -> List[str]:
    """Load all .txt documents from a local directory recursively."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: List[str] = []
    loaded_files: List[str] = []
    for file_path in sorted(path.rglob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            documents.append(content)
            loaded_files.append(str(file_path))

    if not documents:
        raise ValueError(f"No .txt documents found in: {data_dir}")
    print(f"DEBUG: load_text_documents: loaded_files={len(loaded_files)}")
    for fp in loaded_files:
        print(f"DEBUG: load_text_documents: file={fp}")
    print(f"DEBUG: load_text_documents: loaded_documents={len(documents)}")
    return documents


def load_chunks_from_db() -> List[LoadedDbChunk]:
    """
    Load rows from ChunkEmbedding: chunk_text from related PageChunk, vector from JSON.
    Only includes rows with non-empty text and a usable vector (precomputed in DB).
    """
    print("DEBUG: load_chunks_from_db: starting (ChunkEmbedding + PageChunk chunk_text)")

    try:
        from chatbot.models import ChunkEmbedding
    except Exception as e:
        print(f"DEBUG: load_chunks_from_db: django model import failed: {e!r}")
        traceback.print_exc()
        return []

    try:
        from django.db import connection

        connection.ensure_connection()
    except Exception as e:
        print(f"DEBUG: load_chunks_from_db: DB connection check failed: {e!r}")
        traceback.print_exc()
        return []

    try:
        base_qs = ChunkEmbedding.objects.select_related("chunk").order_by("id")
        table_count = ChunkEmbedding.objects.count()
        print(
            f"DEBUG: load_chunks_from_db: ChunkEmbedding table readable, total rows: {table_count}"
        )
    except Exception as e:
        print(f"DEBUG: load_chunks_from_db: cannot query ChunkEmbedding: {e!r}")
        traceback.print_exc()
        return []

    rows: list[LoadedDbChunk] = []
    model_mismatch = 0
    empty_text = 0
    missing_vector = 0

    try:
        for emb in base_qs.iterator(chunk_size=500):
            text = (emb.chunk.chunk_text or "").strip() if emb.chunk else ""
            if not text:
                empty_text += 1
                continue
            if emb.vector is None:
                missing_vector += 1
                continue
            arr = np.asarray(emb.vector, dtype=np.float32)
            if len(arr.shape) == 0:
                missing_vector += 1
                continue
            if len(arr.shape) > 1:
                arr = arr.ravel()
            name = (emb.embedding_model or "").strip()
            if name and name != EXPECTED_EMBEDDING_MODEL:
                model_mismatch += 1
            rows.append(LoadedDbChunk(chunk_text=text, vector=arr))
    except Exception as e:
        print(f"DEBUG: load_chunks_from_db: iteration / row build failed: {e!r}")
        traceback.print_exc()
        return []

    print(
        f"DEBUG: load_chunks_from_db: loaded {len(rows)} text+vector pairs from DB "
        f"(empty_text_skipped={empty_text}, missing_vector={missing_vector}, "
        f"rows_with_embedding_model_mismatch={model_mismatch}, expected_model={EXPECTED_EMBEDDING_MODEL!r})"
    )
    if model_mismatch:
        print(
            "DEBUG: load_chunks_from_db: WARNING: some rows use a different "
            f"embedding_model than {EXPECTED_EMBEDDING_MODEL!r}; consider re-embedding for best recall."
        )

    return rows
