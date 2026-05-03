from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Defaults are kept in sync with the single source of truth in
# ``rag.document_loader`` so that ``build_faiss_index`` / ``embed_query``
# can never silently disagree with what ``ChunkEmbedding`` rows were
# encoded with — a dimension mismatch (wrong model) causes empty
# retrieval and is one of the easier ways to break a RAG demo.
from rag.document_loader import EXPECTED_EMBEDDING_MODEL


@dataclass
class VectorStore:
    index: faiss.IndexFlatL2
    chunks: List[str]
    embedding_model_name: str


def complete_embedding_matrix(
    chunks: List[str],
    per_row: Sequence[Optional[np.ndarray]],
    embedding_model_name: str = EXPECTED_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Build a (n, dim) matrix: use precomputed row vectors when present, encode missing
    rows with the same model settings as the encode-all path (normalize_embeddings=True).
    """
    if len(chunks) != len(per_row):
        raise ValueError("chunks and per_row must have the same length")

    n = len(chunks)
    missing_idx = [i for i in range(n) if per_row[i] is None]
    if not missing_idx:
        return np.stack([per_row[i] for i in range(n)]).astype(np.float32)  # type: ignore[list-item]

    if not any(p is not None for p in per_row):
        model = SentenceTransformer(embedding_model_name)
        return model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype(
            np.float32
        )

    sample = next(p for p in per_row if p is not None)
    dim = int(sample.shape[0])
    out = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        v = per_row[i]
        if v is not None:
            out[i] = v.astype(np.float32, copy=False)

    model = SentenceTransformer(embedding_model_name)
    to_encode = [chunks[i] for i in missing_idx]
    encoded = model.encode(to_encode, convert_to_numpy=True, normalize_embeddings=True)
    for j, i in enumerate(missing_idx):
        out[i] = encoded[j].astype(np.float32, copy=False)
    return out


def build_faiss_index(
    chunks: List[str],
    embedding_model_name: str = EXPECTED_EMBEDDING_MODEL,
    precomputed_vectors: Optional[np.ndarray] = None,
) -> VectorStore:
    """
    Create a FAISS L2 index for chunks. If precomputed_vectors is set (n, dim) float32,
    they are used as-is (no model.encode for chunks). Otherwise chunks are embedded.
    """
    if not chunks and precomputed_vectors is not None and len(precomputed_vectors):
        raise ValueError("precomputed_vectors provided but chunks list is empty")
    if precomputed_vectors is not None:
        embeddings = np.asarray(precomputed_vectors, dtype=np.float32, order="C")
    else:
        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)

    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    if precomputed_vectors is not None and len(chunks) != len(embeddings):
        raise ValueError("chunks and precomputed_vectors row counts must match")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    if embeddings.shape[0] > 0:
        index.add(embeddings)

    return VectorStore(index=index, chunks=chunks, embedding_model_name=embedding_model_name)


def embed_query(query: str, embedding_model_name: str) -> np.ndarray:
    model = SentenceTransformer(embedding_model_name)
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vector.astype(np.float32)


def search_top_k(
    store: VectorStore, query_vector: np.ndarray, k: int = 3
) -> List[Tuple[str, float]]:
    distances, indices = store.index.search(query_vector, k)
    results: List[Tuple[str, float]] = []

    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        results.append((store.chunks[idx], float(distance)))
    return results
