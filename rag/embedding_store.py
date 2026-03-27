from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class VectorStore:
    index: faiss.IndexFlatL2
    chunks: List[str]
    embedding_model_name: str


def build_faiss_index(
    chunks: List[str], embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> VectorStore:
    """Create embeddings for chunks and store them in a FAISS index."""
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
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
