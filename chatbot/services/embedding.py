"""Vector-space retrieval against the ChunkEmbedding table.

Public capabilities:
    * ``_retrieve_top_chunks_by_embedding`` — cosine top-k over normalized vectors.
    * ``_embed_query_normalized`` — turn a query string into a unit vector.
    * ``_embedding_matrix_pack`` — cached materialization of the DB matrix per source filter.

The HTTP-facing strict-rag verification view stays in the legacy module / future
``api/v1/views.py`` and consumes the helpers below — services remain HTTP-agnostic.

Allowed dependency direction: embedding has no fan-out into other services.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np
from django.db.models import Count, Max

logger = logging.getLogger(__name__)

# Surfaced for the strict-rag verification HTTP wrapper; kept here so the constant
# travels with the embedding capability rather than the HTTP adapter.
_STRICT_RAG_NOT_FOUND = "BİLGİ BULUNAMADI (NO CONTEXT FROM DB)"


@lru_cache(maxsize=1)
def _sentence_transformer_for_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_query_normalized(question: str, model_name: str) -> np.ndarray:
    """Single-query embedding, L2-normalized (cosine similarity via dot product)."""
    model = _sentence_transformer_for_model(model_name)
    v = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    out = np.asarray(v[0], dtype=np.float32).ravel()
    n = float(np.linalg.norm(out) + 1e-12)
    return out / n


@lru_cache(maxsize=16)
def _embedding_matrix_pack(cache_key: tuple[str, int, int]) -> tuple[np.ndarray, tuple[dict[str, str | int], ...]]:
    """
    Tüm embedding satırlarını tek seferde matrise çevirir; cache_key (kaynak filtresi + satır sayısı + max id)
    değişene kadar @lru_cache ile bellekte tutulur — her /ask isteğinde 1947 kez JSON parse etmez.
    """
    from chatbot.models import ChunkEmbedding
    from rag.document_loader import EXPECTED_EMBEDDING_MODEL

    kind, _, __ = cache_key
    source_type = None if kind == "__all__" else kind

    qs = ChunkEmbedding.objects.select_related("chunk")
    if source_type:
        qs = qs.filter(chunk__source_type=source_type)

    vectors: list[np.ndarray] = []
    metas: list[dict[str, str | int]] = []

    for emb in qs.iterator(chunk_size=800):
        ch = emb.chunk
        if not ch:
            continue
        text = (ch.chunk_text or "").strip()
        if not text:
            continue
        if emb.vector is None:
            continue
        name = (emb.embedding_model or "").strip()
        if name and name != EXPECTED_EMBEDDING_MODEL:
            continue
        arr = np.asarray(emb.vector, dtype=np.float32).ravel()
        if arr.size == 0:
            continue
        nrm = float(np.linalg.norm(arr) + 1e-12)
        vn = arr / nrm
        vectors.append(vn.astype(np.float32, copy=False))
        metas.append(
            {
                "chunk_id": int(ch.pk),
                "url": (ch.url or "").strip(),
                "title": (ch.title or "").strip(),
                "text": text,
            }
        )

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), ()

    mat = np.stack(vectors, axis=0)
    return mat, tuple(metas)


def _retrieve_top_chunks_by_embedding(
    question: str,
    k: int,
    *,
    source_type: str | None = None,
) -> list[dict]:
    """
    Read-only retrieval: ChunkEmbedding.vector + PageChunk.chunk_text.
    source_type='obs' ile yalnızca OBS chunk'ları taranır (ders kataloğu için çok daha hızlı).
    """
    from chatbot.models import ChunkEmbedding
    from rag.document_loader import EXPECTED_EMBEDDING_MODEL

    base = ChunkEmbedding.objects.select_related("chunk")
    if source_type:
        base = base.filter(chunk__source_type=source_type)
    sig = base.aggregate(c=Count("id"), mx=Max("id"))
    cache_key = (source_type or "__all__", int(sig["c"] or 0), int(sig["mx"] or 0))

    mat, metas_t = _embedding_matrix_pack(cache_key)
    metas = list(metas_t)
    if mat.shape[0] == 0 or mat.shape[1] == 0 or not metas:
        return []

    qv = _embed_query_normalized(question, EXPECTED_EMBEDDING_MODEL)
    if int(qv.shape[0]) != mat.shape[1]:
        logger.warning(
            "embedding_dim_mismatch q_dim=%s mat_dim=%s — boş dönülüyor",
            qv.shape[0],
            mat.shape[1],
        )
        return []

    sims = mat @ qv
    order = np.argsort(-sims)
    k_eff = max(1, min(int(k), len(order)))
    # Examine extra candidates so a similarity floor can still return up to k_eff chunks.
    pool = min(len(order), max(k_eff * 5, k_eff + 16))
    top_idx = order[:pool]

    # Min cosine similarity (cosine = dot product on L2-normalized rows). Lower = more permissive.
    # If nothing meets the bar, fall back to plain top-k so "veri yok" does not trigger from this gate alone.
    raw_min = (os.environ.get("ACU_EMBEDDING_MIN_COSINE") or "0.65").strip()
    try:
        min_cos = float(raw_min)
    except ValueError:
        min_cos = 0.65
    min_cos = max(0.0, min(min_cos, 0.999))

    picked: list[int] = []
    for i in top_idx:
        if float(sims[int(i)]) >= min_cos:
            picked.append(int(i))
        if len(picked) >= k_eff:
            break
    use_idx = picked if picked else [int(i) for i in top_idx[:k_eff]]

    out: list[dict] = []
    for i in use_idx:
        base_row = metas[int(i)]
        row = {
            "chunk_id": base_row["chunk_id"],
            "url": base_row["url"],
            "title": base_row["title"],
            "text": base_row["text"],
            "score": float(sims[int(i)]),
        }
        out.append(row)
    return out


__all__ = [
    "_STRICT_RAG_NOT_FOUND",
    "_sentence_transformer_for_model",
    "_embed_query_normalized",
    "_embedding_matrix_pack",
    "_retrieve_top_chunks_by_embedding",
]
