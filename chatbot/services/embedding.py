"""Vector-space retrieval against the ChunkEmbedding table.

Public capabilities:
    * ``_retrieve_top_chunks_by_embedding`` — cosine top-k over
      normalized vectors, with an optional metadata pre-filter derived
      from :func:`chatbot.services.query_parser.parse_query`.
    * ``_embed_query_normalized`` — turn a query string into a unit
      vector.
    * ``_embedding_matrix_pack`` — cached materialization of the DB
      matrix per source filter.

The HTTP-facing strict-rag verification view stays in the legacy module
/ future ``api/v1/views.py`` and consumes the helpers below — services
remain HTTP-agnostic.

Hybrid filter strategy (Step 4.3)
---------------------------------
RAG quality on a small Bologna corpus is dominated by namespace
disambiguation: "Bilgisayar Mühendisliği bölüm başkanı kim?" should hit
ONE chunk, not the program-overview page just because cosine puts it
0.01 higher. We solve that with a *hybrid* filter:

    1. Parse the query (regex parser, no LLM).
    2. If parser produced any structured filter, run a metadata-aware
       pass against ``PageChunk.metadata`` JSONB columns. If that pass
       returned at least :data:`_HYBRID_MIN_HARD_HITS` chunks, use it as
       the final result.
    3. Otherwise (no parser hits, or hard pass too thin) fall back to
       the original global cosine top-k. This guarantees the retriever
       never returns ``[]`` when there *is* relevant content somewhere.

The hard pass is intentionally small-set: when filters select 5–25
chunks we just stack them and run an O(N) cosine — no caching needed.
The full-corpus path keeps using ``_embedding_matrix_pack`` LRU.

Allowed dependency direction: embedding has no fan-out into other
services. ``chatbot.services.query_parser`` is a dependency of this
module and is deliberately pure (no DB / network); this avoids cycles.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np
from django.db.models import Count, Max, Q

from chatbot.services.query_parser import QueryFilters, parse_query

logger = logging.getLogger(__name__)

# Single source of truth for the embedding model used across the project.
# Moved here from rag/document_loader.py during the Adım 5.5 dead-code purge —
# the embedding capability is the only consumer that still mattered.
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

# Surfaced for the strict-rag verification HTTP wrapper; kept here so the constant
# travels with the embedding capability rather than the HTTP adapter.
_STRICT_RAG_NOT_FOUND = "BİLGİ BULUNAMADI (NO CONTEXT FROM DB)"

# Minimum number of chunks the metadata-aware hard pass must yield
# before we declare it sufficient. Below this the global fallback runs.
# Three strikes the right balance for a single-program pilot: enough to
# give the LLM more than one supporting passage, low enough that a
# specific question like "bölüm başkanı" (one true chunk + a couple of
# program-overview chunks) does not unnecessarily fall back.
_HYBRID_MIN_HARD_HITS = 3


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
def _embedding_matrix_pack(cache_key: tuple[str, int, int]) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """
    Tüm embedding satırlarını tek seferde matrise çevirir; cache_key (kaynak filtresi + satır sayısı + max id)
    değişene kadar @lru_cache ile bellekte tutulur — her /ask isteğinde 1947 kez JSON parse etmez.
    """
    from chatbot.models import ChunkEmbedding

    kind, _, __ = cache_key
    source_type = None if kind == "__all__" else kind

    qs = ChunkEmbedding.objects.select_related("chunk")
    if source_type:
        qs = qs.filter(chunk__source_type=source_type)

    vectors: list[np.ndarray] = []
    metas: list[dict[str, object]] = []

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
                # Surfaced through the API contract so the frontend can render source cards
                # (content_type, department, course_code, ...) without re-fetching the chunk.
                "metadata": dict(ch.metadata or {}),
            }
        )

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), ()

    mat = np.stack(vectors, axis=0)
    return mat, tuple(metas)


def _build_metadata_q(filters: QueryFilters, source_type: str | None) -> Q | None:
    """Translate :class:`QueryFilters` into a Django ``Q`` over PageChunk.

    Returns ``None`` if the parser produced nothing actionable (callers
    short-circuit and skip the hard pass entirely).

    Only the fields that the parser actually populated participate in
    the ``Q`` — a missing filter is *not* a wildcard match, it simply
    contributes nothing. ``content_types`` is OR-joined: a query with
    ``("bologna_contact", "contact")`` accepts either tag.
    """
    clauses: list[Q] = []
    if source_type:
        clauses.append(Q(chunk__source_type=source_type))
    if filters.department:
        clauses.append(Q(chunk__metadata__department=filters.department))
    if filters.faculty:
        clauses.append(Q(chunk__metadata__faculty=filters.faculty))
    if filters.course_code:
        clauses.append(Q(chunk__metadata__course_code=filters.course_code))
    if filters.semester is not None:
        clauses.append(Q(chunk__metadata__semester=filters.semester))
    if filters.content_types:
        ct_q = Q()
        for ct in filters.content_types:
            ct_q |= Q(chunk__metadata__content_type=ct)
        clauses.append(ct_q)

    if not clauses:
        return None

    combined = clauses[0]
    for q in clauses[1:]:
        combined &= q
    return combined


def _retrieve_with_metadata_filter(
    qv: np.ndarray,
    k: int,
    *,
    metadata_q: Q,
) -> list[dict]:
    """Cosine top-k over the ChunkEmbedding subset selected by ``metadata_q``.

    The subset is small by construction (a few dozen rows on the pilot
    corpus, even on a fully-loaded production tenant the per-program
    metadata cardinality keeps it well under a thousand). We pull the
    rows once, normalise them, and run a plain dense dot product; no
    LRU caching because the ``metadata_q`` permutations are unbounded
    and would defeat any reasonable cache budget.
    """
    from chatbot.models import ChunkEmbedding

    qs = (
        ChunkEmbedding.objects.select_related("chunk")
        .filter(metadata_q)
        .filter(Q(embedding_model="") | Q(embedding_model=EXPECTED_EMBEDDING_MODEL))
    )

    vectors: list[np.ndarray] = []
    metas: list[dict] = []
    for emb in qs.iterator(chunk_size=400):
        ch = emb.chunk
        if not ch:
            continue
        text = (ch.chunk_text or "").strip()
        if not text or emb.vector is None:
            continue
        arr = np.asarray(emb.vector, dtype=np.float32).ravel()
        if arr.size == 0:
            continue
        nrm = float(np.linalg.norm(arr) + 1e-12)
        vectors.append((arr / nrm).astype(np.float32, copy=False))
        metas.append(
            {
                "chunk_id": int(ch.pk),
                "url": (ch.url or "").strip(),
                "title": (ch.title or "").strip(),
                "text": text,
                # Same shape as the global pass so callers can serialize either result
                # (content_type, department, course_code, ...) uniformly.
                "metadata": dict(ch.metadata or {}),
            }
        )

    if not vectors:
        return []

    mat = np.stack(vectors, axis=0)
    if int(qv.shape[0]) != mat.shape[1]:
        logger.warning(
            "embedding_dim_mismatch (hard pass) q_dim=%s mat_dim=%s — boş dönülüyor",
            qv.shape[0],
            mat.shape[1],
        )
        return []

    sims = mat @ qv
    order = np.argsort(-sims)
    k_eff = max(1, min(int(k), len(order)))
    out: list[dict] = []
    for i in order[:k_eff]:
        row = dict(metas[int(i)])
        row["score"] = float(sims[int(i)])
        out.append(row)
    return out


def _retrieve_top_chunks_by_embedding(
    question: str,
    k: int,
    *,
    source_type: str | None = None,
    filters: QueryFilters | None = None,
) -> list[dict]:
    """
    Hybrid metadata-aware retrieval: PageChunk.metadata pre-filter on
    the parser hits, with global cosine fallback when the hard pass is
    empty or too thin (< _HYBRID_MIN_HARD_HITS).

    ``filters`` is optional: pass an explicit :class:`QueryFilters` to
    skip parsing (useful in tests), or ``None`` to let this function
    invoke :func:`parse_query` on ``question`` itself. ``source_type``
    can pin the search to a single ingestion source ("obs", "www") in
    addition to whatever the parser produced — they AND together.
    """
    from chatbot.models import ChunkEmbedding

    if filters is None:
        filters = parse_query(question)

    qv = _embed_query_normalized(question, EXPECTED_EMBEDDING_MODEL)

    # ---- 1) Hard metadata pass ----
    hard_hits: list[dict] = []
    metadata_q = _build_metadata_q(filters, source_type)
    if metadata_q is not None:
        hard_hits = _retrieve_with_metadata_filter(qv, k=k, metadata_q=metadata_q)
        logger.info(
            "retrieve(hybrid,hard): matched=%s filters=%s",
            len(hard_hits),
            filters.matched_terms,
        )
        if len(hard_hits) >= _HYBRID_MIN_HARD_HITS or len(hard_hits) >= max(1, k):
            return hard_hits
        # Hard pass produced something but not enough — keep what we have
        # and TOP UP from the global pass below; we never throw the
        # parser-selected chunks away on the way to the fallback.
        logger.info(
            "retrieve(hybrid,topup): hard_hits=%s below threshold (%s); merging with global",
            len(hard_hits),
            _HYBRID_MIN_HARD_HITS,
        )

    # ---- 2) Global cosine pass ----
    base = ChunkEmbedding.objects.select_related("chunk")
    if source_type:
        base = base.filter(chunk__source_type=source_type)
    sig = base.aggregate(c=Count("id"), mx=Max("id"))
    cache_key = (source_type or "__all__", int(sig["c"] or 0), int(sig["mx"] or 0))

    mat, metas_t = _embedding_matrix_pack(cache_key)
    metas = list(metas_t)
    if mat.shape[0] == 0 or mat.shape[1] == 0 or not metas:
        return []

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
    # Default 0.55 was calibrated in Adım 4.4 against the
    # paraphrase-multilingual-MiniLM-L12-v2 distribution: in-scope hard-pass
    # bands sit at 0.62-0.84 and out-of-scope/noise at 0.36-0.47, so 0.55 cleanly
    # separates them while preserving short course chunks (semester=1 grew below
    # 0.65 with the new model).
    raw_min = (os.environ.get("ACU_EMBEDDING_MIN_COSINE") or "0.55").strip()
    try:
        min_cos = float(raw_min)
    except ValueError:
        min_cos = 0.55
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
            # base_row["metadata"] is the cached dict; copy so downstream mutations
            # cannot bleed into the lru_cache-held tuple.
            "metadata": dict(base_row.get("metadata") or {}),
        }
        out.append(row)

    # ---- 3) Merge: parser-selected chunks always lead, global fills the rest. ----
    if hard_hits:
        merged: list[dict] = list(hard_hits)
        seen_ids = {h["chunk_id"] for h in merged}
        for row in out:
            if row["chunk_id"] in seen_ids:
                continue
            merged.append(row)
            seen_ids.add(row["chunk_id"])
            if len(merged) >= max(k, len(hard_hits)):
                break
        return merged[: max(k, len(hard_hits))]
    return out


__all__ = [
    "_STRICT_RAG_NOT_FOUND",
    "_sentence_transformer_for_model",
    "_embed_query_normalized",
    "_embedding_matrix_pack",
    "_retrieve_top_chunks_by_embedding",
    "_HYBRID_MIN_HARD_HITS",
]
