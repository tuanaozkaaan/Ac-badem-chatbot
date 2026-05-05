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

Hybrid filter strategy (Step 4.3) — **hard pass opt-in**
-------------------------------------------------------
The metadata-aware *hard* pass (parser → SQL pre-filter) is **disabled by
default (set :envvar:`ACU_ENABLE_HYBRID_HARD_PASS` to ``1`` to restore).
Without it, every query uses the tier-aware global cosine path only, which
avoids empty or off-partition hits on cross-cutting questions (people, FAQ).

When enabled, the historical behaviour returns: structured filters from
:func:`~chatbot.services.query_parser.parse_query` may shrink the candidate
set before cosine ranking.

Tiered global ranking (``data/*.txt`` VIP)
-------------------------------------------
The full-corpus path uses ``_embedding_matrix_pack`` LRU. Rows with
``metadata["tier"]=="primary"`` are checked first: if the best cosine on that
subset clears :envvar:`ACU_TIER_PRIMARY_TRUST_MIN`, those rows win exclusively.
Otherwise **all** primary and secondary row indices are pooled, sorted by cosine
descending, deduped implicitly by top-k selection from that order (secondary is
never skipped when primary confidence is below the trust bar).

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

from chatbot.services.constants import (
    CORPUS_TIER_PRIMARY,
    DEFAULT_ACU_EMBEDDING_MIN_COSINE,
    DEFAULT_ACU_TIER_PRIMARY_TRUST_MIN,
)
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


def _hybrid_hard_pass_enabled() -> bool:
    raw = (os.environ.get("ACU_ENABLE_HYBRID_HARD_PASS") or "0").strip().lower()
    return raw in ("1", "true", "yes")


def _read_min_cosine_floor() -> float:
    raw_min = (os.environ.get("ACU_EMBEDDING_MIN_COSINE") or "").strip()
    base = DEFAULT_ACU_EMBEDDING_MIN_COSINE if not raw_min else raw_min
    try:
        min_cos = float(base)
    except ValueError:
        min_cos = DEFAULT_ACU_EMBEDDING_MIN_COSINE
    return max(0.0, min(min_cos, 0.999))


def _read_primary_tier_trust_min() -> float:
    """Best primary cosine must reach this to skip mixed primary+secondary ranking."""
    raw = (os.environ.get("ACU_TIER_PRIMARY_TRUST_MIN") or "").strip()
    base = DEFAULT_ACU_TIER_PRIMARY_TRUST_MIN if not raw else raw
    try:
        v = float(base)
    except ValueError:
        v = DEFAULT_ACU_TIER_PRIMARY_TRUST_MIN
    v = max(0.0, min(v, 0.999))
    # Guard: mis-set env (e.g. 0.35) must not enable primary-only on weak matches.
    if v < 0.60:
        v = DEFAULT_ACU_TIER_PRIMARY_TRUST_MIN
    return v


def _chunk_tier(meta_row: dict[str, object]) -> str | None:
    md = meta_row.get("metadata")
    if not isinstance(md, dict):
        return None
    t = md.get("tier")
    return str(t) if t is not None else None


def _pick_use_idx_ordered(
    sims: np.ndarray,
    ordered_global_idx: list[int],
    k_eff: int,
    min_cos: float,
) -> list[int]:
    """Prefer scores above ``min_cos``, but always fall back to plain top-``k_eff``.

    Short entity queries can land below a cosine floor; the LLM still needs those
    chunks. Set ``ACU_EMBEDDING_MIN_COSINE=0`` to skip the soft preference entirely.
    """
    pool_n = min(len(ordered_global_idx), max(k_eff * 5, k_eff + 16))
    top_idx = ordered_global_idx[:pool_n]
    if min_cos <= 0:
        return [int(i) for i in top_idx[:k_eff]]
    picked: list[int] = []
    for i in top_idx:
        if float(sims[int(i)]) >= min_cos:
            picked.append(int(i))
        if len(picked) >= k_eff:
            break
    return picked if picked else [int(i) for i in top_idx[:k_eff]]


def _merge_primary_secondary_picks(
    sims: np.ndarray,
    idx_primary: list[int],
    idx_secondary: list[int],
    k_eff: int,
    min_cos: float,
) -> list[int]:
    """Pool primary + secondary rows, rank by cosine only, then apply top-k / min_cos gate.

    Primary-first ordering would starve high-scoring scraped hits; the LLM receives
    the best matches regardless of tier when we are not in the high-confidence
    primary-only mode.
    """
    combined: list[int] = sorted(
        {int(j) for j in (*idx_primary, *idx_secondary)},
        key=lambda j: float(sims[int(j)]),
        reverse=True,
    )
    if not combined:
        return []
    return _pick_use_idx_ordered(sims, combined, k_eff, min_cos)


def _tiered_global_use_indices(
    sims: np.ndarray,
    metas: list[dict[str, object]],
    k_eff: int,
    min_cos: float,
    tier_trust: float,
) -> tuple[list[int], str]:
    """Tier-aware ranking: strict primary-only when confident; else primary+secondary merge."""
    n = len(metas)
    idx_primary = [j for j in range(n) if _chunk_tier(metas[j]) == CORPUS_TIER_PRIMARY]
    idx_secondary = [j for j in range(n) if _chunk_tier(metas[j]) != CORPUS_TIER_PRIMARY]

    if idx_primary:
        p_arr = np.array(idx_primary, dtype=np.intp)
        sub_sims = sims[p_arr]
        best_primary = float(np.max(sub_sims))
        if best_primary >= tier_trust:
            order_local = np.argsort(-sub_sims)
            ordered_global = [int(idx_primary[int(i)]) for i in order_local]
            return (
                _pick_use_idx_ordered(sims, ordered_global, k_eff, min_cos),
                f"primary_trust best_cos={best_primary:.4f}>={tier_trust}",
            )
        merged = _merge_primary_secondary_picks(
            sims, idx_primary, idx_secondary, k_eff, min_cos
        )
        return (
            merged,
            f"merge_primary_secondary primary_best={best_primary:.4f}<{tier_trust}",
        )

    ordered_global = [int(i) for i in np.argsort(-sims)]
    return (
        _pick_use_idx_ordered(sims, ordered_global, k_eff, min_cos),
        "no_primary_tier_rows",
    )


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

    Scope rule (Adım 5.5+)
    ----------------------
    ``faculty`` and ``content_types`` are applied only when the question
    already pins a *programmatic* slice of the corpus via ``department``,
    ``course_code``, or ``semester``. Otherwise a lone intent tag (e.g.
    ``bologna_academic_staff`` from "personnel" wording) or a bare faculty
    mention would shrink retrieval to the wrong partition and miss
    entity/location answers that live under unrelated ``content_type``
    rows or generic www text.
    """
    clauses: list[Q] = []
    if source_type:
        clauses.append(Q(chunk__source_type=source_type))

    narrow_program_scope = bool(
        filters.department or filters.course_code or filters.semester is not None
    )

    if filters.department:
        clauses.append(Q(chunk__metadata__department=filters.department))
    if filters.course_code:
        clauses.append(Q(chunk__metadata__course_code=filters.course_code))
    if filters.semester is not None:
        clauses.append(Q(chunk__metadata__semester=filters.semester))

    if narrow_program_scope:
        if filters.faculty:
            clauses.append(Q(chunk__metadata__faculty=filters.faculty))
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

    # ---- 1) Hard metadata pass (opt-in: ACU_ENABLE_HYBRID_HARD_PASS=1) ----
    hard_hits: list[dict] = []
    metadata_q: Q | None = None
    if _hybrid_hard_pass_enabled():
        metadata_q = _build_metadata_q(filters, source_type)
        if metadata_q is not None:
            hard_hits = _retrieve_with_metadata_filter(qv, k=k, metadata_q=metadata_q)
            logger.info(
                "retrieve(hybrid,hard): matched=%s filters=%s",
                len(hard_hits),
                filters.matched_terms,
            )
            if len(hard_hits) >= _HYBRID_MIN_HARD_HITS or len(hard_hits) >= max(1, k):
                logger.warning(
                    "RAG_RETRIEVE hybrid_done count=%s mode=hard_only k=%s matched_terms=%s",
                    len(hard_hits),
                    k,
                    filters.matched_terms,
                )
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
        logger.warning(
            "RAG_RETRIEVE empty_embedding_matrix: no ChunkEmbedding rows usable "
            "(expected_model=%r). Run ingest + chunk + create_embeddings.",
            EXPECTED_EMBEDDING_MODEL,
        )
        return []

    if int(qv.shape[0]) != mat.shape[1]:
        logger.warning(
            "embedding_dim_mismatch q_dim=%s mat_dim=%s — boş dönülüyor",
            qv.shape[0],
            mat.shape[1],
        )
        return []

    sims = mat @ qv
    k_eff = max(1, min(int(k), len(sims)))
    tier_trust = _read_primary_tier_trust_min()
    min_cos = _read_min_cosine_floor()

    use_idx, tier_mode = _tiered_global_use_indices(sims, metas, k_eff, min_cos, tier_trust)

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

    logger.warning(
        "RAG_RETRIEVE tiered_global mode=%s k_eff=%s min_cos=%s tier_trust=%s",
        tier_mode,
        k_eff,
        min_cos,
        tier_trust,
    )

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
        final = merged[: max(k, len(hard_hits))]
        logger.warning(
            "RAG_RETRIEVE hybrid_done count=%s mode=merged hard_hits=%s k=%s matched_terms=%s",
            len(final),
            len(hard_hits),
            k,
            filters.matched_terms,
        )
        return final
    logger.warning(
        "RAG_RETRIEVE hybrid_done count=%s mode=global hard_hits=%s k=%s matched_terms=%s",
        len(out),
        len(hard_hits),
        k,
        filters.matched_terms,
    )
    return out


__all__ = [
    "_STRICT_RAG_NOT_FOUND",
    "_sentence_transformer_for_model",
    "_embed_query_normalized",
    "_embedding_matrix_pack",
    "_retrieve_top_chunks_by_embedding",
    "_HYBRID_MIN_HARD_HITS",
]
