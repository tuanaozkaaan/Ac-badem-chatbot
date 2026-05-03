from __future__ import annotations

"""
RAG flow: retrieval here; LLM calls go through model.local_llm.LocalLLM.
Ollama URL is OLLAMA_BASE_URL (e.g. http://ollama:11434 in Docker —
Compose service name `ollama`, not the container name).

Step 4.3 (this revision)
------------------------
The previous version of this module hosted a constellation of intent-
specific heuristics — engineering-only keyword boosts, an
"Eczacılık fakülte sayma" exclusion list, a department-alias merge,
a hand-written engineering-departments local-data short-circuit, and
post-processing that surgically removed duplicate "MBG / Moleküler
Biyoloji" lines. All of those decisions were specific to a single
faculty in the pilot and made adding the next program a copy-paste
chore.

They are now centralized in :mod:`chatbot.services.query_parser`, which
returns a :class:`QueryFilters` instance the retriever uses to focus on
the right metadata namespace. The pipeline below is a thin shell:
embed query → cosine top-k from the in-memory FAISS store →
parser-aware metadata sanity filter → LLM. New intents and new
departments are now one regex line in the parser, not new branches in
this file.
"""
import json
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

import numpy as np

from model.local_llm import LocalLLM
from rag.document_loader import (
    EXPECTED_EMBEDDING_MODEL,
    load_chunks_from_db,
    load_text_documents,
)
from rag.embedding_store import (
    VectorStore,
    build_faiss_index,
    complete_embedding_matrix,
    embed_query,
    search_top_k,
)
from rag.text_splitter import split_into_chunks


@dataclass
class RAGConfig:
    data_dir: str = "data"
    prefer_db_chunks: bool = True
    # Must match create_embeddings and ChunkEmbedding defaults
    # (EXPECTED_EMBEDDING_MODEL).
    embedding_model_name: str = EXPECTED_EMBEDDING_MODEL
    top_k: int = 7
    # IndexFlatL2 / embed_query: squared L2 on normalized vectors.
    # Larger = worse match. Reject when best match is worse than this
    # (strict inequality: > threshold => no context).
    max_distance_threshold: float = 2.35


class RAGSystem:
    def __init__(self, llm: LocalLLM, config: RAGConfig | None = None) -> None:
        self.llm = llm
        self.config = config or RAGConfig()
        self.store: VectorStore | None = None

    # ------------------------------------------------------------------
    # Knowledge base construction
    # ------------------------------------------------------------------
    def build_knowledge_base(self) -> None:
        total_start = perf_counter()
        if self.config.embedding_model_name != EXPECTED_EMBEDDING_MODEL:
            print(
                "DEBUG: build_knowledge_base: WARNING: embedding_model_name="
                f"{self.config.embedding_model_name!r} != stored vectors' "
                f"model {EXPECTED_EMBEDDING_MODEL!r}"
            )

        chunks: List[str] = []
        per_row: Optional[List[Optional[np.ndarray]]] = None

        if self.config.prefer_db_chunks:
            print("DEBUG: build_knowledge_base: prefer_db_chunks=True, loading from DB...")
            loaded = load_chunks_from_db()
            if loaded:
                chunks = [row.chunk_text for row in loaded]
                per_row = [row.vector for row in loaded]
                print(
                    f"DEBUG: build_knowledge_base: DB path — {len(chunks)} chunk texts, "
                    f"vectors present per row: "
                    f"{per_row is not None and all(v is not None for v in per_row)}"
                )
            else:
                print("DEBUG: build_knowledge_base: DB load returned 0 rows, will try files.")

        if not chunks:
            print(
                "DEBUG: build_knowledge_base: loading from data_dir="
                f"{self.config.data_dir!r} (txt files)"
            )
            docs = load_text_documents(self.config.data_dir)
            chunks = split_into_chunks(docs)
            per_row = None
            print(f"DEBUG: build_knowledge_base: {len(chunks)} chunks from files after split")

        embedding_start = perf_counter()
        if per_row is not None and any(v is not None for v in per_row):
            if all(v is not None for v in per_row):
                matrix = np.stack([t for t in per_row if t is not None]).astype(np.float32)
                print(
                    "DEBUG: build_knowledge_base: FAISS from DB precomputed vectors only, "
                    f"matrix shape {matrix.shape}, no chunk re-encoding"
                )
                self.store = build_faiss_index(
                    chunks=chunks,
                    embedding_model_name=self.config.embedding_model_name,
                    precomputed_vectors=matrix,
                )
            else:
                print(
                    "DEBUG: build_knowledge_base: mixed precomputed + missing rows; "
                    "filling with SentenceTransformer encode"
                )
                matrix = complete_embedding_matrix(
                    chunks, per_row, self.config.embedding_model_name
                )
                self.store = build_faiss_index(
                    chunks=chunks,
                    embedding_model_name=self.config.embedding_model_name,
                    precomputed_vectors=matrix,
                )
        else:
            print("DEBUG: build_knowledge_base: FAISS from full encode (no DB vectors for this run)")
            self.store = build_faiss_index(
                chunks=chunks,
                embedding_model_name=self.config.embedding_model_name,
                precomputed_vectors=None,
            )
        embedding_ms = (perf_counter() - embedding_start) * 1000
        total_ms = (perf_counter() - total_start) * 1000
        print(
            json.dumps(
                {
                    "event": "latency",
                    "stage": "build_knowledge_base",
                    "embedding_index_ms": round(embedding_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "chunks": len(chunks),
                },
                ensure_ascii=True,
            )
        )

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def answer(self, question: str) -> str:
        # Imported lazily so a CLI that only calls ``build_knowledge_base``
        # does not have to load the parser regex tables.
        from chatbot.services.query_parser import QueryFilters, parse_query

        total_start = perf_counter()
        if not self.store:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")

        filters = parse_query(question)

        embedding_start = perf_counter()
        query_vector = embed_query(question, self.store.embedding_model_name)
        embedding_ms = (perf_counter() - embedding_start) * 1000

        # Pull a wider candidate pool than ``top_k`` so that a parser-
        # driven post-filter can still produce ``top_k`` survivors.
        retrieval_start = perf_counter()
        pool_k = max(self.config.top_k * 3, self.config.top_k + 6)
        retrieved = search_top_k(self.store, query_vector, k=pool_k)
        retrieved = self._apply_parser_filter(retrieved, filters)
        retrieved = retrieved[: self.config.top_k]
        retrieval_ms = (perf_counter() - retrieval_start) * 1000

        if not retrieved:
            print(
                "DEBUG: answer: search_top_k empty (index empty?); "
                f"index ntotal={getattr(self.store.index, 'ntotal', 'n/a')}"
            )
            self._log_latency(
                stage="answer",
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_start=total_start,
                retrieved=0,
                kept=0,
                reason="empty_retrieval",
            )
            return "The requested information is not available in the provided context."

        # FAISS returns squared L2 for IndexFlatL2; compare in the same
        # space as ``max_distance_threshold``.
        best_distance = retrieved[0][1]
        print(
            f"DEBUG: answer: top_k={self.config.top_k}, best_l2_sq={best_distance:.4f}, "
            f"threshold={self.config.max_distance_threshold}, "
            f"model={self.store.embedding_model_name!r}, "
            f"parser_filters={filters.matched_terms or 'none'}"
        )
        if best_distance > self.config.max_distance_threshold:
            print(
                "DEBUG: answer: best squared L2 over threshold; returning not-available message"
            )
            self._log_latency(
                stage="answer",
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_start=total_start,
                retrieved=len(retrieved),
                kept=0,
                reason="distance_threshold",
            )
            return "The requested information is not available in the provided context."

        context_blocks = self._deduplicate_context_blocks([chunk for chunk, _ in retrieved])
        context = "\n\n".join(context_blocks)

        prompt = self._build_prompt(question=question, context=context, filters=filters)

        print("DEBUG: answer: --- context (before Ollama) ---")
        print(context)
        print("DEBUG: answer: --- end context ---")

        generation_start = perf_counter()
        response = self.llm.generate(prompt=prompt)
        generation_ms = (perf_counter() - generation_start) * 1000
        self._log_latency(
            stage="answer",
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_start=total_start,
            retrieved=len(retrieved),
            kept=len(context_blocks),
            reason="ok",
        )
        if not response:
            print(
                "DEBUG: answer: LLM returned empty/whitespace; "
                "using not-available (see Ollama error logs above if any)."
            )
            return "The requested information is not available in the provided context."

        return response.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_parser_filter(
        retrieved: List[tuple[str, float]],
        filters,
    ) -> List[tuple[str, float]]:
        """Down-rank chunks whose text body contradicts the parsed filters.

        The in-memory FAISS store does not carry ``PageChunk.metadata``
        (DB layer does), so we run a lightweight text-substring check
        here as a defensive layer. The contract is:

          * A chunk that obviously names the WRONG department for a
            question whose parser pinned a specific one is dropped.
          * Otherwise we leave the cosine ordering untouched — this
            method is conservative: if it cannot prove a chunk is
            wrong, it lets it through.
        """
        if filters is None or filters.is_empty():
            return retrieved
        if not filters.department:
            return retrieved

        wanted = filters.department.lower()
        # Bare fragment: "Bilgisayar Mühendisliği" → "bilgisayar mühendisli"
        # so we forgive trailing inflection ("...ği'nin", "...ğine").
        wanted_root = wanted.rstrip("i").rstrip("ı").rstrip(" ")

        kept: list[tuple[str, float]] = []
        for chunk, distance in retrieved:
            body = (chunk or "").lower()
            # If the wanted department's root appears, keep.
            if wanted_root in body:
                kept.append((chunk, distance))
                continue
            # If no other engineering / medical department names appear
            # either, the chunk is generic and remains a candidate.
            if not RAGSystem._mentions_other_department(body, wanted_root):
                kept.append((chunk, distance))
        return kept or retrieved

    @staticmethod
    def _mentions_other_department(body: str, wanted_root: str) -> bool:
        # Cheap heuristic: looking for "X Mühendisliği" substrings that
        # are not the wanted root. Keeps the list short on purpose so
        # adding a new department in the parser does not also force an
        # edit here.
        candidates = (
            "biyomedikal mühendisli",
            "endüstri mühendisli",
            "elektrik-elektronik mühendisli",
            "moleküler biyoloji",
            "tıp fakülte",
            "diş hekimli",
            "eczacılık",
            "hemşirelik",
            "fizyoterapi",
        )
        for cand in candidates:
            if cand == wanted_root:
                continue
            if cand in body:
                return True
        return False

    @staticmethod
    def _deduplicate_context_blocks(context_blocks: List[str]) -> List[str]:
        """Plain whitespace-normalised dedup, no alias merging.

        The previous implementation tried to canonicalise "MBG" vs
        "Moleküler Biyoloji ve Genetik" before deduping; the parser now
        owns alias handling, and the LLM does not need us to rewrite
        chunks for it.
        """
        deduplicated: list[str] = []
        seen: set[str] = set()
        for block in context_blocks:
            fingerprint = " ".join((block or "").split()).strip().lower()
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduplicated.append(block)
        return deduplicated

    @staticmethod
    def _build_prompt(*, question: str, context: str, filters) -> str:
        focus_lines: list[str] = []
        if filters and not filters.is_empty():
            if filters.department:
                focus_lines.append(f"Hedeflenen bölüm: {filters.department}")
            if filters.faculty:
                focus_lines.append(f"Hedeflenen fakülte: {filters.faculty}")
            if filters.course_code:
                focus_lines.append(f"Ders kodu: {filters.course_code}")
            if filters.semester is not None:
                focus_lines.append(f"Yarıyıl: {filters.semester}")
            if filters.content_types:
                focus_lines.append("İçerik türü: " + ", ".join(filters.content_types))
        focus_block = ("\n".join(focus_lines) + "\n\n") if focus_lines else ""

        return (
            "You are a precise question-answering assistant.\n"
            "Use only the context. Do not invent facts.\n"
            "If the context includes a section list, return only those sections "
            "as bullet points.\n"
            "Ignore announcements and seminars unless the question is explicitly "
            "about them.\n\n"
            f"{focus_block}"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @staticmethod
    def _log_latency(
        *,
        stage: str,
        embedding_ms: float,
        retrieval_ms: float,
        generation_ms: float,
        total_start: float,
        retrieved: int,
        kept: int,
        reason: str,
    ) -> None:
        total_ms = (perf_counter() - total_start) * 1000
        print(
            json.dumps(
                {
                    "event": "latency",
                    "stage": stage,
                    "embedding_ms": round(embedding_ms, 1),
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "retrieved": retrieved,
                    "kept": kept,
                    "reason": reason,
                },
                ensure_ascii=True,
            )
        )
