from __future__ import annotations

"""
RAG flow: retrieval here; LLM calls go through model.local_llm.LocalLLM.
Ollama URL is OLLAMA_BASE_URL (e.g. http://ollama:11434 in Docker — Compose service name `ollama`, not the container name).
"""
from dataclasses import dataclass
import json
import re
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import numpy as np
from model.local_llm import LocalLLM
from rag.document_loader import (
    EXPECTED_EMBEDDING_MODEL,
    load_chunks_from_db,
    load_text_documents,
)
from rag.text_splitter import split_into_chunks
from rag.embedding_store import (
    VectorStore,
    build_faiss_index,
    complete_embedding_matrix,
    embed_query,
    search_top_k,
)


EXCLUDED_CONTEXT_TERMS: tuple[str, ...] = ("seminar", "seminer")
ENGINEERING_CONTEXT_EXCLUDE_TERMS: tuple[str, ...] = ("eczacilik", "eczacılık", "seminer", "seminar")
ENGINEERING_BOOST_TERMS: tuple[str, ...] = (
    "muhendislik",
    "mühendislik",
    "bilgisayar muhendisligi",
    "bilgisayar mühendisliği",
    "biyomedikal muhendisligi",
    "biyomedikal mühendisliği",
    "mbg",
    "molekuler biyoloji",
    "moleküler biyoloji",
)
DEPARTMENT_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "Molekuler Biyoloji ve Genetik (MBG)": (
        "molekuler biyoloji ve genetik",
        "moleküler biyoloji ve genetik",
        "molekuler biyoloji",
        "moleküler biyoloji",
        "mbg",
    ),
    "Bilgisayar Muhendisligi": (
        "bilgisayar muhendisligi",
        "bilgisayar mühendisliği",
    ),
    "Biyomedikal Muhendisligi": (
        "biyomedikal muhendisligi",
        "biyomedikal mühendisliği",
    ),
}
ENGINEERING_FACULTY_LOCAL_DATA_FILE = "engineering_natural_sciences_departments.txt"


@dataclass
class RAGConfig:
    data_dir: str = "data"
    prefer_db_chunks: bool = True
    # Must match create_embeddings and ChunkEmbedding defaults (EXPECTED_EMBEDDING_MODEL).
    embedding_model_name: str = EXPECTED_EMBEDDING_MODEL
    top_k: int = 7
    # IndexFlatL2 / embed_query: squared L2 on normalized vectors. Larger = worse match.
    # Reject when best match is worse than this (strict inequality: > threshold => no context).
    max_distance_threshold: float = 2.0


class RAGSystem:
    def __init__(self, llm: LocalLLM, config: RAGConfig | None = None) -> None:
        self.llm = llm
        self.config = config or RAGConfig()
        self.store: VectorStore | None = None

    def build_knowledge_base(self) -> None:
        total_start = perf_counter()
        if self.config.embedding_model_name != EXPECTED_EMBEDDING_MODEL:
            print(
                f"DEBUG: build_knowledge_base: WARNING: embedding_model_name={self.config.embedding_model_name!r} "
                f"!= stored vectors' model {EXPECTED_EMBEDDING_MODEL!r}"
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
                    f"vectors present per row: {per_row is not None and all(v is not None for v in per_row)}"
                )
            else:
                print("DEBUG: build_knowledge_base: DB load returned 0 rows, will try files.")

        if not chunks:
            print(
                f"DEBUG: build_knowledge_base: loading from data_dir={self.config.data_dir!r} (txt files)"
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

    def answer(self, question: str) -> str:
        total_start = perf_counter()
        if not self.store:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")

        if self._is_engineering_faculty_departments_question(question):
            local_departments = self._load_engineering_departments_from_local_data()
            if local_departments:
                return self._format_engineering_departments_answer(local_departments)

        embedding_start = perf_counter()
        query_vector = embed_query(question, self.store.embedding_model_name)
        embedding_ms = (perf_counter() - embedding_start) * 1000

        retrieval_start = perf_counter()
        retrieved = search_top_k(self.store, query_vector, k=self.config.top_k)
        retrieved = self._apply_engineering_keyword_boost(question, retrieved)
        retrieval_ms = (perf_counter() - retrieval_start) * 1000

        if not retrieved:
            print(
                "DEBUG: answer: search_top_k empty (index empty?); "
                f"index ntotal={getattr(self.store.index, 'ntotal', 'n/a')}"
            )
            total_ms = (perf_counter() - total_start) * 1000
            print(
                json.dumps(
                    {
                        "event": "latency",
                        "stage": "answer",
                        "embedding_ms": round(embedding_ms, 1),
                        "retrieval_ms": round(retrieval_ms, 1),
                        "generation_ms": 0.0,
                        "total_ms": round(total_ms, 1),
                        "retrieved": 0,
                        "kept": 0,
                        "reason": "empty_retrieval",
                    },
                    ensure_ascii=True,
                )
            )
            return "The requested information is not available in the provided context."

        # FAISS returns squared L2 for IndexFlatL2; compare in the same space as max_distance_threshold.
        best_distance = retrieved[0][1]
        print(
            f"DEBUG: answer: top_k={self.config.top_k}, best_l2_sq={best_distance:.4f}, "
            f"threshold={self.config.max_distance_threshold}, model={self.store.embedding_model_name!r}"
        )
        if best_distance > self.config.max_distance_threshold:
            print(
                "DEBUG: answer: best squared L2 over threshold; returning not-available message"
            )
            total_ms = (perf_counter() - total_start) * 1000
            print(
                json.dumps(
                    {
                        "event": "latency",
                        "stage": "answer",
                        "embedding_ms": round(embedding_ms, 1),
                        "retrieval_ms": round(retrieval_ms, 1),
                        "generation_ms": 0.0,
                        "total_ms": round(total_ms, 1),
                        "retrieved": len(retrieved),
                        "kept": 0,
                        "reason": "distance_threshold",
                    },
                    ensure_ascii=True,
                )
            )
            return "The requested information is not available in the provided context."

        filtered_retrieved = [
            (chunk, distance)
            for chunk, distance in retrieved
            if not self._is_excluded_chunk(chunk)
            and not self._is_engineering_excluded_chunk(question, chunk)
        ]
        if not filtered_retrieved:
            print("DEBUG: answer: all retrieved chunks excluded by seminar filter.")
            total_ms = (perf_counter() - total_start) * 1000
            print(
                json.dumps(
                    {
                        "event": "latency",
                        "stage": "answer",
                        "embedding_ms": round(embedding_ms, 1),
                        "retrieval_ms": round(retrieval_ms, 1),
                        "generation_ms": 0.0,
                        "total_ms": round(total_ms, 1),
                        "retrieved": len(retrieved),
                        "kept": 0,
                        "reason": "excluded_by_seminar_filter",
                    },
                    ensure_ascii=True,
                )
            )
            return "The requested information is not available in the provided context."

        context_blocks: List[str] = [chunk for chunk, _ in filtered_retrieved]
        context_blocks = self._deduplicate_context_blocks(context_blocks)
        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a precise question-answering assistant.\n"
            "Use only the context. Do not invent facts.\n"
            "If the context includes a section list, return only those sections as bullet points.\n"
            "When a user asks which departments a faculty contains (\"hangi bölümleri içerir\"), "
            "answer with a concise bullet list of department names only.\n"
            "For department-list answers, end with a short note that the information comes from local data.\n"
            "Ignore announcements and seminars.\n"
            "When listing departments under the Faculty of Engineering and Natural Sciences, "
            "consider ONLY these: Bilgisayar Muhendisligi, Biyomedikal Muhendisligi, "
            "Molekuler Biyoloji ve Genetik (MBG).\n"
            "Never list seminar/event items or names from other faculties as if they were departments "
            "of this faculty.\n"
            "Do not repeat the same department with aliases (for example MBG and Molekuler Biyoloji); "
            "merge them into one canonical item.\n"
            "Do not treat Eczacilik Fakultesi as a department.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        print("DEBUG: answer: --- context (before Ollama) ---")
        print(context)
        print("DEBUG: answer: --- end context ---")
        print("DEBUG: answer: --- full prompt (before Ollama) ---")
        print(prompt)
        print("DEBUG: answer: --- end prompt ---")

        generation_start = perf_counter()
        response = self.llm.generate(prompt=prompt)
        generation_ms = (perf_counter() - generation_start) * 1000
        total_ms = (perf_counter() - total_start) * 1000
        print(
            json.dumps(
                {
                    "event": "latency",
                    "stage": "answer",
                    "embedding_ms": round(embedding_ms, 1),
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "retrieved": len(retrieved),
                    "kept": len(filtered_retrieved),
                    "reason": "ok",
                },
                ensure_ascii=True,
            )
        )
        if not response:
            print(
                "DEBUG: answer: LLM returned empty/whitespace; "
                "using not-available (see Ollama error logs above if any)."
            )
            return "The requested information is not available in the provided context."

        return self._postprocess_response(question, response)

    @staticmethod
    def _is_excluded_chunk(chunk: str) -> bool:
        lower_chunk = chunk.lower()
        return any(term in lower_chunk for term in EXCLUDED_CONTEXT_TERMS)

    @staticmethod
    def _is_engineering_departments_question(question: str) -> bool:
        normalized = question.lower()
        has_engineering = "muhendislik" in normalized or "mühendislik" in normalized
        has_department = (
            "bolum" in normalized
            or "bölüm" in normalized
            or "fakulte" in normalized
            or "fakülte" in normalized
        )
        return has_engineering and has_department

    @staticmethod
    def _is_engineering_faculty_departments_question(question: str) -> bool:
        if not RAGSystem._is_engineering_departments_question(question):
            return False
        normalized = question.lower()
        return ("doğa bilimleri" in normalized) or ("doga bilimleri" in normalized)

    def _load_engineering_departments_from_local_data(self) -> List[str]:
        data_file = Path(self.config.data_dir) / ENGINEERING_FACULTY_LOCAL_DATA_FILE
        if not data_file.exists():
            return []

        departments: List[str] = []
        for raw_line in data_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("- "):
                item = line[2:].strip()
                if item:
                    departments.append(item)

        # Keep unique order in case of accidental duplicate lines.
        unique_departments: List[str] = []
        seen: set[str] = set()
        for department in departments:
            key = department.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_departments.append(department)
        return unique_departments

    @staticmethod
    def _format_engineering_departments_answer(departments: List[str]) -> str:
        lines = ["Mühendislik ve Doğa Bilimleri Fakültesi şu bölümleri içerir:"]
        lines.extend(f"- {department}" for department in departments)
        lines.append("")
        lines.append("Not: Bu bilgi yerel veri dosyalarından derlenmiştir.")
        return "\n".join(lines)

    @staticmethod
    def _is_engineering_excluded_chunk(question: str, chunk: str) -> bool:
        if not RAGSystem._is_engineering_departments_question(question):
            return False
        lower_chunk = chunk.lower()
        return any(term in lower_chunk for term in ENGINEERING_CONTEXT_EXCLUDE_TERMS)

    @staticmethod
    def _apply_engineering_keyword_boost(
        question: str, retrieved: List[tuple[str, float]]
    ) -> List[tuple[str, float]]:
        if not RAGSystem._is_engineering_departments_question(question):
            return retrieved
        boosted: List[tuple[float, str, float]] = []
        for chunk, distance in retrieved:
            lower_chunk = chunk.lower()
            bonus = sum(1 for term in ENGINEERING_BOOST_TERMS if term in lower_chunk)
            adjusted_distance = distance - (0.08 * bonus)
            boosted.append((adjusted_distance, chunk, distance))
        boosted.sort(key=lambda item: item[0])
        return [(chunk, original_distance) for _, chunk, original_distance in boosted]

    @staticmethod
    def _deduplicate_context_blocks(context_blocks: List[str]) -> List[str]:
        deduplicated: List[str] = []
        seen_fingerprints: set[str] = set()
        for block in context_blocks:
            normalized_block = block
            for canonical_name, aliases in DEPARTMENT_ALIAS_GROUPS.items():
                if any(alias in normalized_block.lower() for alias in aliases):
                    normalized_block = RAGSystem._replace_aliases(
                        normalized_block, aliases, canonical_name
                    )
            fingerprint = re.sub(r"\s+", " ", normalized_block).strip().lower()
            if not fingerprint or fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            deduplicated.append(normalized_block)
        return deduplicated

    @staticmethod
    def _replace_aliases(text: str, aliases: tuple[str, ...], canonical_name: str) -> str:
        updated_text = text
        for alias in aliases:
            pattern = re.compile(re.escape(alias), flags=re.IGNORECASE)
            updated_text = pattern.sub(canonical_name, updated_text)
        return updated_text

    @staticmethod
    def _postprocess_response(question: str, response: str) -> str:
        if not RAGSystem._is_engineering_departments_question(question):
            return response
        lower = response.lower()
        has_mbg = "mbg" in lower
        has_molekuler = "moleküler biyoloji" in lower or "molekuler biyoloji" in lower
        if has_mbg and has_molekuler:
            response = re.sub(
                r"(?im)^\s*[-*]?\s*Molek[üu]ler Biyoloji(?: ve Genetik)?(?:\s*\(MBG\))?\s*$\n?",
                "",
                response,
            )
        return response.strip()
