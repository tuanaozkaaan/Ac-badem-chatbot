from __future__ import annotations

"""
RAG flow: retrieval here; LLM calls go through model.local_llm.LocalLLM.
Ollama URL is OLLAMA_BASE_URL (e.g. http://ollama:11434 in Docker — Compose service name `ollama`, not the container name).
"""
from dataclasses import dataclass
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


@dataclass
class RAGConfig:
    data_dir: str = "data"
    prefer_db_chunks: bool = True
    # Must match create_embeddings and ChunkEmbedding defaults (EXPECTED_EMBEDDING_MODEL).
    embedding_model_name: str = EXPECTED_EMBEDDING_MODEL
    top_k: int = 10
    # IndexFlatL2 / embed_query: squared L2 on normalized vectors. Larger = worse match.
    # Reject when best match is worse than this (strict inequality: > threshold => no context).
    max_distance_threshold: float = 2.0


class RAGSystem:
    def __init__(self, llm: LocalLLM, config: RAGConfig | None = None) -> None:
        self.llm = llm
        self.config = config or RAGConfig()
        self.store: VectorStore | None = None

    def build_knowledge_base(self) -> None:
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

    def answer(self, question: str) -> str:
        if not self.store:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")

        query_vector = embed_query(question, self.store.embedding_model_name)
        retrieved = search_top_k(self.store, query_vector, k=self.config.top_k)

        if not retrieved:
            print(
                "DEBUG: answer: search_top_k empty (index empty?); "
                f"index ntotal={getattr(self.store.index, 'ntotal', 'n/a')}"
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
            return "The requested information is not available in the provided context."

        context_blocks: List[str] = [chunk for chunk, _ in retrieved]
        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a question-answering assistant.\n"
            "Answer using only the information in the context below. Do not invent facts that are "
            "not supported by the context.\n"
            "If the context is only partly relevant, still try to give a useful answer from what is "
            "there, and state clearly when something is not specified in the context.\n\n"
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

        response = self.llm.generate(prompt=prompt)
        if not response:
            print(
                "DEBUG: answer: LLM returned empty/whitespace; "
                "using not-available (see Ollama error logs above if any)."
            )
            return "The requested information is not available in the provided context."

        return response
