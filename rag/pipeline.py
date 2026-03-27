from dataclasses import dataclass
from typing import List

from model.local_llm import LocalLLM
from rag.document_loader import load_text_documents
from rag.embedding_store import VectorStore, build_faiss_index, embed_query, search_top_k
from rag.text_splitter import split_into_chunks


@dataclass
class RAGConfig:
    data_dir: str = "data"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3
    max_distance_threshold: float = 1.2


class RAGSystem:
    def __init__(self, llm: LocalLLM, config: RAGConfig | None = None):
        self.llm = llm
        self.config = config or RAGConfig()
        self.store: VectorStore | None = None

    def build_knowledge_base(self) -> None:
        docs = load_text_documents(self.config.data_dir)
        chunks = split_into_chunks(docs)
        self.store = build_faiss_index(
            chunks=chunks, embedding_model_name=self.config.embedding_model_name
        )

    def answer(self, question: str) -> str:
        if not self.store:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")

        query_vector = embed_query(question, self.store.embedding_model_name)
        retrieved = search_top_k(self.store, query_vector, k=self.config.top_k)

        if not retrieved:
            return "The requested information is not available in the provided context."

        best_distance = retrieved[0][1]
        if best_distance > self.config.max_distance_threshold:
            return "The requested information is not available in the provided context."

        context_blocks: List[str] = [chunk for chunk, _ in retrieved]
        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a question-answering assistant.\n"
            "Answer the question using only the context below.\n"
            "If the answer is not clearly in the context, respond exactly with:\n"
            '"The requested information is not available in the provided context."\n\n'
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        response = self.llm.generate(prompt=prompt)
        if not response:
            return "The requested information is not available in the provided context."

        return response
