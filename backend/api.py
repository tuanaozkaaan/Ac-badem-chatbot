from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model.local_llm import LocalLLM
from rag.pipeline import RAGConfig, RAGSystem


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


app = FastAPI(title="Acibadem RAG API", version="0.1.0")

_rag_system: RAGSystem | None = None


def get_rag() -> RAGSystem:
    global _rag_system
    if _rag_system is None:
        raise RuntimeError("RAG system is not initialized. Start with a valid model path.")
    return _rag_system


def init_rag(model_path: str | None = None) -> None:
    global _rag_system
    llm = LocalLLM(model_path=model_path)
    rag = RAGSystem(llm=llm, config=RAGConfig())
    rag.build_knowledge_base()
    _rag_system = rag


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    rag = get_rag()
    answer = rag.answer(question)
    return AskResponse(answer=answer)
