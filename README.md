# Minimal Local RAG Prototype (Acibadem University)

This project is a minimal working prototype of a local Retrieval-Augmented Generation (RAG) system.
It answers questions about Acibadem University using local text files and a local open-source LLM.

## Project Structure

```text
.
├── backend/
│   ├── __init__.py
│   ├── api.py
│   └── run_api.py
├── data/
│   ├── acibadem_overview.txt
│   ├── acibadem_facilities.txt
│   └── acibadem_admissions.txt
├── frontend/
│   └── index.html
├── model/
│   ├── __init__.py
│   └── local_llm.py
├── rag/
│   ├── __init__.py
│   ├── embedding_store.py
│   ├── document_loader.py
│   ├── pipeline.py
│   └── text_splitter.py
├── main.py
├── requirements.txt
└── README.md
```

## What It Does

1. Loads local `.txt` files from `data/`
2. Splits documents into chunks
3. Creates embeddings with Sentence Transformers
4. Stores vectors in FAISS
5. Retrieves top relevant chunks for a question
6. Uses a local GGUF LLM (via `llama-cpp-python`) to generate an answer only from retrieved context
7. Returns a fallback message when the information is not available

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download a local GGUF model (example: TinyLlama or Mistral instruct GGUF) and note its path.

## Run

### CLI (existing)

Single question:

```bash
python main.py --model-path /path/to/your/model.gguf --question "Where is Acibadem University located?"
```

Interactive mode:

```bash
python main.py --model-path /path/to/your/model.gguf
```

Type `exit` to stop interactive mode.

### Backend API + Frontend UI

1. Start backend API:

```bash
python -m backend.run_api --model-path /path/to/your/model.gguf --host 127.0.0.1 --port 8000
```

2. In a second terminal, serve frontend:

```bash
python -m http.server 5500 --directory frontend
```

3. Open:

- `http://127.0.0.1:5500`

4. Ask questions from the web UI (it calls `POST /ask` on port `8000`).

## Notes

- This is a demo-first baseline intended for later expansion to Django + PostgreSQL + Docker.
- You can add more local files into `data/` to improve answer quality.
- The fallback sentence is:
  `The requested information is not available in the provided context.`
