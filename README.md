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

## PostgreSQL (Local Dev)

This project uses **PostgreSQL** (not SQLite). Database config is read from environment variables:

- `POSTGRES_HOST` (default: `localhost`)
- `POSTGRES_PORT` (default: `5432`)
- `POSTGRES_DB` (default: `acibadem`)
- `POSTGRES_USER` (default: `acibadem`)
- `POSTGRES_PASSWORD` (default: `acibadem`)

If you see:

`OperationalError: ... FATAL: role "acibadem" does not exist`

it means the configured Postgres user/role is missing in the Postgres instance you are connecting to.

### Option A (recommended): use Docker Postgres from `docker-compose.yml`

The `db` service is exposed on host port **5433** by default.

```bash
docker compose up -d db
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_DB=acibadem
export POSTGRES_USER=acibadem
export POSTGRES_PASSWORD=acibadem
python3 manage.py migrate
```

### Option B: use your local Postgres (port 5432)

Create the role + database (adjust password if you want):

```bash
psql postgres -c "CREATE ROLE acibadem WITH LOGIN PASSWORD 'acibadem' CREATEDB;"
psql postgres -c "CREATE DATABASE acibadem OWNER acibadem;"
```

Then run migrations:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=acibadem
export POSTGRES_USER=acibadem
export POSTGRES_PASSWORD=acibadem
python3 manage.py migrate
```

### Quick test

```bash
python3 manage.py shell
```

```python
from chatbot.models import ScrapedPage, PageChunk
ScrapedPage.objects.count(), PageChunk.objects.count()
```

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

## Responsible Data Ingestion (New)

This project now includes a production-oriented ingestion pipeline for Acibadem public pages:

- Respects `robots.txt` for each domain
- Crawls only allowed public pages under:
  - `https://www.acibadem.edu.tr`
  - `https://obs.acibadem.edu.tr`
- Uses request delays (`1-2` seconds by default), max page cap, visited URL tracking, and duplicate prevention
- Cleans noisy HTML (nav/header/footer/boilerplate) and stores normalized content in PostgreSQL
- Supports optional Playwright fallback for JS-heavy OBS pages

Run ingestion:

```bash
python manage.py ingest_acibadem --max-pages 150 --min-delay 1 --max-delay 2 --log-level INFO
```

Optional Playwright fallback:

```bash
python manage.py ingest_acibadem --enable-playwright-obs
```

Docker run:

```bash
docker compose exec web python manage.py ingest_acibadem --max-pages 150
```
