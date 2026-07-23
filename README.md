# DocIntel — Backend

FastAPI backend for **DocIntel**, a local RAG knowledge assistant: upload PDFs,
ask questions, get **streamed answers with page citations** — running **fully
locally**. No per-token API fees, no data leaving your machine.

Frontend lives in a separate repo: **[docintel-frontend](https://github.com/Vasanthx7/docintel-frontend)**.

## Retrieval pipeline

```
question → BGE-M3 (dense + sparse) → Qdrant hybrid search (top-20)
        → bge-reranker-v2-m3 (rerank → top-5) → Qwen 2.5 (grounded, cited) → stream
```

## Stack

| Layer | Tech |
|-------|------|
| LLM | Ollama + `qwen2.5:7b-instruct` (OpenAI-compatible API) |
| Embeddings | BGE-M3 (dense + sparse) via FlagEmbedding |
| Reranker | bge-reranker-v2-m3 (cross-encoder) |
| Vector DB | Qdrant (native hybrid search) |
| Parsing | Docling (layout-aware) + PyMuPDF fallback |
| API | FastAPI + SSE streaming |
| Ingestion | Celery + Redis |
| Metadata | Postgres (SQLAlchemy) |

## Prerequisites

- Docker + Docker Compose
- [Ollama](https://ollama.com) installed on the host

## Quick start

```bash
# 1. Pull the model (Ollama runs on the host; the app reaches it via
#    host.docker.internal). Make sure `ollama serve` is running.
ollama pull qwen2.5:7b-instruct

# 2. Config
cp .env.example .env

# 3. Build + run the full stack (postgres, redis, qdrant, api, worker)
docker compose up --build
```

First boot is slow: the image installs torch + downloads BGE-M3 and the reranker
weights on first use (cached in the `models` volume afterwards).

Then open:

- API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

## API

```
POST   /documents               # upload PDF (async: processing → ready)
GET    /documents               # list
DELETE /documents/{id}

POST   /conversations           {title?, document_ids?}   # optionally pin docs
GET    /conversations                                     # list (newest first)
GET    /conversations/{id}                                # thread + messages
DELETE /conversations/{id}

POST   /query   {question, conversation_id?, document_id?}   # SSE streaming RAG
```

### Conversations & history

Queries can be stateless (single-shot) or run inside a **conversation** that
persists history and enables real multi-turn follow-ups.

- Pass `conversation_id` to `/query` and the turn is saved (user + assistant
  messages, with the assistant's citations) and prior turns are fed back in.
- **Follow-ups are condensed:** before retrieval, a follow-up like *"summarize
  it"* is rewritten into a standalone question using the history, so retrieval
  actually works. (Skipped when there's no history — no extra LLM call.)
- **Doc scoping:** pin `document_ids` on a conversation to restrict retrieval to
  those docs. Precedence per query: explicit `document_id` → the conversation's
  pinned docs → the whole collection.
- Omit `conversation_id` for a one-shot, stateless query.

## Local dev without Docker

```bash
# infra
docker compose up postgres redis qdrant

# backend (terminal 1)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# point env at localhost instead of docker service names:
export LLM_BASE_URL=http://localhost:11434/v1
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/docintel
export REDIS_URL=redis://localhost:6379/0
export QDRANT_URL=http://localhost:6333
uvicorn app.main:app --reload

# worker (terminal 2, same venv + env)
celery -A app.celery_app.celery_app worker --loglevel=info --concurrency=1
```

## Swap local ↔ cloud

The LLM is behind an OpenAI-compatible client, so switching to a cloud provider
is env-only:

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

## Layout

```
backend/
├── docker-compose.yml  # postgres, redis, qdrant, api, worker
├── Dockerfile
├── .env.example
├── requirements.txt
└── app/
    ├── main.py         # FastAPI app + startup
    ├── config.py       # env-driven settings
    ├── db.py / models.py / schemas.py   # + Conversation / Message models
    ├── chunking.py     # Docling parse + chunk
    ├── embeddings.py   # BGE-M3 (dense + sparse)
    ├── reranker.py     # bge-reranker-v2-m3
    ├── vectorstore.py  # Qdrant hybrid search
    ├── llm.py          # Ollama streaming + grounding + follow-up condense
    ├── celery_app.py / tasks.py   # async ingestion
    └── routers/
        ├── documents.py       # upload / list / delete
        ├── conversations.py   # chat threads + history
        └── query.py           # SSE streaming RAG (+ multi-turn)
```
