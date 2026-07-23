from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import Base, engine
from .routers import conversations, documents, query
from .vectorstore import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables + Qdrant collection on startup. For a real project,
    # swap table creation for Alembic migrations.
    Base.metadata.create_all(bind=engine)
    try:
        ensure_collection()
    except Exception:
        # Qdrant may not be up yet on first boot; the ingest task also ensures it.
        pass

    # Warm the heavy models at startup so the first user request isn't slow.
    from .device import device_label
    from .embeddings import _model
    from .reranker import _reranker

    print(f"[startup] loading embed + rerank models on {device_label()}...")
    _model()
    _reranker()
    print("[startup] models ready")
    yield


app = FastAPI(title="DocIntel", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(conversations.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}
