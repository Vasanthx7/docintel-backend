"""Background ingestion: parse → chunk → embed → store.

Runs in the Celery worker (not the API) because parsing/embedding is slow.
"""
from .celery_app import celery_app
from .chunking import chunk_pages, parse_document
from .db import SessionLocal
from .embeddings import embed_texts
from .models import Chunk, Document
from .vectorstore import ensure_collection, upsert_chunks


@celery_app.task(name="app.tasks.ingest_document")
def ingest_document(document_id: str, path: str) -> dict:
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            return {"status": "missing"}

        pages = parse_document(path)
        chunks = chunk_pages(pages)
        if not chunks:
            doc.status = "failed"
            doc.error = "No extractable text found."
            db.commit()
            return {"status": "failed", "reason": "no text"}

        # Persist chunk metadata in Postgres.
        rows = [
            Chunk(
                document_id=document_id,
                page=ch.page,
                chunk_index=ch.chunk_index,
                text=ch.text,
            )
            for ch in chunks
        ]
        db.add_all(rows)
        db.flush()  # assign ids

        # Embed and upsert vectors into Qdrant (batched to bound memory).
        ensure_collection()
        batch = 32
        for i in range(0, len(rows), batch):
            group = rows[i : i + batch]
            vectors = embed_texts([r.text for r in group])
            points = [
                {
                    "id": r.id,
                    "dense": v["dense"],
                    "sparse": v["sparse"],
                    "payload": {
                        "document_id": document_id,
                        "page": r.page,
                        "text": r.text,
                    },
                }
                for r, v in zip(group, vectors)
            ]
            upsert_chunks(points)

        doc.num_pages = len(pages)
        doc.num_chunks = len(rows)
        doc.status = "ready"
        db.commit()
        return {"status": "ready", "pages": len(pages), "chunks": len(rows)}
    except Exception as exc:  # noqa: BLE001 — record failure for the UI
        db.rollback()
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.status = "failed"
            doc.error = str(exc)[:1000]
            db.commit()
        raise
    finally:
        db.close()
