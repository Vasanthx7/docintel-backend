"""Query endpoint — the full retrieval pipeline, streamed over SSE.

  question
    → (condense follow-up using chat history, if a conversation is given)
    → BGE-M3 (dense + sparse)
    → Qdrant hybrid search (top-20, scoped to the conversation's pinned docs)
    → bge-reranker rerank (top-5)
    → Qwen 2.5 (grounded, cited)  ── streamed token by token

When ``conversation_id`` is supplied the turn is persisted (user + assistant
messages) and prior turns feed both the condense step and the answer prompt.

SSE event protocol (each line: `data: {json}\n\n`):
  {"type": "sources", "citations": [...]}   first
  {"type": "token", "text": "..."}          many
  {"type": "done", "conversation_id": "..."} last
"""
import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..config import settings
from ..db import SessionLocal
from ..embeddings import embed_query
from ..llm import condense_question, stream_answer
from ..models import Conversation, Message
from ..reranker import rerank
from ..schemas import QueryIn
from ..vectorstore import hybrid_search

router = APIRouter(tags=["query"])

# How many prior messages to feed the condense + answer steps. Kept small so
# the prompt stays cheap for a local 7B model.
HISTORY_LIMIT = 6


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _run(body: QueryIn) -> Iterator[str]:
    db = SessionLocal()
    try:
        conv: Conversation | None = None
        history: list[dict] = []
        pinned_ids: list[str] = []

        if body.conversation_id:
            conv = db.get(Conversation, body.conversation_id)
            if conv is None:
                # Surface as an SSE error rather than a mid-stream crash.
                yield _sse({"type": "error", "message": "Conversation not found"})
                return
            history = [
                {"role": m.role, "content": m.content}
                for m in conv.messages[-HISTORY_LIMIT:]
            ]
            pinned_ids = [d.id for d in conv.documents]

        # Retrieval scope: explicit document_id wins, else the conversation's
        # pinned docs, else the whole collection.
        search_ids = [body.document_id] if body.document_id else pinned_ids

        # Rewrite a follow-up into a standalone query before embedding.
        search_question = condense_question(body.question, history)

        q = embed_query(search_question)
        candidates = hybrid_search(
            dense=q["dense"],
            sparse=q["sparse"],
            limit=settings.retrieve_top_k,
            document_ids=search_ids or None,
        )

        citations: list[dict] = []
        answer_parts: list[str] = []

        if not candidates:
            citations = []
            yield _sse({"type": "sources", "citations": citations})
            msg = "I don't have any documents that cover this yet."
            answer_parts.append(msg)
            yield _sse({"type": "token", "text": msg})
        else:
            top = rerank(body.question, candidates, settings.rerank_top_k)
            citations = [
                {
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "page": c["page"],
                    "score": round(c.get("rerank_score", c["score"]), 4),
                    "snippet": c["text"][:280],
                }
                for c in top
            ]
            yield _sse({"type": "sources", "citations": citations})

            for token in stream_answer(body.question, top, history):
                answer_parts.append(token)
                yield _sse({"type": "token", "text": token})

        # Persist the turn once the answer is complete.
        if conv is not None:
            answer = "".join(answer_parts)
            if not conv.messages:
                conv.title = body.question[:80]
            db.add(Message(conversation_id=conv.id, role="user", content=body.question))
            db.add(
                Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=answer,
                    citations=citations or None,
                )
            )
            db.commit()

        yield _sse(
            {"type": "done", "conversation_id": conv.id if conv else None}
        )
    finally:
        db.close()


@router.post("/query")
def query(body: QueryIn):
    return StreamingResponse(
        _run(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
