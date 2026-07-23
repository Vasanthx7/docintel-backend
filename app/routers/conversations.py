"""Conversation history — create/list/read/delete chat threads.

A conversation owns an ordered list of messages and, optionally, a set of
pinned documents that scope retrieval for every turn in that thread.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Conversation, Document
from ..schemas import ConversationCreate, ConversationDetail, ConversationOut

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _to_out(conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "document_ids": [d.id for d in conv.documents],
    }


@router.post("", response_model=ConversationOut)
def create_conversation(body: ConversationCreate, db: Session = Depends(get_db)):
    conv = Conversation(title=body.title or "New chat")
    if body.document_ids:
        docs = db.scalars(
            select(Document).where(Document.id.in_(body.document_ids))
        ).all()
        found = {d.id for d in docs}
        missing = set(body.document_ids) - found
        if missing:
            raise HTTPException(404, f"Unknown document(s): {', '.join(sorted(missing))}")
        conv.documents = list(docs)
    db.add(conv)
    db.commit()
    return _to_out(conv)


@router.get("", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    convs = db.scalars(
        select(Conversation).order_by(Conversation.updated_at.desc())
    ).all()
    return [_to_out(c) for c in convs]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    out = _to_out(conv)
    out["messages"] = conv.messages
    return out


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    db.delete(conv)
    db.commit()
    return {"deleted": conversation_id}
