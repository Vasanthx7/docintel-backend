from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    num_pages: int
    num_chunks: int
    error: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class QueryIn(BaseModel):
    question: str
    document_id: str | None = None  # optional: scope to one document
    conversation_id: str | None = None  # optional: persist + use chat history


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    page: int
    score: float
    snippet: str


class ConversationCreate(BaseModel):
    title: str | None = None
    # Optionally pin the conversation to a set of documents ("docs related to it").
    document_ids: list[str] = []


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    document_ids: list[str] = []

    class Config:
        from_attributes = True


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []
