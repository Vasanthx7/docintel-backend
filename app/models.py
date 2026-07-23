import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    # processing | ready | failed
    status: Mapped[str] = mapped_column(String(32), default="processing")
    num_pages: Mapped[int] = mapped_column(Integer, default=0)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    # Same id is used as the Qdrant point id.
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


# A conversation can be pinned to a set of documents ("docs related to it").
# If a conversation has no pinned documents, retrieval searches everything.
conversation_documents = Table(
    "conversation_documents",
    Base.metadata,
    Column(
        "conversation_id",
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "document_id",
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(512), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    documents: Mapped[list[Document]] = relationship(secondary=conversation_documents)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    # user | assistant
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    # Citations attached to an assistant answer (list of citation dicts); null for user turns.
    # none_as_null=True so a Python None stores as SQL NULL, not a JSON `null` literal.
    citations: Mapped[list | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
