from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from pgvector.sqlalchemy import Vector as PgVector


class Base(DeclarativeBase):
    pass


def json_type_for(dialect_name: str = ""):
    return JSONB if dialect_name == "postgresql" else JSON


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chats = relationship("ChatModel", back_populates="user")


class PasswordResetModel(Base):
    __tablename__ = "password_resets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(255), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatModel(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    user = relationship("UserModel", back_populates="chats")
    messages = relationship("ChatMessageModel", back_populates="chat", cascade="all, delete-orphan")


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chat = relationship("ChatModel", back_populates="messages")


class DocumentModel(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("dataset", "doc_id", name="uq_documents_dataset_doc_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(100), index=True)
    doc_id: Mapped[str] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    embedding_vector: Mapped[list[float] | None] = mapped_column(PgVector(64), nullable=True)
    quantum_vector: Mapped[list[float] | None] = mapped_column(PgVector(64), nullable=True)
    statistical_vector: Mapped[list[float] | None] = mapped_column(PgVector(64), nullable=True)


class QueryModel(Base):
    __tablename__ = "queries"
    __table_args__ = (UniqueConstraint("dataset", "split", "query_id", name="uq_queries_dataset_split_query_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(100), index=True)
    split: Mapped[str] = mapped_column(String(32), default="test")
    query_id: Mapped[str] = mapped_column(String(255))
    query_text: Mapped[str] = mapped_column(Text)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class QrelModel(Base):
    __tablename__ = "qrels"
    __table_args__ = (UniqueConstraint("dataset", "split", "query_id", "doc_id", name="uq_qrels_dataset_split_query_doc"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(100), index=True)
    split: Mapped[str] = mapped_column(String(32), default="test")
    query_id: Mapped[str] = mapped_column(String(255), index=True)
    doc_id: Mapped[str] = mapped_column(String(255))
    relevance: Mapped[int] = mapped_column(Integer, default=0)


class DatasetSnapshotModel(Base):
    __tablename__ = "dataset_snapshots"
    __table_args__ = (UniqueConstraint("dataset_id", name="uq_dataset_snapshots_dataset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_urls: Mapped[list[str]] = mapped_column(JSON, default=list)
    max_docs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_queries: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    query_count: Mapped[int] = mapped_column(Integer, default=0)
    document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    queries_json: Mapped[list[dict]] = mapped_column("queries", JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
