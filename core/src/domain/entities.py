from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Pipeline(str, Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    STATISTICAL = "statistical"
    COMPARE = "compare"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(slots=True)
class User:
    id: int | None
    email: str
    password_hash: str
    name: str | None = None
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime | None = None


@dataclass(slots=True)
class PasswordResetToken:
    id: int | None
    user_id: int
    token_hash: str
    expires_at: datetime
    used_at: datetime | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class Document:
    dataset: str
    doc_id: str
    title: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_vector: list[float] | None = None
    quantum_vector: list[float] | None = None
    statistical_vector: list[float] | None = None


@dataclass(slots=True)
class Query:
    query_id: str | None
    query_text: str
    dataset: str | None = None
    user_id: int | None = None


@dataclass(slots=True)
class SearchResult:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    query_id: str | None
    query_text: str
    pipeline: str
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    top_k_doc_ids: list[str]


@dataclass(slots=True)
class GroundTruth:
    query_id: str
    query_text: str
    relevant_doc_ids: list[str]
    dataset: str
    user_id: int | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class DatasetSnapshot:
    dataset_id: str
    name: str
    provider: str
    description: str
    source_url: str | None = None
    reference_urls: list[str] = field(default_factory=list)
    max_docs: int | None = None
    max_queries: int | None = None
    document_count: int = 0
    query_count: int = 0
    document_ids: list[str] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class Chat:
    id: int | None
    user_id: int
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None


@dataclass(slots=True)
class ChatMessage:
    id: int | None
    chat_id: int
    role: MessageRole
    content: str
    created_at: datetime | None = None
