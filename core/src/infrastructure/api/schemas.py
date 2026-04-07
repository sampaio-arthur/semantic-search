from __future__ import annotations

from pydantic import BaseModel


class UserOut(BaseModel):
    id: int
    email: str
    created_at: str | None = None


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: str | None = None


class SignUpRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ChatCreateRequest(BaseModel):
    title: str | None = None


class ChatRenameRequest(BaseModel):
    title: str


class ChatMessageCreateRequest(BaseModel):
    role: str
    content: str


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str | None = None


class ChatOut(BaseModel):
    id: int
    title: str
    created_at: str | None = None


class ChatDetailOut(ChatOut):
    messages: list[ChatMessageOut]


class IndexRequest(BaseModel):
    dataset_id: str = "beir/trec-covid"
    force_reindex: bool = False


class SearchRequest(BaseModel):
    dataset_id: str = "beir/trec-covid"
    query: str
    query_id: str | None = None
    mode: str = "compare"
    top_k: int = 25
    chat_id: int | None = None


class GroundTruthUpsertRequest(BaseModel):
    dataset_id: str = "beir/trec-covid"
    query_id: str
    query_text: str
    relevant_doc_ids: list[str]


class EvaluateRequest(BaseModel):
    dataset_id: str = "beir/trec-covid"
    pipeline: str = "compare"
    k: int = 25


class BatchEvaluateRequest(BaseModel):
    dataset_id: str = "beir/trec-covid"
    pipelines: list[str] = ["classical", "quantum", "statistical"]
    k: int = 25


class BenchmarkLabelInput(BaseModel):
    dataset_id: str
    query_text: str
    relevant_doc_ids: list[str] | None = None


class FileSearchForm(BaseModel):
    query: str
    mode: str = "compare"
