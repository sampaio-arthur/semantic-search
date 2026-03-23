from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordRequestForm

from audit import audit_print, category_log, preview_text
from domain.entities import Document, SearchResult
from domain.exceptions import ConflictError, DomainError, NotFoundError, UnauthorizedError, ValidationError
from domain.ir import FIXED_TOP_K
from infrastructure.api.deps import Services, get_current_user_id, get_services
from infrastructure.api.evaluation_jobs import evaluation_job_registry
from infrastructure.api.index_jobs import index_job_registry
from infrastructure.api.schemas import (
    BatchEvaluateRequest,
    BenchmarkLabelInput,
    ChatCreateRequest,
    ChatDetailOut,
    ChatMessageCreateRequest,
    ChatMessageOut,
    ChatOut,
    ChatRenameRequest,
    EvaluateRequest,
    ForgotPasswordRequest,
    GroundTruthUpsertRequest,
    IndexRequest,
    RefreshTokenRequest,
    ResetPasswordRequest,
    SearchRequest,
    SignUpRequest,
    TokenOut,
    UserOut,
)
from infrastructure.metrics.ir_measures_adapter import IrMeasuresAdapter


router = APIRouter(prefix="/api")
compat_router = APIRouter()
_metrics_adapter = IrMeasuresAdapter()


def _serialize_dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _user_out(user) -> UserOut:
    return UserOut(id=user.id, email=user.email, created_at=_serialize_dt(user.created_at))


def _chat_out(chat) -> ChatOut:
    return ChatOut(id=chat.id, title=chat.title, created_at=_serialize_dt(chat.created_at))


def _msg_out(msg) -> ChatMessageOut:
    return ChatMessageOut(id=msg.id, role=msg.role.value if hasattr(msg.role, "value") else str(msg.role), content=msg.content, created_at=_serialize_dt(msg.created_at))


def _serialize_results(results: list[SearchResult]) -> list[dict]:
    return [{"doc_id": r.doc_id, "text": r.text, "score": r.score} for r in results]


def _handle_domain_error(exc: DomainError) -> HTTPException:
    if isinstance(exc, ConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, UnauthorizedError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal error")


def _find_ground_truth(services: Services, dataset_id: str, query_id: str | None, query_text: str):
    if query_id:
        item = services.ground_truths.get(dataset_id, query_id)
        if item:
            return item
    normalized = (query_text or "").strip()
    if not normalized:
        return None
    return services.ground_truths.get_by_query_text(dataset_id, normalized)


def _enrich_metrics(existing: dict | None, evaluated, k: int) -> dict:
    merged = dict(existing or {})
    merged["precision_at_k"] = evaluated.precision_at_k
    merged["recall_at_k"] = evaluated.recall_at_k
    merged["ndcg_at_k"] = evaluated.ndcg_at_k
    merged["mrr"] = evaluated.mrr
    merged["k"] = k
    merged["has_labels"] = True
    merged["ground_truth_query_id"] = evaluated.query_id
    return merged


def _attach_ir_metrics(
    output: dict[str, Any],
    services: Services,
    dataset_id: str,
    query_id: str | None,
    query_text: str,
    k: int,
) -> None:
    gt = _find_ground_truth(services, dataset_id, query_id, query_text)
    if not gt:
        return
    if "comparison" in output:
        for pipeline_key in ("classical", "quantum", "statistical"):
            pipeline_data = output["comparison"].get(pipeline_key)
            if not pipeline_data:
                continue
            pipeline_results = pipeline_data.get("results") or []
            pipeline_eval = _metrics_adapter.evaluate_query(
                query_id=gt.query_id,
                query_text=gt.query_text,
                pipeline=pipeline_key,
                retrieved_doc_ids=[item["doc_id"] for item in pipeline_results],
                retrieved_scores=[float(item["score"]) for item in pipeline_results],
                relevant_doc_ids=gt.relevant_doc_ids,
                k=k,
            )
            output["comparison"][pipeline_key]["metrics"] = _enrich_metrics(
                pipeline_data.get("metrics"), pipeline_eval, k
            )
        _attach_answer_similarity(output, gt, services)
        return
    if output.get("mode") in {"classical", "quantum", "statistical"}:
        results = output.get("results") or []
        eval_result = _metrics_adapter.evaluate_query(
            query_id=gt.query_id,
            query_text=gt.query_text,
            pipeline=output["mode"],
            retrieved_doc_ids=[item["doc_id"] for item in results],
            retrieved_scores=[float(item["score"]) for item in results],
            relevant_doc_ids=gt.relevant_doc_ids,
            k=k,
        )
        output["metrics"] = _enrich_metrics(output.get("metrics"), eval_result, k)
        _attach_answer_similarity(output, gt, services)


def _attach_answer_similarity(output: dict[str, Any], gt, services: Services) -> None:
    if not gt.ideal_answer:
        return
    if "comparison" in output:
        for pipeline_key in ("classical", "quantum", "statistical"):
            pipeline_data = output["comparison"].get(pipeline_key)
            if not pipeline_data:
                continue
            results = pipeline_data.get("results") or []
            answer_text = " ".join(r["text"] for r in results[:3])
            similarity = services.answer_similarity.compute(answer_text, gt.ideal_answer)
            metrics = pipeline_data.get("metrics") or {}
            metrics["answer_similarity"] = round(similarity, 4)
            metrics["has_ideal_answer"] = True
            pipeline_data["metrics"] = metrics
            category_log(
                "SEMANTIC EVAL",
                _extra={"query_id": gt.query_id, "pipeline": pipeline_key, "similarity": round(similarity, 4)},
            )
        return
    if output.get("mode") in {"classical", "quantum", "statistical"}:
        results = output.get("results") or []
        answer_text = " ".join(r["text"] for r in results[:3])
        similarity = services.answer_similarity.compute(answer_text, gt.ideal_answer)
        metrics = output.get("metrics") or {}
        metrics["answer_similarity"] = round(similarity, 4)
        metrics["has_ideal_answer"] = True
        output["metrics"] = metrics
        category_log(
            "SEMANTIC EVAL",
            _extra={"query_id": gt.query_id, "pipeline": output["mode"], "similarity": round(similarity, 4)},
        )


@router.get("/health")
def api_health() -> dict:
    return {"status": "ok"}


@compat_router.get("/health")
def compat_health() -> dict:
    return api_health()


@router.post("/auth/signup", response_model=UserOut, status_code=201)
def sign_up(payload: SignUpRequest, services: Services = Depends(get_services)):
    try:
        return _user_out(services.sign_up.execute(payload.email, payload.password, payload.name))
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.post("/auth/register", response_model=UserOut, status_code=201)
def register_compat(payload: SignUpRequest, services: Services = Depends(get_services)):
    return sign_up(payload, services)


@router.post("/auth/login", response_model=TokenOut)
def login_api(form: OAuth2PasswordRequestForm = Depends(), services: Services = Depends(get_services)):
    try:
        return services.sign_in.execute(form.username, form.password)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.post("/auth/login", response_model=TokenOut)
def login_compat(form: OAuth2PasswordRequestForm = Depends(), services: Services = Depends(get_services)):
    return login_api(form, services)


@router.get("/auth/me", response_model=UserOut)
@compat_router.get("/auth/me", response_model=UserOut)
def me(user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    user = services.users.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_out(user)


@router.post("/auth/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, services: Services = Depends(get_services)):
    services.request_reset.execute(payload.email)
    return {"detail": "If the email exists, a reset token has been generated."}


@router.post("/auth/reset-password")
def reset_password(payload: ResetPasswordRequest, services: Services = Depends(get_services)):
    try:
        services.confirm_reset.execute(payload.token, payload.new_password)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc
    return {"detail": "Password updated."}


@router.post("/auth/refresh", response_model=TokenOut)
def refresh_token(payload: RefreshTokenRequest, services: Services = Depends(get_services)):
    try:
        return services.refresh.execute(payload.refresh_token)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@router.post("/chats", response_model=ChatOut, status_code=201)
def create_chat(payload: ChatCreateRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        return _chat_out(services.create_chat.execute(user_id, payload.title))
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.post("/conversations", response_model=ChatOut, status_code=201)
def compat_create_conversation(payload: ChatCreateRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    return create_chat(payload, user_id, services)


@router.get("/chats", response_model=list[ChatOut])
def list_chats(
    page: int = 1,
    page_size: int = 20,
    user_id: int = Depends(get_current_user_id),
    services: Services = Depends(get_services),
):
    return [_chat_out(x) for x in services.list_chats.execute(user_id, page, page_size)]


@compat_router.get("/conversations", response_model=list[ChatOut])
def compat_list_conversations(user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    return list_chats(1, 50, user_id, services)


@router.get("/chats/{chat_id}", response_model=ChatDetailOut)
def get_chat(chat_id: int, page: int = 1, page_size: int = 200, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        chat, messages = services.get_chat.execute(user_id, chat_id, page, page_size)
        return ChatDetailOut(id=chat.id or 0, title=chat.title, created_at=_serialize_dt(chat.created_at), messages=[_msg_out(m) for m in messages])
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.get("/conversations/{chat_id}", response_model=ChatDetailOut)
def compat_get_conversation(chat_id: int, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    return get_chat(chat_id, 1, 500, user_id, services)


@router.post("/chats/{chat_id}/messages", response_model=ChatMessageOut, status_code=201)
def add_message(chat_id: int, payload: ChatMessageCreateRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        msg = services.add_message.execute(user_id, chat_id, payload.role, payload.content)
        return _msg_out(msg)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.post("/conversations/{chat_id}/messages", response_model=ChatMessageOut, status_code=201)
def compat_add_message(chat_id: int, payload: ChatMessageCreateRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    return add_message(chat_id, payload, user_id, services)


@router.patch("/chats/{chat_id}", response_model=ChatOut)
def rename_chat(chat_id: int, payload: ChatRenameRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        return _chat_out(services.rename_chat.execute(user_id, chat_id, payload.title))
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.patch("/conversations/{chat_id}", response_model=ChatOut)
def compat_rename_conversation(chat_id: int, payload: ChatRenameRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    return rename_chat(chat_id, payload, user_id, services)


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        services.delete_chat.execute(user_id, chat_id)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.delete("/conversations/{chat_id}", status_code=204)
def compat_delete_conversation(chat_id: int, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    return delete_chat(chat_id, user_id, services)


@router.post("/index")
def index_dataset(payload: IndexRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    user = services.users.get_by_id(user_id)
    if services.settings.require_admin_for_indexing and not (user and user.is_admin):
        raise HTTPException(status_code=403, detail="Admin required")
    try:
        audit_print("api.index.sync.request", user_id=user_id, dataset_id=payload.dataset_id, force_reindex=payload.force_reindex)
        return services.index_dataset.execute(payload.dataset_id)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.post("/search/dataset/index")
def compat_index_dataset(payload: IndexRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    user = services.users.get_by_id(user_id)
    if services.settings.require_admin_for_indexing and not (user and user.is_admin):
        raise HTTPException(status_code=403, detail="Admin required")
    audit_print("api.index.async.request", user_id=user_id, dataset_id=payload.dataset_id, force_reindex=payload.force_reindex)
    state = index_job_registry.start(payload.dataset_id, force_reindex=payload.force_reindex)
    return {"accepted": True, **state.to_dict()}


@compat_router.get("/search/dataset/index/status")
def compat_index_dataset_status(dataset_id: str, services: Services = Depends(get_services)):
    audit_print("api.index.status.request", dataset_id=dataset_id)
    state = index_job_registry.get(dataset_id)
    if state.status == "idle":
        snapshot = services.dataset_snapshots.get(dataset_id)
        docs_count = services.documents.count_by_dataset(dataset_id)
        encoders_fitted = (
            services.index_dataset.classical_encoder.is_fitted
            and services.index_dataset.quantum_encoder.is_fitted
            and services.index_dataset.statistical_encoder.is_fitted
        )
        if snapshot and docs_count > 0 and snapshot.document_count == docs_count and encoders_fitted:
            audit_print("api.index.status.completed_from_snapshot", dataset_id=dataset_id, docs_count=docs_count)
            return {
                "dataset_id": dataset_id,
                "status": "completed",
                "indexed_count": docs_count,
                "total_hint": snapshot.document_count,
                "result": {
                    "dataset_id": dataset_id,
                    "indexed_count": docs_count,
                    "skipped": True,
                    "reason": "already_indexed",
                    "snapshot_document_count": snapshot.document_count,
                    "snapshot_query_count": snapshot.query_count,
                },
                "error": None,
                "started_at": None,
                "finished_at": None,
                "updated_at": None,
            }
    return state.to_dict()


def _search_and_maybe_persist(payload: SearchRequest, user_id: int, services: Services) -> dict:
    effective_top_k = FIXED_TOP_K
    audit_print(
        "api.search.request",
        user_id=user_id,
        dataset_id=payload.dataset_id,
        mode=payload.mode,
        top_k=effective_top_k,
        chat_id=payload.chat_id,
        query_id=payload.query_id,
        query=preview_text(payload.query),
    )
    category_log("SEARCH", pipeline=payload.mode, query=preview_text(payload.query), top_k=effective_top_k)
    if payload.query_id:
        gt = _find_ground_truth(services, payload.dataset_id, payload.query_id, payload.query)
        if gt:
            category_log("EVAL QUERY USED", query_id=payload.query_id, pipeline=payload.mode)
    result = services.search.execute(dataset=payload.dataset_id, query=payload.query, mode=payload.mode, top_k=effective_top_k)
    output: dict[str, Any] = {
        "query": result["query"],
        "mode": result["mode"],
        "results": _serialize_results(result["results"]),
    }
    if result.get("metrics") is not None:
        output["metrics"] = result["metrics"]
    if result.get("algorithm_details") is not None:
        output["algorithm_details"] = result["algorithm_details"]
    if "comparison" in result:
        serialized_comparison: dict[str, Any] = {}
        for pipeline_key in ("classical", "quantum", "statistical"):
            pipeline_data = result["comparison"].get(pipeline_key)
            if pipeline_data is not None:
                serialized_comparison[pipeline_key] = {
                    "results": _serialize_results(pipeline_data["results"]),
                    "metrics": pipeline_data.get("metrics"),
                    "algorithm_details": pipeline_data.get("algorithm_details"),
                }
        output["comparison"] = serialized_comparison
    if result.get("comparison_metrics") is not None:
        output["comparison_metrics"] = result["comparison_metrics"]
    _attach_ir_metrics(output, services, payload.dataset_id, payload.query_id, payload.query, effective_top_k)
    if payload.chat_id is not None:
        assistant_payload = services.build_assistant_message.execute(result)
        services.add_message.save_assistant_retrieval_result(user_id, payload.chat_id, assistant_payload)
        output["chat_persisted"] = True
    audit_print(
        "api.search.response",
        user_id=user_id,
        dataset_id=payload.dataset_id,
        mode=output.get("mode"),
        result_count=len(output.get("results") or []),
        has_comparison="comparison" in output,
    )
    return output


@router.post("/search")
def search(payload: SearchRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        return _search_and_maybe_persist(payload, user_id, services)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.post("/search/dataset")
def compat_search_dataset(payload: SearchRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    return search(payload, user_id, services)


@compat_router.post("/search/file")
async def compat_search_file(
    query: str = Form(...),
    mode: str = Form("compare"),
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    services: Services = Depends(get_services),
):
    raise HTTPException(
        status_code=410,
        detail="File attachments/search by file were removed. Use dataset-backed chat search via /search/dataset or /api/search.",
    )


@router.post("/ground-truth")
def upsert_ground_truth(payload: GroundTruthUpsertRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        item = services.upsert_ground_truth.execute(payload.dataset_id, payload.query_id, payload.query_text, payload.relevant_doc_ids, user_id)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc
    return {
        "dataset_id": item.dataset,
        "query_id": item.query_id,
        "query_text": item.query_text,
        "relevant_doc_ids": item.relevant_doc_ids,
    }


@router.post("/evaluate")
def evaluate(payload: EvaluateRequest, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    try:
        return services.evaluate.execute(payload.dataset_id, payload.pipeline, FIXED_TOP_K)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@compat_router.get("/datasets")
def list_datasets(services: Services = Depends(get_services)):
    snapshots = services.dataset_snapshots.list_all()
    if snapshots:
        return [
            {
                "dataset_id": s.dataset_id,
                "name": s.name,
                "description": s.description,
                "document_count": s.document_count,
                "query_count": s.query_count,
            }
            for s in snapshots
        ]
    provider = services.index_dataset.datasets
    try:
        datasets = provider.list_datasets()
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc
    return [
        {
            "dataset_id": d["dataset_id"],
            "name": d["name"],
            "description": d["description"],
            "document_count": d["document_count"],
            "query_count": d["query_count"],
        }
        for d in datasets
    ]


@compat_router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, services: Services = Depends(get_services)):
    snapshot = services.dataset_snapshots.get(dataset_id)
    if snapshot:
        return {
            "dataset_id": snapshot.dataset_id,
            "name": snapshot.name,
            "provider": snapshot.provider,
            "source_url": snapshot.source_url,
            "reference_urls": snapshot.reference_urls,
            "description": snapshot.description,
            "subset": {"max_docs": snapshot.max_docs, "max_queries": snapshot.max_queries},
            "document_count": snapshot.document_count,
            "query_count": snapshot.query_count,
            "queries": [
                {
                    "query_id": q.get("query_id"),
                    "query": q.get("query_text", ""),
                    "relevant_count": len(q.get("relevant_doc_ids") or []),
                }
                for q in snapshot.queries
            ],
            "snapshot_queries": snapshot.queries,
            "documents": [{"doc_id": doc_id} for doc_id in snapshot.document_ids[:50]],
            "snapshot": {
                "persisted": True,
                "document_ids_count": len(snapshot.document_ids),
                "documents_preview_count": min(len(snapshot.document_ids), 50),
                "queries_count": len(snapshot.queries),
                "created_at": _serialize_dt(snapshot.created_at),
                "updated_at": _serialize_dt(snapshot.updated_at),
            },
        }
    provider = services.index_dataset.datasets
    try:
        dataset = provider.get_dataset(dataset_id)
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@compat_router.get("/benchmarks/labels")
def list_benchmark_labels(dataset_id: str, services: Services = Depends(get_services)):
    items = services.ground_truths.list_by_dataset(dataset_id)
    return {
        "items": [
            {
                "benchmark_id": item.query_id,
                "dataset_id": item.dataset,
                "query_text": item.query_text,
                "ideal_answer": item.ideal_answer or "",
                "relevant_doc_ids": item.relevant_doc_ids,
            }
            for item in items
        ]
    }


@compat_router.post("/benchmarks/labels")
def upsert_benchmark_label(payload: BenchmarkLabelInput, user_id: int = Depends(get_current_user_id), services: Services = Depends(get_services)):
    # Reuse existing entry's query_id if one already exists for this query text,
    # so we don't create a duplicate row with a text-derived id.
    existing = _find_ground_truth(services, payload.dataset_id, None, payload.query_text)
    query_id = existing.query_id if existing else payload.query_text.lower().replace(" ", "-")[:64]
    relevant = payload.relevant_doc_ids or (existing.relevant_doc_ids if existing else [])
    if not relevant:
        candidate = services.search.execute(payload.dataset_id, payload.query_text, "classical", 5)
        relevant = [r.doc_id for r in (candidate.get("results") or [])]
    if not relevant:
        raise HTTPException(status_code=400, detail="Nao achou relevancia na pergunta.")
    item = services.upsert_ground_truth.execute(
        dataset=payload.dataset_id,
        query_id=query_id,
        query_text=payload.query_text,
        relevant_doc_ids=relevant,
        user_id=user_id,
    )
    return {
        "benchmark_id": item.query_id,
        "dataset_id": item.dataset,
        "query_text": item.query_text,
        "ideal_answer": item.ideal_answer or "",
        "relevant_doc_ids": item.relevant_doc_ids,
    }


@compat_router.delete("/benchmarks/labels/{dataset_id}/{benchmark_id}", status_code=204)
def delete_benchmark_label(dataset_id: str, benchmark_id: str, services: Services = Depends(get_services)):
    if not services.ground_truths.delete(dataset_id, benchmark_id):
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return None


@compat_router.get("/evaluation/queries")
def list_evaluation_queries(dataset_id: str = "beir/trec-covid", services: Services = Depends(get_services)):
    items = services.ground_truths.list_by_dataset(dataset_id)
    return [
        {"query_id": item.query_id, "query": item.query_text, "ideal_answer": item.ideal_answer}
        for item in items
    ]


@compat_router.post("/benchmarks/evaluate/start")
def start_batch_evaluation(
    req: BatchEvaluateRequest,
    user_id: int = Depends(get_current_user_id),
    services: Services = Depends(get_services),
):
    snapshot = services.dataset_snapshots.get(req.dataset_id)
    if not snapshot:
        raise HTTPException(status_code=400, detail="Dataset not indexed")

    encoders_fitted = (
        services.index_dataset.classical_encoder.is_fitted
        and services.index_dataset.quantum_encoder.is_fitted
        and services.index_dataset.statistical_encoder.is_fitted
    )
    if not encoders_fitted:
        raise HTTPException(status_code=400, detail="Encoders not fitted. Index the dataset first.")

    valid_gts = services.ground_truths.list_by_dataset(req.dataset_id)
    if not valid_gts:
        raise HTTPException(status_code=400, detail="No valid queries found")

    current_status = evaluation_job_registry.status
    if current_status["status"] == "running":
        return {"accepted": False, "detail": "Evaluation already running", **current_status}

    started = evaluation_job_registry.start(
        services.evaluate.execute,
        dataset_id=req.dataset_id,
        pipelines=req.pipelines,
        k=FIXED_TOP_K,
        query_count=len(valid_gts),
    )
    return {"accepted": started, **evaluation_job_registry.status}


@compat_router.get("/benchmarks/evaluate/status")
def batch_evaluation_status(
    user_id: int = Depends(get_current_user_id),
    services: Services = Depends(get_services),
):
    return evaluation_job_registry.status
