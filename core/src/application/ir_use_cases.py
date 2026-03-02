from __future__ import annotations

import time
from dataclasses import dataclass

from audit import audit_print, preview_results, preview_text, preview_vector
from domain.entities import DatasetSnapshot, Document, EvaluationResult, GroundTruth, Pipeline, Query
from domain.exceptions import NotFoundError, ValidationError
from domain.ports import (
    DatasetProviderPort,
    DatasetSnapshotRepositoryPort,
    DocumentRepositoryPort,
    EncoderPort,
    EvaluationAggregate,
    GroundTruthRepositoryPort,
    MetricsPort,
)


@dataclass(slots=True)
class SearchPipelineResponse:
    results: list
    pipeline: str


class IndexDatasetUseCase:
    def __init__(
        self,
        datasets: DatasetProviderPort,
        documents: DocumentRepositoryPort,
        embedding_encoder: EncoderPort,
        quantum_encoder: EncoderPort,
        dataset_snapshots: DatasetSnapshotRepositoryPort | None = None,
        ground_truths: GroundTruthRepositoryPort | None = None,
    ) -> None:
        self.datasets = datasets
        self.documents = documents
        self.embedding_encoder = embedding_encoder
        self.quantum_encoder = quantum_encoder
        self.dataset_snapshots = dataset_snapshots
        self.ground_truths = ground_truths

    def execute(self, dataset_id: str, progress_callback=None) -> dict:
        dataset_meta = self.datasets.get_dataset(dataset_id)
        if not dataset_meta:
            raise NotFoundError("Dataset not found.")
        audit_print(
            "index.use_case.start",
            dataset_id=dataset_id,
            dataset_name=dataset_meta.get("name"),
            document_count_hint=dataset_meta.get("document_count"),
            query_count_hint=dataset_meta.get("query_count"),
        )
        batch: list[Document] = []
        document_ids: list[str] = []
        total = 0
        for item in self.datasets.iter_documents(dataset_id):
            text = item["text"]
            audit_print(
                "index.document.received",
                dataset_id=dataset_id,
                doc_id=item["doc_id"],
                title=preview_text(item.get("title"), limit=80),
                text=preview_text(text),
                metadata_keys=sorted((item.get("metadata") or {}).keys()),
            )
            embedding_vector = self.embedding_encoder.encode(text)
            quantum_vector = self.quantum_encoder.encode(text)
            audit_print(
                "index.document.encoded",
                dataset_id=dataset_id,
                doc_id=item["doc_id"],
                embedding=preview_vector(embedding_vector),
                quantum=preview_vector(quantum_vector),
            )
            document_ids.append(item["doc_id"])
            batch.append(
                Document(
                    dataset=dataset_id,
                    doc_id=item["doc_id"],
                    title=item.get("title"),
                    text=text,
                    metadata=item.get("metadata", {}),
                    embedding_vector=embedding_vector,
                    quantum_vector=quantum_vector,
                )
            )
            if len(batch) >= 64:
                audit_print(
                    "index.batch.flush",
                    dataset_id=dataset_id,
                    batch_size=len(batch),
                    doc_ids=[doc.doc_id for doc in batch],
                )
                total += self.documents.upsert_documents(batch)
                audit_print(
                    "index.batch.persisted",
                    dataset_id=dataset_id,
                    batch_size=len(batch),
                    indexed_total=total,
                )
                if progress_callback is not None:
                    progress_callback(total)
                batch.clear()
        if batch:
            audit_print(
                "index.batch.flush",
                dataset_id=dataset_id,
                batch_size=len(batch),
                doc_ids=[doc.doc_id for doc in batch],
            )
            total += self.documents.upsert_documents(batch)
            audit_print(
                "index.batch.persisted",
                dataset_id=dataset_id,
                batch_size=len(batch),
                indexed_total=total,
            )
            if progress_callback is not None:
                progress_callback(total)
        query_snapshot = []
        qrels_count = 0
        for q in self.datasets.iter_queries(dataset_id):
            relevant_doc_ids = list(q.get("relevant_doc_ids") or [])
            qrels = {str(doc_id): int(rel) for doc_id, rel in (q.get("qrels") or {}).items()}
            query_snapshot.append(
                {
                    "query_id": q["query_id"],
                    "query_text": q["query_text"],
                    "relevant_doc_ids": relevant_doc_ids,
                }
            )
            audit_print(
                "index.query.received",
                dataset_id=dataset_id,
                query_id=q["query_id"],
                split=str(q.get("split") or "test"),
                query_text=preview_text(str(q["query_text"])),
                relevant_doc_ids_preview=relevant_doc_ids[:10],
                relevant_doc_ids_count=len(relevant_doc_ids),
                qrels_count=len(qrels),
            )
            qrels_count += len(qrels)
            if self.ground_truths is not None:
                self.ground_truths.upsert_qrels(
                    dataset=dataset_id,
                    split=str(q.get("split") or "test"),
                    query_id=str(q["query_id"]),
                    query_text=str(q["query_text"]),
                    qrels=qrels,
                )
        if self.dataset_snapshots is not None:
            subset = dataset_meta.get("subset") or {}
            self.dataset_snapshots.upsert(
                DatasetSnapshot(
                    dataset_id=dataset_id,
                    name=dataset_meta.get("name") or dataset_id,
                    provider=dataset_meta.get("provider") or "unknown",
                    description=dataset_meta.get("description") or "",
                    source_url=dataset_meta.get("source_url"),
                    reference_urls=list(dataset_meta.get("reference_urls") or []),
                    max_docs=subset.get("max_docs"),
                    max_queries=subset.get("max_queries"),
                    document_count=len(document_ids),
                    query_count=len(query_snapshot),
                    document_ids=document_ids,
                    queries=query_snapshot,
                )
            )
        result = {
            "dataset_id": dataset_id,
            "indexed_count": total,
            "embedding_dim": self.embedding_encoder.dim,
            "quantum_dim": self.quantum_encoder.dim,
            "snapshot_document_count": len(document_ids),
            "snapshot_query_count": len(query_snapshot),
            "qrels_count": qrels_count,
        }
        audit_print("index.use_case.completed", **result)
        return result


class SearchUseCase:
    def __init__(self, documents: DocumentRepositoryPort, embedding_encoder: EncoderPort, quantum_encoder: EncoderPort) -> None:
        self.documents = documents
        self.embedding_encoder = embedding_encoder
        self.quantum_encoder = quantum_encoder

    def _search_single(self, dataset: str, query: str, pipeline: Pipeline, top_k: int):
        started = time.perf_counter()
        if pipeline == Pipeline.CLASSICAL:
            audit_print(
                "search.pipeline.start",
                dataset_id=dataset,
                pipeline=pipeline.value,
                top_k=top_k,
                query=preview_text(query),
            )
            qv = self.embedding_encoder.encode(query)
            results = self.documents.search_by_embedding(dataset, qv, top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            audit_print(
                "search.pipeline.completed",
                dataset_id=dataset,
                pipeline=pipeline.value,
                latency_ms=round(latency_ms, 3),
                query_vector=preview_vector(qv),
                results=preview_results(results),
            )
            return results, latency_ms
        if pipeline == Pipeline.QUANTUM:
            audit_print(
                "search.pipeline.start",
                dataset_id=dataset,
                pipeline=pipeline.value,
                top_k=top_k,
                query=preview_text(query),
            )
            qv = self.quantum_encoder.encode(query)
            results = self.documents.search_by_quantum(dataset, qv, top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            audit_print(
                "search.pipeline.completed",
                dataset_id=dataset,
                pipeline=pipeline.value,
                latency_ms=round(latency_ms, 3),
                query_vector=preview_vector(qv),
                results=preview_results(results),
            )
            return results, latency_ms
        raise ValidationError("Invalid pipeline.")

    def _algorithm_details(self, pipeline: Pipeline) -> dict:
        if pipeline == Pipeline.CLASSICAL:
            return {
                "algorithm": "classical-sbert-cosine",
                "comparator": "cosine similarity (score = 1 - cosine_distance)",
                "candidate_strategy": "full dataset ranking in embedding_vector space",
                "description": "Query and documents are encoded with the same classical encoder (sBERT), L2-normalized, and ranked by cosine similarity.",
                "debug": {
                    "vector_space": "embedding_vector",
                    "encoder": "encode_embedding(text)",
                    "normalization": "L2",
                    "steps": [
                        "Recebe a pergunta em texto",
                        "Transforma a pergunta em embedding classico (sBERT)",
                        "Usa o mesmo encoder classico usado na indexacao dos documentos",
                        "Consulta a coluna embedding_vector no banco",
                        "Calcula similaridade cosseno e ordena por score",
                        "Retorna top-k documentos classicos",
                    ],
                },
            }
        return {
            "algorithm": "quantum-probability-cosine",
            "comparator": "cosine similarity (score = 1 - cosine_distance)",
            "candidate_strategy": "full dataset ranking in quantum_vector space",
            "description": "Query and documents are encoded with the same deterministic quantum-inspired encoder (PennyLane probabilities), L2-normalized, and ranked by cosine similarity.",
            "debug": {
                "vector_space": "quantum_vector",
                "encoder": "encode_quantum(text)",
                "normalization": "L2",
                "steps": [
                    "Recebe a pergunta em texto",
                    "Transforma a pergunta em vetor quantico-inspirado deterministico",
                    "Usa o mesmo encoder quantico usado na indexacao dos documentos",
                    "Consulta a coluna quantum_vector no banco",
                    "Calcula similaridade cosseno e ordena por score",
                    "Retorna top-k documentos quanticos",
                ],
            },
        }

    def _search_metrics(self, results, latency_ms: float, top_k: int) -> dict:
        scores = [float(x.score) for x in results]
        return {
            "accuracy_at_k": None,
            "precision_at_k": None,
            "recall_at_k": None,
            "f1_at_k": None,
            "mrr": None,
            "ndcg_at_k": None,
            "answer_similarity": None,
            "has_ideal_answer": False,
            "latency_ms": latency_ms,
            "k": top_k,
            "candidate_k": top_k,
            "has_labels": False,
            "debug": {
                "retrieved_count": len(results),
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
            },
        }

    def _compare_rankings(self, classical_results, quantum_results, top_k: int) -> dict:
        c_ids = [item.doc_id for item in classical_results[:top_k]]
        q_ids = [item.doc_id for item in quantum_results[:top_k]]
        c_set = set(c_ids)
        q_set = set(q_ids)
        intersection = sorted(c_set.intersection(q_set))

        return {
            "top_k": top_k,
            "common_doc_ids": intersection,
        }

    def execute(self, dataset: str, query: str, mode: str = "compare", top_k: int = 5) -> dict:
        audit_print(
            "search.use_case.start",
            dataset_id=dataset,
            mode=mode,
            top_k=top_k,
            query=preview_text(query),
        )
        if mode == Pipeline.CLASSICAL.value:
            results, latency_ms = self._search_single(dataset, query, Pipeline.CLASSICAL, top_k)
            payload = {
                "query": query,
                "mode": mode,
                "results": results,
                "metrics": self._search_metrics(results, latency_ms, top_k),
                "algorithm_details": self._algorithm_details(Pipeline.CLASSICAL),
            }
            audit_print("search.use_case.completed", dataset_id=dataset, mode=mode, results=preview_results(results))
            return payload
        if mode == Pipeline.QUANTUM.value:
            results, latency_ms = self._search_single(dataset, query, Pipeline.QUANTUM, top_k)
            payload = {
                "query": query,
                "mode": mode,
                "results": results,
                "metrics": self._search_metrics(results, latency_ms, top_k),
                "algorithm_details": self._algorithm_details(Pipeline.QUANTUM),
            }
            audit_print("search.use_case.completed", dataset_id=dataset, mode=mode, results=preview_results(results))
            return payload
        if mode == Pipeline.COMPARE.value:
            classical, classical_latency = self._search_single(dataset, query, Pipeline.CLASSICAL, top_k)
            quantum, quantum_latency = self._search_single(dataset, query, Pipeline.QUANTUM, top_k)
            payload = {
                "query": query,
                "mode": Pipeline.COMPARE.value,
                "results": classical,
                "comparison": {
                    "classical": {
                        "results": classical,
                        "metrics": self._search_metrics(classical, classical_latency, top_k),
                        "algorithm_details": self._algorithm_details(Pipeline.CLASSICAL),
                    },
                    "quantum": {
                        "results": quantum,
                        "metrics": self._search_metrics(quantum, quantum_latency, top_k),
                        "algorithm_details": self._algorithm_details(Pipeline.QUANTUM),
                    },
                },
                "comparison_metrics": self._compare_rankings(classical, quantum, top_k),
            }
            audit_print(
                "search.use_case.completed",
                dataset_id=dataset,
                mode=mode,
                classical_results=preview_results(classical),
                quantum_results=preview_results(quantum),
                common_doc_ids=payload["comparison_metrics"]["common_doc_ids"],
            )
            return payload
        raise ValidationError("Invalid mode. Use 'classical', 'quantum', or 'compare'.")


class UpsertGroundTruthUseCase:
    def __init__(self, ground_truths: GroundTruthRepositoryPort) -> None:
        self.ground_truths = ground_truths

    def execute(self, dataset: str, query_id: str, query_text: str, relevant_doc_ids: list[str], user_id: int | None = None) -> GroundTruth:
        if not relevant_doc_ids:
            raise ValidationError("relevant_doc_ids must not be empty.")
        return self.ground_truths.upsert(
            GroundTruth(query_id=query_id, query_text=query_text, relevant_doc_ids=relevant_doc_ids, dataset=dataset, user_id=user_id)
        )


class EvaluateUseCase:
    def __init__(
        self,
        ground_truths: GroundTruthRepositoryPort,
        search: SearchUseCase,
        metrics: MetricsPort,
    ) -> None:
        self.ground_truths = ground_truths
        self.search = search
        self.metrics = metrics

    def execute(self, dataset: str, pipeline: str = "compare", k: int = 5) -> dict:
        gts = self.ground_truths.list_by_dataset(dataset)
        if not gts:
            raise NotFoundError("No ground truth entries found for dataset.")
        pipelines = [Pipeline.CLASSICAL.value, Pipeline.QUANTUM.value] if pipeline == "compare" else [pipeline]
        aggregates: list[EvaluationAggregate] = []
        for current_pipeline in pipelines:
            per_query: list[EvaluationResult] = []
            for gt in gts:
                response = self.search.execute(dataset=dataset, query=gt.query_text, mode=current_pipeline, top_k=k)
                items = response["results"]
                per_query.append(
                    self.metrics.evaluate_query(
                        query_id=gt.query_id,
                        query_text=gt.query_text,
                        pipeline=current_pipeline,
                        retrieved_doc_ids=[item.doc_id for item in items],
                        retrieved_scores=[item.score for item in items],
                        relevant_doc_ids=gt.relevant_doc_ids,
                        k=k,
                    )
                )
            n = max(len(per_query), 1)
            aggregates.append(
                EvaluationAggregate(
                    pipeline=current_pipeline,
                    k=k,
                    per_query=per_query,
                    mean_precision_at_k=sum(x.precision_at_k for x in per_query) / n,
                    mean_recall_at_k=sum(x.recall_at_k for x in per_query) / n,
                    mean_ndcg_at_k=sum(x.ndcg_at_k for x in per_query) / n,
                    mean_spearman=sum(x.spearman for x in per_query) / n,
                )
            )
        return {
            "dataset_id": dataset,
            "k": k,
            "pipelines": [
                {
                    "pipeline": agg.pipeline,
                    "mean_precision_at_k": agg.mean_precision_at_k,
                    "mean_recall_at_k": agg.mean_recall_at_k,
                    "mean_ndcg_at_k": agg.mean_ndcg_at_k,
                    "mean_spearman": agg.mean_spearman,
                    "per_query": [
                        {
                            "query_id": q.query_id,
                            "query_text": q.query_text,
                            "precision_at_k": q.precision_at_k,
                            "recall_at_k": q.recall_at_k,
                            "ndcg_at_k": q.ndcg_at_k,
                            "spearman": q.spearman,
                            "top_k_doc_ids": q.top_k_doc_ids,
                        }
                        for q in agg.per_query
                    ],
                }
                for agg in aggregates
            ],
        }


class BuildAssistantRetrievalMessageUseCase:
    def execute(self, search_payload: dict) -> dict:
        return {
            "type": "retrieval_result",
            "query": search_payload.get("query"),
            "mode": search_payload.get("mode"),
            "top_k": len(search_payload.get("results") or []),
            "results": [
                {"doc_id": r.doc_id, "score": r.score}
                for r in (search_payload.get("results") or [])
            ],
        }
