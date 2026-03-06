from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from audit import audit_print, category_log, preview_results, preview_text, preview_vector
from domain.entities import DatasetSnapshot, Document, EvaluationResult, GroundTruth, Pipeline, Query
from domain.exceptions import NotFoundError, ValidationError
from domain.ports import (
    DatasetProviderPort,
    DatasetSnapshotRepositoryPort,
    DocumentRepositoryPort,
    EvaluationAggregate,
    GroundTruthRepositoryPort,
    MetricsPort,
)

if TYPE_CHECKING:
    from infrastructure.encoders.classical import ClassicalPipelineEncoder
    from infrastructure.encoders.quantum import QuantumPipelineEncoder
    from infrastructure.encoders.statistical import StatisticalPipelineEncoder


@dataclass(slots=True)
class SearchPipelineResponse:
    results: list
    pipeline: str


class IndexDatasetUseCase:
    """Two-pass indexing:
    Pass 1 — collect all documents and generate raw SBERT embeddings.
    Fit — PCA/SVD transformations are fit on the full corpus embeddings.
    Pass 2 — apply transformations and upsert all three pipeline vectors.
    """

    def __init__(
        self,
        datasets: DatasetProviderPort,
        documents: DocumentRepositoryPort,
        classical_encoder: "ClassicalPipelineEncoder",
        quantum_encoder: "QuantumPipelineEncoder",
        statistical_encoder: "StatisticalPipelineEncoder",
        dataset_snapshots: DatasetSnapshotRepositoryPort | None = None,
        ground_truths: GroundTruthRepositoryPort | None = None,
        encoder_state_dir: str | None = None,
    ) -> None:
        self.datasets = datasets
        self.documents = documents
        self.classical_encoder = classical_encoder
        self.quantum_encoder = quantum_encoder
        self.statistical_encoder = statistical_encoder
        self.dataset_snapshots = dataset_snapshots
        self.ground_truths = ground_truths
        self.encoder_state_dir = encoder_state_dir

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

        # ── Pass 1: Collect all document data ─────────────────────────────────
        doc_infos: list[dict] = []
        texts: list[str] = []

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
            texts.append(text)
            doc_infos.append(item)

        audit_print(
            "index.pass1.completed",
            dataset_id=dataset_id,
            total_documents=len(texts),
        )

        # ── Fit: generate raw SBERT embeddings and fit all transformers ────────
        audit_print("index.fit.start", dataset_id=dataset_id, n_docs=len(texts))
        raw_embeddings = self.classical_encoder.base.encode_batch(texts)
        audit_print(
            "index.fit.raw_embeddings_ready",
            dataset_id=dataset_id,
            shape=list(raw_embeddings.shape),
        )

        self.classical_encoder.fit(raw_embeddings)
        audit_print(
            "index.fit.classical_completed",
            dataset_id=dataset_id,
            output_dim=self.classical_encoder.dim,
        )

        self.quantum_encoder.fit(raw_embeddings)
        audit_print(
            "index.fit.quantum_completed",
            dataset_id=dataset_id,
            base_pca_dim=self.quantum_encoder.dim,
            angle_pca_dim=self.quantum_encoder.n_qubits,
            circuit_output_dim=self.quantum_encoder.circuit_output_dim,
            final_dim=self.quantum_encoder.dim,
        )

        self.statistical_encoder.fit(raw_embeddings)
        audit_print(
            "index.fit.statistical_completed",
            dataset_id=dataset_id,
            pca_intermediate_dim=self.statistical_encoder.pca_intermediate_dim,
            output_dim=self.statistical_encoder.dim,
        )

        if self.encoder_state_dir:
            self.classical_encoder.save_state(f"{self.encoder_state_dir}/classical.joblib")
            self.quantum_encoder.save_state(f"{self.encoder_state_dir}/quantum.joblib")
            self.statistical_encoder.save_state(f"{self.encoder_state_dir}/statistical.joblib")
            audit_print("index.fit.encoder_state_saved", encoder_state_dir=self.encoder_state_dir)

        # ── Pass 2: Apply transformations in batches and upsert ───────────────
        batch_size = 64
        batch: list[Document] = []
        document_ids: list[str] = []
        total = 0

        expected_dim = self.classical_encoder.dim
        for i, (item, raw) in enumerate(zip(doc_infos, raw_embeddings)):
            embedding_vector = self.classical_encoder.transform(raw)
            quantum_vector = self.quantum_encoder.transform(raw)
            statistical_vector = self.statistical_encoder.transform(raw)

            # Explicit dimension validation before upsert
            for vec, name in [
                (embedding_vector, "classical"),
                (quantum_vector, "quantum"),
                (statistical_vector, "statistical"),
            ]:
                if vec is not None and len(vec) != expected_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: {name} has {len(vec)}, expected {expected_dim}"
                    )

            audit_print(
                "index.document.encoded",
                dataset_id=dataset_id,
                doc_id=item["doc_id"],
                embedding=preview_vector(embedding_vector),
                quantum=preview_vector(quantum_vector),
                statistical=preview_vector(statistical_vector),
            )

            # Emit [VECTOR SAMPLE] on first doc and every 100 docs
            if i % 100 == 0:
                category_log("VECTOR SAMPLE classical", _extra={"values": str([round(v, 4) for v in embedding_vector[:8]])})
                category_log("VECTOR SAMPLE quantum", _extra={"values": str([round(v, 4) for v in quantum_vector[:8]])})
                category_log("VECTOR SAMPLE statistical", _extra={"values": str([round(v, 4) for v in statistical_vector[:8]])})

            document_ids.append(item["doc_id"])
            batch.append(
                Document(
                    dataset=dataset_id,
                    doc_id=item["doc_id"],
                    title=item.get("title"),
                    text=item["text"],
                    metadata=item.get("metadata", {}),
                    embedding_vector=embedding_vector,
                    quantum_vector=quantum_vector,
                    statistical_vector=statistical_vector,
                )
            )

            if len(batch) >= batch_size:
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

        # ── Persist queries and qrels ─────────────────────────────────────────
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
            "classical_dim": self.classical_encoder.dim,
            "quantum_dim": self.quantum_encoder.dim,
            "statistical_dim": self.statistical_encoder.dim,
            "snapshot_document_count": len(document_ids),
            "snapshot_query_count": len(query_snapshot),
            "qrels_count": qrels_count,
        }
        audit_print("index.use_case.completed", **result)
        return result


class SearchUseCase:
    def __init__(
        self,
        documents: DocumentRepositoryPort,
        classical_encoder: "ClassicalPipelineEncoder",
        quantum_encoder: "QuantumPipelineEncoder",
        statistical_encoder: "StatisticalPipelineEncoder",
    ) -> None:
        self.documents = documents
        self.classical_encoder = classical_encoder
        self.quantum_encoder = quantum_encoder
        self.statistical_encoder = statistical_encoder

    def _search_single(self, dataset: str, query: str, pipeline: Pipeline, top_k: int):
        t0 = time.perf_counter()
        audit_print(
            "search.pipeline.start",
            dataset_id=dataset,
            pipeline=pipeline.value,
            top_k=top_k,
            query=preview_text(query),
        )

        if pipeline == Pipeline.CLASSICAL:
            qv = self.classical_encoder.encode(query)
            t1 = time.perf_counter()
            category_log("VECTOR SAMPLE classical", _extra={"values": str([round(v, 4) for v in qv[:8]])})
            results = self.documents.search_by_embedding(dataset, qv, top_k)
        elif pipeline == Pipeline.QUANTUM:
            qv = self.quantum_encoder.encode(query)
            t1 = time.perf_counter()
            category_log("VECTOR SAMPLE quantum", _extra={"values": str([round(v, 4) for v in qv[:8]])})
            results = self.documents.search_by_quantum(dataset, qv, top_k)
        elif pipeline == Pipeline.STATISTICAL:
            qv = self.statistical_encoder.encode(query)
            t1 = time.perf_counter()
            category_log("VECTOR SAMPLE statistical", _extra={"values": str([round(v, 4) for v in qv[:8]])})
            results = self.documents.search_by_statistical(dataset, qv, top_k)
        else:
            raise ValidationError("Invalid pipeline.")

        t2 = time.perf_counter()
        encode_time_ms = round((t1 - t0) * 1000.0, 3)
        search_time_ms = round((t2 - t1) * 1000.0, 3)
        total_time_ms = round((t2 - t0) * 1000.0, 3)
        audit_print(
            "search.pipeline.completed",
            dataset_id=dataset,
            pipeline=pipeline.value,
            encode_time_ms=encode_time_ms,
            search_time_ms=search_time_ms,
            total_time_ms=total_time_ms,
            query_vector=preview_vector(qv),
            results=preview_results(results),
        )
        category_log("TIME", _extra={"pipeline": pipeline.value, "encode_time_ms": encode_time_ms})
        category_log("TIME", _extra={"pipeline": pipeline.value, "search_time_ms": search_time_ms})
        category_log("TIME", _extra={"pipeline": pipeline.value, "total_time_ms": total_time_ms})
        category_log("SEARCH", pipeline=pipeline.value, top_k=top_k, results=len(results))
        return results, encode_time_ms, search_time_ms, total_time_ms

    def _algorithm_details(self, pipeline: Pipeline) -> dict:
        if pipeline == Pipeline.CLASSICAL:
            return {
                "algorithm": "classical-sbert-pca-cosine",
                "comparator": "cosine similarity (score = 1 - cosine_distance)",
                "candidate_strategy": "full dataset ranking in embedding_vector space",
                "description": "BERT → PCA(64) → L2 normalize → cosine similarity ranking.",
                "debug": {
                    "vector_space": "embedding_vector",
                    "encoder": "ClassicalPipelineEncoder",
                    "normalization": "L2",
                    "steps": [
                        "Recebe a pergunta em texto",
                        "Gera embedding semântico base via SentenceTransformer (384-dim)",
                        "Reduz dimensionalidade via PCA (384→64)",
                        "Normaliza com norma L2",
                        "Consulta a coluna embedding_vector no banco via cosine similarity",
                        "Retorna top-k documentos ordenados por score",
                    ],
                },
            }
        if pipeline == Pipeline.QUANTUM:
            return {
                "algorithm": "residual-quantum-feature-map-cosine",
                "comparator": "cosine similarity (score = 1 - cosine_distance)",
                "candidate_strategy": "full dataset ranking in quantum_vector space",
                "description": "BERT → PCA_base(64) → PCA_angles(6) → AngleEmbedding + StronglyEntanglingLayers → probs → Hellinger(64) → concat(128) → PCA_final(64) → L2 normalize → cosine similarity ranking.",
                "debug": {
                    "vector_space": "quantum_vector",
                    "encoder": "QuantumPipelineEncoder",
                    "normalization": "L2",
                    "steps": [
                        "Recebe a pergunta em texto",
                        "Gera embedding semântico base via SentenceTransformer (384-dim)",
                        "Reduz dimensionalidade via PCA_base (384→64) → base_vector_64",
                        "Reduz para ângulos via PCA_angles (64→6) e normaliza para [0,π]",
                        "Codifica ângulos no circuito quântico via AngleEmbedding (6 qubits)",
                        "Aplica camadas entrelaçadas via StronglyEntanglingLayers",
                        "Mede probabilidades de estado quântico (2^6 = 64-dim)",
                        "Aplica transformação de Hellinger: sqrt(probabilidades) → quantum_vector_64",
                        "Concatena base_vector_64 + quantum_vector_64 → vector_128",
                        "Reduz via PCA_final (128→64) → final_vector_64",
                        "Normaliza com norma L2",
                        "Consulta a coluna quantum_vector no banco via cosine similarity",
                        "Retorna top-k documentos ordenados por score",
                    ],
                },
            }
        if pipeline == Pipeline.STATISTICAL:
            return {
                "algorithm": "statistical-sbert-svd-cosine",
                "comparator": "cosine similarity (score = 1 - cosine_distance)",
                "candidate_strategy": "full dataset ranking in statistical_vector space",
                "description": "BERT → PCA(128) → TruncatedSVD(64) → L2 normalize → cosine similarity ranking.",
                "debug": {
                    "vector_space": "statistical_vector",
                    "encoder": "StatisticalPipelineEncoder",
                    "normalization": "L2",
                    "steps": [
                        "Recebe a pergunta em texto",
                        "Gera embedding semântico base via SentenceTransformer (384-dim)",
                        "Reduz dimensionalidade e centraliza via PCA (384→128)",
                        "Aplica fatoração matricial via TruncatedSVD (128→64)",
                        "Normaliza com norma L2",
                        "Consulta a coluna statistical_vector no banco via cosine similarity",
                        "Retorna top-k documentos ordenados por score",
                    ],
                },
            }
        return {}

    def _search_metrics(self, results, encode_time_ms: float, search_time_ms: float, total_time_ms: float, top_k: int) -> dict:
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
            "encode_time_ms": encode_time_ms,
            "search_time_ms": search_time_ms,
            "total_time_ms": total_time_ms,
            "k": top_k,
            "candidate_k": top_k,
            "has_labels": False,
            "debug": {
                "retrieved_count": len(results),
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
            },
        }

    def _compare_rankings(self, classical_results, quantum_results, statistical_results, top_k: int) -> dict:
        c_ids = {item.doc_id for item in classical_results[:top_k]}
        q_ids = {item.doc_id for item in quantum_results[:top_k]}
        s_ids = {item.doc_id for item in statistical_results[:top_k]}
        return {
            "top_k": top_k,
            "common_doc_ids": sorted(c_ids & q_ids & s_ids),
            "common_classical_quantum": sorted(c_ids & q_ids),
            "common_classical_statistical": sorted(c_ids & s_ids),
            "common_quantum_statistical": sorted(q_ids & s_ids),
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
            results, enc_ms, srch_ms, tot_ms = self._search_single(dataset, query, Pipeline.CLASSICAL, top_k)
            payload = {
                "query": query,
                "mode": mode,
                "results": results,
                "metrics": self._search_metrics(results, enc_ms, srch_ms, tot_ms, top_k),
                "algorithm_details": self._algorithm_details(Pipeline.CLASSICAL),
            }
            audit_print("search.use_case.completed", dataset_id=dataset, mode=mode, results=preview_results(results))
            return payload

        if mode == Pipeline.QUANTUM.value:
            results, enc_ms, srch_ms, tot_ms = self._search_single(dataset, query, Pipeline.QUANTUM, top_k)
            payload = {
                "query": query,
                "mode": mode,
                "results": results,
                "metrics": self._search_metrics(results, enc_ms, srch_ms, tot_ms, top_k),
                "algorithm_details": self._algorithm_details(Pipeline.QUANTUM),
            }
            audit_print("search.use_case.completed", dataset_id=dataset, mode=mode, results=preview_results(results))
            return payload

        if mode == Pipeline.STATISTICAL.value:
            results, enc_ms, srch_ms, tot_ms = self._search_single(dataset, query, Pipeline.STATISTICAL, top_k)
            payload = {
                "query": query,
                "mode": mode,
                "results": results,
                "metrics": self._search_metrics(results, enc_ms, srch_ms, tot_ms, top_k),
                "algorithm_details": self._algorithm_details(Pipeline.STATISTICAL),
            }
            audit_print("search.use_case.completed", dataset_id=dataset, mode=mode, results=preview_results(results))
            return payload

        if mode == Pipeline.COMPARE.value:
            classical, c_enc, c_srch, c_tot = self._search_single(dataset, query, Pipeline.CLASSICAL, top_k)
            quantum, q_enc, q_srch, q_tot = self._search_single(dataset, query, Pipeline.QUANTUM, top_k)
            statistical, s_enc, s_srch, s_tot = self._search_single(dataset, query, Pipeline.STATISTICAL, top_k)
            payload = {
                "query": query,
                "mode": Pipeline.COMPARE.value,
                "results": classical,
                "comparison": {
                    "classical": {
                        "results": classical,
                        "metrics": self._search_metrics(classical, c_enc, c_srch, c_tot, top_k),
                        "algorithm_details": self._algorithm_details(Pipeline.CLASSICAL),
                    },
                    "quantum": {
                        "results": quantum,
                        "metrics": self._search_metrics(quantum, q_enc, q_srch, q_tot, top_k),
                        "algorithm_details": self._algorithm_details(Pipeline.QUANTUM),
                    },
                    "statistical": {
                        "results": statistical,
                        "metrics": self._search_metrics(statistical, s_enc, s_srch, s_tot, top_k),
                        "algorithm_details": self._algorithm_details(Pipeline.STATISTICAL),
                    },
                },
                "comparison_metrics": self._compare_rankings(classical, quantum, statistical, top_k),
            }
            audit_print(
                "search.use_case.completed",
                dataset_id=dataset,
                mode=mode,
                classical_results=preview_results(classical),
                quantum_results=preview_results(quantum),
                statistical_results=preview_results(statistical),
                common_doc_ids=payload["comparison_metrics"]["common_doc_ids"],
            )
            return payload

        raise ValidationError("Invalid mode. Use 'classical', 'quantum', 'statistical', or 'compare'.")


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

    def execute(self, dataset: str, pipeline: str = "compare", k: int = 10) -> dict:
        gts = self.ground_truths.list_by_dataset(dataset)
        if not gts:
            raise NotFoundError("No ground truth entries found for dataset.")

        if pipeline == "compare":
            pipelines = [Pipeline.CLASSICAL.value, Pipeline.QUANTUM.value, Pipeline.STATISTICAL.value]
        else:
            pipelines = [pipeline]

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
                    mean_mrr=sum(x.mrr for x in per_query) / n,
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
                    "mean_mrr": agg.mean_mrr,
                    "per_query": [
                        {
                            "query_id": q.query_id,
                            "query_text": q.query_text,
                            "precision_at_k": q.precision_at_k,
                            "recall_at_k": q.recall_at_k,
                            "ndcg_at_k": q.ndcg_at_k,
                            "mrr": q.mrr,
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
