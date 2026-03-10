from __future__ import annotations

import ir_measures  # type: ignore
from ir_measures import MRR, P, R, nDCG  # type: ignore

from audit import audit_print, category_log
from domain.entities import EvaluationResult


class IrMeasuresAdapter:
    """Metrics adapter using the ir_measures library.

    Computes the following Information Retrieval metrics per query:
      - nDCG@k  (Normalized Discounted Cumulative Gain)
      - Recall@k
      - MRR     (Mean Reciprocal Rank)
      - P@k     (Precision@k)

    No metric is implemented manually — all computation delegates to ir_measures.
    """

    def evaluate_query(
        self,
        *,
        query_id: str | None,
        query_text: str,
        pipeline: str,
        retrieved_doc_ids: list[str],
        retrieved_scores: list[float],
        relevant_doc_ids: list[str],
        k: int,
    ) -> EvaluationResult:
        qid = str(query_id) if query_id else "q0"
        relevant_set = set(relevant_doc_ids)

        audit_print(
            "metrics.ir_measures.evaluate.start",
            pipeline=pipeline,
            query_id=qid,
            k=k,
            retrieved_count=len(retrieved_doc_ids),
            relevant_count=len(relevant_doc_ids),
        )
        category_log(
            "METRICS INPUT",
            _extra={"pipeline": pipeline, "run_docs": len(retrieved_doc_ids), "qrels_docs": len(relevant_doc_ids)},
        )

        # Build qrels: all relevant docs with relevance=1
        qrels = [
            ir_measures.Qrel(query_id=qid, doc_id=str(doc_id), relevance=1)
            for doc_id in relevant_set
        ]

        # Build run: retrieved docs with their scores (descending order preserved)
        run = [
            ir_measures.ScoredDoc(query_id=qid, doc_id=str(doc_id), score=float(score))
            for doc_id, score in zip(retrieved_doc_ids, retrieved_scores)
        ]

        measures = [nDCG @ k, R @ k, MRR @ k, P @ k]

        results = ir_measures.calc_aggregate(measures, qrels, run)

        ndcg_val = float(results[nDCG @ k])
        recall_val = float(results[R @ k])
        mrr_val = float(results[MRR @ k])
        precision_val = float(results[P @ k])

        audit_print(
            "metrics.ir_measures.evaluate.completed",
            pipeline=pipeline,
            query_id=qid,
            k=k,
            ndcg=round(ndcg_val, 4),
            recall=round(recall_val, 4),
            mrr=round(mrr_val, 4),
            precision=round(precision_val, 4),
        )
        category_log(
            "METRICS RESULT",
            _extra={
                "pipeline": pipeline,
                f"nDCG@{k}": round(ndcg_val, 4),
                f"Recall@{k}": round(recall_val, 4),
                "MRR": round(mrr_val, 4),
                f"P@{k}": round(precision_val, 4),
            },
        )

        return EvaluationResult(
            query_id=query_id,
            query_text=query_text,
            pipeline=pipeline,
            precision_at_k=precision_val,
            recall_at_k=recall_val,
            ndcg_at_k=ndcg_val,
            mrr=mrr_val,
            top_k_doc_ids=retrieved_doc_ids[:k],
        )
