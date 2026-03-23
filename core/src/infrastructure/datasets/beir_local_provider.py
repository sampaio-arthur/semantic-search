from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from domain.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BeirLocalDatasetProvider:
    data_root: Path = Path(__file__).resolve().parents[3] / "data" / "beir"
    default_split: str = "test"

    def _normalize_dataset_id(self, dataset_id: str) -> str:
        value = (dataset_id or "").strip().replace("\\", "/").strip("/")
        if not value:
            raise ValidationError("dataset_id is required.")
        if value.startswith("beir/"):
            name = value.split("/", 1)[1]
        else:
            name = value
        if not name:
            raise ValidationError("Invalid dataset_id. Use format 'beir/<dataset_name>'.")
        return f"beir/{name}"

    def _dataset_dir(self, dataset_id: str) -> tuple[str, Path]:
        normalized = self._normalize_dataset_id(dataset_id)
        dataset_name = normalized.split("/", 1)[1]
        return normalized, self.data_root / dataset_name

    def _paths(self, dataset_id: str) -> tuple[str, Path, Path, Path]:
        normalized, base_dir = self._dataset_dir(dataset_id)
        corpus_path = base_dir / "corpus.jsonl"
        queries_path = base_dir / "queries.jsonl"
        qrels_path = base_dir / "qrels" / f"{self.default_split}.tsv"
        return normalized, corpus_path, queries_path, qrels_path

    def _ensure_files(self, dataset_id: str) -> tuple[str, Path, Path, Path]:
        normalized, corpus_path, queries_path, qrels_path = self._paths(dataset_id)
        missing = [str(p) for p in (corpus_path, queries_path, qrels_path) if not p.exists()]
        if missing:
            joined = ", ".join(missing)
            raise ValidationError(
                f"BEIR dataset files not found for '{normalized}'. Expected local files: {joined}. "
                "Place dataset files under core/data/beir/<dataset_name>/ and retry."
            )
        return normalized, corpus_path, queries_path, qrels_path

    def _iter_jsonl(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"Invalid JSONL at {path}:{line_number}.") from exc

    def _read_qrels(self, qrels_path: Path) -> dict[str, dict[str, int]]:
        qrels: dict[str, dict[str, int]] = {}
        with qrels_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                if raw.lower().startswith("query-id"):
                    continue
                parts = raw.split("\t")
                if len(parts) < 3:
                    raise ValidationError(f"Invalid qrels format at {qrels_path}:{line_number}.")
                if len(parts) >= 4 and parts[1].upper() == "Q0":
                    query_id, doc_id, score_raw = parts[0], parts[2], parts[3]
                else:
                    query_id, doc_id, score_raw = parts[0], parts[1], parts[2]
                try:
                    relevance = int(float(score_raw))
                except ValueError as exc:
                    raise ValidationError(f"Invalid relevance score at {qrels_path}:{line_number}.") from exc
                qrels.setdefault(str(query_id), {})[str(doc_id)] = relevance
        return qrels

    def _read_queries(self, queries_path: Path) -> dict[str, str]:
        queries: dict[str, str] = {}
        for item in self._iter_jsonl(queries_path):
            query_id = str(item.get("_id", "")).strip()
            query_text = str(item.get("text", "")).strip()
            if not query_id:
                continue
            queries[query_id] = query_text
        return queries

    def _read_corpus_ids(self, corpus_path: Path) -> set[str]:
        corpus_ids: set[str] = set()
        for item in self._iter_jsonl(corpus_path):
            doc_id = str(item.get("_id", "")).strip()
            if doc_id:
                corpus_ids.add(doc_id)
        return corpus_ids

    def list_datasets(self) -> list[dict]:
        if not self.data_root.exists():
            return []
        entries: list[dict] = []
        errors: list[str] = []
        for candidate in sorted(self.data_root.iterdir()):
            if not candidate.is_dir():
                continue
            dataset_id = f"beir/{candidate.name}"
            try:
                entries.append(self.get_dataset(dataset_id))
            except ValidationError as exc:
                errors.append(f"{dataset_id}: {exc}")
        if errors:
            raise ValidationError("Invalid dataset configuration(s): " + " | ".join(errors))
        return entries

    def get_dataset(self, dataset_id: str) -> dict | None:
        normalized, corpus_path, queries_path, qrels_path = self._ensure_files(dataset_id)
        qrels = self._read_qrels(qrels_path)
        queries = self._read_queries(queries_path)
        corpus_count = len(self._read_corpus_ids(corpus_path))
        filtered_queries = [qid for qid in queries.keys() if qid in qrels]
        total_qrels = sum(len(items) for items in qrels.values())
        logger.info(
            "BEIR dataset loaded: dataset=%s documents=%d queries=%d qrels=%d",
            normalized,
            corpus_count,
            len(filtered_queries),
            total_qrels,
        )
        return {
            "dataset_id": normalized,
            "name": f"BEIR - {normalized.split('/', 1)[1]}",
            "provider": "beir-local",
            "source_url": "https://github.com/beir-cellar/beir",
            "reference_urls": [
                "https://github.com/beir-cellar/beir",
                "https://huggingface.co/collections/BeIR/beir-datasets-64e5c66c4a7c11a7c1f59f3e",
            ],
            "description": "Local offline BEIR dataset loaded from core/data/beir/<dataset_name>/",
            "subset": {"split": self.default_split},
            "document_count": corpus_count,
            "query_count": len(filtered_queries),
            "qrels_count": total_qrels,
            "queries": [
                {
                    "query_id": qid,
                    "query": queries[qid],
                    "relevant_count": len(qrels.get(qid, {})),
                }
                for qid in filtered_queries[:100]
            ],
            "documents": [],
        }

    def iter_documents(self, dataset_id: str):
        normalized, corpus_path, _, _ = self._ensure_files(dataset_id)
        for item in self._iter_jsonl(corpus_path):
            doc_id = str(item.get("_id", "")).strip()
            text = item.get("text")
            if not doc_id:
                continue
            if text is None:
                raise ValidationError(f"Document '{doc_id}' in '{normalized}' is missing required field 'text'.")
            title = str(item.get("title", "")).strip()
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            if title:
                metadata = {**metadata, "title": title}
            yield {
                "doc_id": doc_id,
                "title": title,
                "text": str(text),
                "metadata": metadata,
            }

    def iter_queries(self, dataset_id: str):
        normalized, corpus_path, queries_path, qrels_path = self._ensure_files(dataset_id)
        qrels = self._read_qrels(qrels_path)
        queries = self._read_queries(queries_path)
        corpus_ids = self._read_corpus_ids(corpus_path)

        missing_docs = 0
        filtered_query_ids = []
        for query_id, query_qrels in qrels.items():
            filtered_query_ids.append(query_id)
            for doc_id in query_qrels.keys():
                if doc_id not in corpus_ids:
                    missing_docs += 1
                    logger.warning(
                        "Qrels doc_id not found in corpus: dataset=%s split=%s query_id=%s doc_id=%s",
                        normalized,
                        self.default_split,
                        query_id,
                        doc_id,
                    )
        if missing_docs:
            logger.warning(
                "BEIR qrels with missing docs: dataset=%s split=%s missing_doc_refs=%d",
                normalized,
                self.default_split,
                missing_docs,
            )
        logger.info(
            "BEIR query/qrels summary: dataset=%s queries=%d qrels=%d",
            normalized,
            len(filtered_query_ids),
            sum(len(items) for items in qrels.values()),
        )
        for query_id in filtered_query_ids:
            query_text = queries.get(query_id)
            if query_text is None:
                logger.warning(
                    "Query id present in qrels but missing in queries.jsonl: dataset=%s query_id=%s",
                    normalized,
                    query_id,
                )
                continue
            query_qrels = dict(qrels.get(query_id, {}))
            yield {
                "split": self.default_split,
                "query_id": query_id,
                "query_text": query_text,
                "qrels": query_qrels,
                "relevant_doc_ids": [doc_id for doc_id, rel in query_qrels.items() if int(rel) > 0],
            }
