from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from audit import audit_print
from infrastructure.api.deps import build_services
from infrastructure.config import get_settings
from infrastructure.db.session import get_session_factory


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class IndexJobState:
    dataset_id: str
    status: str = "idle"  # idle|running|completed|failed
    started_at: datetime | None = None
    finished_at: datetime | None = None
    indexed_count: int = 0
    total_hint: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "updated_at": self.updated_at.isoformat(),
            "indexed_count": self.indexed_count,
            "total_hint": self.total_hint,
            "error": self.error,
            "result": self.result,
        }


class IndexJobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IndexJobState] = {}

    def get(self, dataset_id: str) -> IndexJobState:
        with self._lock:
            state = self._jobs.get(dataset_id)
            if state is None:
                return IndexJobState(dataset_id=dataset_id)
            return IndexJobState(**asdict(state))

    def start(self, dataset_id: str, force_reindex: bool = False) -> IndexJobState:
        with self._lock:
            current = self._jobs.get(dataset_id)
            if current and current.status == "running":
                audit_print("index.job.start.reused", dataset_id=dataset_id, force_reindex=force_reindex)
                return IndexJobState(**asdict(current))
            state = IndexJobState(
                dataset_id=dataset_id,
                status="running",
                started_at=_utcnow(),
                finished_at=None,
                indexed_count=0,
                error=None,
                result=None,
                updated_at=_utcnow(),
            )
            self._jobs[dataset_id] = state
        audit_print("index.job.start.created", dataset_id=dataset_id, force_reindex=force_reindex)
        thread = threading.Thread(
            target=self._run_job,
            args=(dataset_id, force_reindex),
            daemon=True,
            name=f"index-{dataset_id}",
        )
        thread.start()
        return self.get(dataset_id)

    def _update(self, dataset_id: str, mutator: Callable[[IndexJobState], None]) -> None:
        with self._lock:
            state = self._jobs.get(dataset_id)
            if state is None:
                state = IndexJobState(dataset_id=dataset_id)
                self._jobs[dataset_id] = state
            mutator(state)
            state.updated_at = _utcnow()

    def _run_job(self, dataset_id: str, force_reindex: bool) -> None:
        settings = get_settings()
        session = get_session_factory(settings)()
        try:
            audit_print("index.job.run.start", dataset_id=dataset_id, force_reindex=force_reindex)
            services = build_services(session, settings)
            if not force_reindex:
                snap = services.dataset_snapshots.get(dataset_id)
                docs_count = services.documents.count_by_dataset(dataset_id)
                encoders_fitted = (
                    services.index_dataset.classical_encoder.is_fitted
                    and services.index_dataset.quantum_encoder.is_fitted
                    and services.index_dataset.statistical_encoder.is_fitted
                )
                if snap and docs_count > 0 and snap.document_count == docs_count and encoders_fitted:
                    def _mark_skipped(s: IndexJobState) -> None:
                        s.status = "completed"
                        s.finished_at = _utcnow()
                        s.result = {
                            "dataset_id": dataset_id,
                            "indexed_count": docs_count,
                            "skipped": True,
                            "reason": "already_indexed",
                            "snapshot_document_count": snap.document_count,
                            "snapshot_query_count": snap.query_count,
                        }
                        s.indexed_count = docs_count
                        s.error = None

                    self._update(dataset_id, _mark_skipped)
                    audit_print("index.job.run.skipped", dataset_id=dataset_id, docs_count=docs_count, reason="already_indexed")
                    return

            dataset_meta = services.index_dataset.datasets.get_dataset(dataset_id)
            total_hint = dataset_meta.get("document_count") if isinstance(dataset_meta, dict) else None
            self._update(dataset_id, lambda s: setattr(s, "total_hint", total_hint))
            audit_print("index.job.run.dataset_loaded", dataset_id=dataset_id, total_hint=total_hint)

            def progress_cb(indexed_count: int) -> None:
                self._update(dataset_id, lambda s: setattr(s, "indexed_count", indexed_count))
                audit_print("index.job.run.progress", dataset_id=dataset_id, indexed_count=indexed_count, total_hint=total_hint)

            result = services.index_dataset.execute(dataset_id, progress_callback=progress_cb)
            def _mark_completed(s: IndexJobState) -> None:
                s.status = "completed"
                s.finished_at = _utcnow()
                s.result = result
                s.indexed_count = int(result.get("indexed_count", s.indexed_count))
                s.error = None

            self._update(dataset_id, _mark_completed)
            audit_print("index.job.run.completed", dataset_id=dataset_id, result=result)
        except Exception as exc:
            def _mark_failed(s: IndexJobState) -> None:
                s.status = "failed"
                s.finished_at = _utcnow()
                s.error = str(exc)

            self._update(dataset_id, _mark_failed)
            audit_print("index.job.run.failed", dataset_id=dataset_id, error=str(exc))
        finally:
            session.close()


index_job_registry = IndexJobRegistry()
