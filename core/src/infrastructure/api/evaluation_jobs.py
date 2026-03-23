"""Thread-safe registry for batch evaluation jobs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class EvaluationProgress:
    current_query: int = 0
    total_queries: int = 0
    current_pipeline: str = ""
    completed_pipelines: list[str] = field(default_factory=list)


class EvaluationJobRegistry:
    """Thread-safe singleton registry for async batch evaluation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: str = "idle"
        self._result: dict | None = None
        self._progress = EvaluationProgress()
        self._error: str | None = None
        self._started_at: float | None = None

    def start(self, evaluate_fn, *, dataset_id: str, pipelines: list[str], k: int, query_count: int) -> bool:
        with self._lock:
            if self._status == "running":
                return False
            self._status = "running"
            self._result = None
            self._error = None
            self._started_at = time.time()
            self._progress = EvaluationProgress(total_queries=query_count)

        thread = threading.Thread(
            target=self._run,
            args=(evaluate_fn, dataset_id, pipelines, k),
            daemon=True,
        )
        thread.start()
        return True

    def _run(self, evaluate_fn, dataset_id: str, pipelines: list[str], k: int) -> None:
        try:
            result = evaluate_fn(
                dataset=dataset_id,
                pipeline="compare",
                k=k,
                progress_callback=self._update_progress,
            )
            with self._lock:
                self._result = result
                self._status = "completed"
        except Exception as e:
            with self._lock:
                self._error = str(e)
                self._status = "failed"

    def _update_progress(self, current_query: int, current_pipeline: str, completed_pipelines: list[str]) -> None:
        with self._lock:
            self._progress.current_query = current_query
            self._progress.current_pipeline = current_pipeline
            self._progress.completed_pipelines = list(completed_pipelines)

    @property
    def status(self) -> dict:
        with self._lock:
            elapsed = (time.time() - self._started_at) if self._started_at else 0
            return {
                "status": self._status,
                "progress": {
                    "current_query": self._progress.current_query,
                    "total_queries": self._progress.total_queries,
                    "current_pipeline": self._progress.current_pipeline,
                    "completed_pipelines": list(self._progress.completed_pipelines),
                },
                "elapsed_seconds": round(elapsed, 1),
                "result": self._result if self._status == "completed" else None,
                "error": self._error if self._status == "failed" else None,
            }


evaluation_job_registry = EvaluationJobRegistry()
