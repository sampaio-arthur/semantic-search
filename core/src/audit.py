from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any


def preview_text(text: str | None, limit: int = 120) -> str | None:
    if text is None:
        return None
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def preview_vector(vector: list[float] | None, limit: int = 6) -> dict[str, Any] | None:
    if vector is None:
        return None
    head = [round(float(value), 6) for value in vector[:limit]]
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    return {
        "dim": len(vector),
        "head": head,
        "norm": round(norm, 6),
    }


def preview_results(results: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in results[:limit]:
        items.append(
            {
                "doc_id": getattr(item, "doc_id", None),
                "score": round(float(getattr(item, "score", 0.0)), 6),
                "text": preview_text(getattr(item, "text", None), limit=80),
            }
        )
    return items


def audit_print(event: str, **payload: Any) -> None:
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **payload,
    }
    print("[AUDIT] " + json.dumps(record, ensure_ascii=False, default=str), flush=True)


def category_log(category: str, _extra: dict[str, Any] | None = None, **payload: Any) -> None:
    """Emit a bracketed category log line as required by spec.

    Use keyword args for simple identifiers:
        category_log("BASE", embedding_dim=384)
        → [BASE] embedding_dim=384

    Use _extra dict for keys that are not valid Python identifiers (e.g. nDCG@10):
        category_log("METRICS", _extra={"pipeline": "classical", "nDCG@10": 0.42})
        → [METRICS] pipeline=classical nDCG@10=0.42
    """
    combined: dict[str, Any] = {**payload, **(_extra or {})}
    parts = " ".join(f"{k}={v}" for k, v in combined.items())
    print(f"[{category}] {parts}", flush=True)
