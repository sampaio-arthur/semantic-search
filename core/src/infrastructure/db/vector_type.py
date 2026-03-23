from __future__ import annotations

from typing import Any

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator

try:
    from pgvector.sqlalchemy import Vector as PgVector  # type: ignore
except Exception:  # pragma: no cover
    PgVector = None


class VectorType(TypeDecorator):
    impl = JSON
    cache_ok = True
    # Expose pgvector SQLAlchemy comparator methods (e.g., cosine_distance)
    # on ORM columns that use this TypeDecorator.
    comparator_factory = PgVector.comparator_factory if PgVector is not None else TypeDecorator.Comparator

    def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name != "postgresql":
            raise RuntimeError("VectorType requires PostgreSQL with pgvector.")
        if PgVector is None:
            raise RuntimeError("pgvector.sqlalchemy is required for PostgreSQL vector columns.")
        return dialect.type_descriptor(PgVector(self.dim))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return [float(x) for x in value]

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return [float(x) for x in value]
