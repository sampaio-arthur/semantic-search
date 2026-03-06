from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.config import Settings, get_settings
from infrastructure.db.models import Base

_engine = None
_SessionLocal = None


def get_engine(settings: Settings | None = None):
    global _engine
    if _engine is None:
        cfg = settings or get_settings()
        _engine = create_engine(cfg.database_url, future=True, pool_pre_ping=True)
    return _engine


def get_session_factory(settings: Settings | None = None):
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(settings), autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)
    return _SessionLocal


def init_db(settings: Settings | None = None) -> None:
    engine = get_engine(settings)
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Only PostgreSQL is supported.")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS title TEXT"))
        # Migrate vector columns to shared 64-dim space and add statistical_vector.
        # Dropping old columns invalidates existing indexed vectors — re-indexing required.
        conn.execute(text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='documents' AND column_name='embedding_vector'
                ) THEN
                    ALTER TABLE documents DROP COLUMN embedding_vector;
                END IF;
                ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_vector vector(64);

                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='documents' AND column_name='quantum_vector'
                ) THEN
                    ALTER TABLE documents DROP COLUMN quantum_vector;
                END IF;
                ALTER TABLE documents ADD COLUMN IF NOT EXISTS quantum_vector vector(64);

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='documents' AND column_name='statistical_vector'
                ) THEN
                    ALTER TABLE documents ADD COLUMN statistical_vector vector(64);
                END IF;
            END $$;
        """))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id SERIAL PRIMARY KEY,
                    dataset VARCHAR(100) NOT NULL,
                    split VARCHAR(32) NOT NULL DEFAULT 'test',
                    query_id VARCHAR(255) NOT NULL,
                    query_text TEXT NOT NULL,
                    user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_queries_dataset_split_query_id UNIQUE (dataset, split, query_id)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS qrels (
                    id SERIAL PRIMARY KEY,
                    dataset VARCHAR(100) NOT NULL,
                    split VARCHAR(32) NOT NULL DEFAULT 'test',
                    query_id VARCHAR(255) NOT NULL,
                    doc_id VARCHAR(255) NOT NULL,
                    relevance INTEGER NOT NULL DEFAULT 0,
                    CONSTRAINT uq_qrels_dataset_split_query_doc UNIQUE (dataset, split, query_id, doc_id)
                )
                """
            )
        )


def db_session() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
