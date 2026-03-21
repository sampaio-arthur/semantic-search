"""Remove excluded queries and their qrels from the database.

Revision ID: 002
Revises: 001
Create Date: 2026-03-19

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

EXCLUDED_TEXTS = [
    "what causes death from Covid-19?",
    "what types of rapid testing for Covid-19 have been developed?",
    "how has COVID-19 affected Canada",
    "what are the guidelines for triaging patients infected with coronavirus?",
    "what are the transmission routes of coronavirus?",
    "are there any clinical trials available for the coronavirus",
    "which biomarkers predict the severe clinical course of 2019-nCOV infection?",
    "what is known about those infected with Covid-19 but are asymptomatic?",
    "Does SARS-CoV-2 have any subtypes, and if so what are they?",
    "How has the COVID-19 pandemic impacted violence in society, including violent crimes?",
    "what are the health outcomes for children who contract COVID-19?",
]

DATASET = "beir/trec-covid"


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Find query_ids for excluded queries
    result = conn.execute(
        sa.text(
            "SELECT query_id FROM queries WHERE dataset = :dataset AND query_text = ANY(:texts)"
        ),
        {"dataset": DATASET, "texts": EXCLUDED_TEXTS},
    )
    query_ids = [row[0] for row in result]

    if not query_ids:
        return

    # 2. Delete qrels for excluded queries
    qrels_result = conn.execute(
        sa.text(
            "DELETE FROM qrels WHERE dataset = :dataset AND query_id = ANY(:ids)"
        ),
        {"dataset": DATASET, "ids": query_ids},
    )
    print(f"[MIGRATION 002] Deleted {qrels_result.rowcount} qrels rows")

    # 3. Delete the excluded queries
    queries_result = conn.execute(
        sa.text(
            "DELETE FROM queries WHERE dataset = :dataset AND query_id = ANY(:ids)"
        ),
        {"dataset": DATASET, "ids": query_ids},
    )
    print(f"[MIGRATION 002] Deleted {queries_result.rowcount} queries rows")

    # 4. Update dataset snapshot counts
    conn.execute(
        sa.text("""
            UPDATE dataset_snapshots
            SET query_count = (SELECT COUNT(*) FROM queries WHERE dataset = :dataset),
                updated_at = NOW()
            WHERE dataset_id = :dataset
        """),
        {"dataset": DATASET},
    )
    print(f"[MIGRATION 002] Updated dataset_snapshots for {DATASET}")


def downgrade() -> None:
    # Re-indexing the dataset will restore the queries
    pass
