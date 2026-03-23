"""Populate ideal_answer for all 39 valid BEIR trec-covid queries.

Revision ID: 003
Revises: 002
Create Date: 2026-03-22
"""

from alembic import op
import sqlalchemy as sa

from domain.ideal_answers import IDEAL_ANSWERS

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for qid, answer in IDEAL_ANSWERS.items():
        op.execute(
            sa.text(
                "UPDATE queries SET ideal_answer = :answer "
                "WHERE dataset = 'beir/trec-covid' AND query_id = :qid"
            ).bindparams(answer=answer, qid=qid)
        )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE queries SET ideal_answer = NULL "
            "WHERE dataset = 'beir/trec-covid'"
        )
    )
