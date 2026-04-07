"""Populate ideal_answer for all 39 valid BEIR trec-covid queries.

NOTE: This migration is superseded by 004_remove_ideal_answer_column which
drops the ideal_answer column entirely. The upgrade/downgrade are now no-ops.

Revision ID: 003
Revises: 002
Create Date: 2026-03-22
"""

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
