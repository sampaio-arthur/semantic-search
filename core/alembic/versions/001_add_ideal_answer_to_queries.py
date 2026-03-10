"""Add ideal_answer column to queries table.

Revision ID: 001
Revises:
Create Date: 2026-03-10

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("queries", sa.Column("ideal_answer", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("queries", "ideal_answer")
