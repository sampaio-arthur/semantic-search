"""Remove ideal_answer column from queries table.

Revision ID: 004
Revises: 003
Create Date: 2026-04-06

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='queries' AND column_name='ideal_answer'"
        )
    ).fetchone()
    if result:
        op.drop_column("queries", "ideal_answer")


def downgrade() -> None:
    pass
