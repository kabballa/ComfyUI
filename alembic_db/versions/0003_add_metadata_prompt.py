"""
Add system_metadata and prompt_id columns to asset_references.

Revision ID: 0003_add_metadata_prompt
Revises: 0002_merge_to_asset_references
Create Date: 2026-03-09
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_add_metadata_prompt"
down_revision = "0002_merge_to_asset_references"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.add_column(
            sa.Column("system_metadata", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("prompt_id", sa.String(length=36), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.drop_column("prompt_id")
        batch_op.drop_column("system_metadata")
