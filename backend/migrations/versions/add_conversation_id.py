"""Add conversation_id to query_history

Revision ID: add_conversation_id
Revises: initial_migration
Create Date: 2023-07-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_conversation_id'
down_revision = 'initial_migration'
branch_labels = None
depends_on = None


def upgrade():
    # Add conversation_id column to query_history table
    op.add_column('query_history', sa.Column('conversation_id', sa.Integer(), nullable=True))
    
    # Update existing records to set conversation_id = id
    op.execute("""
    UPDATE query_history
    SET conversation_id = id
    WHERE conversation_id IS NULL
    """)


def downgrade():
    # Remove conversation_id column from query_history table
    op.drop_column('query_history', 'conversation_id')
