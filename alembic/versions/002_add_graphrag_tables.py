"""Add GraphRAG tables: rag_documents, knowledge graph, literature_index.

Revision ID: 002_graphrag
Revises: 001_notifications
Create Date: 2026-07-13
"""
from alembic import op
import sqlalchemy as sa

revision = "002_graphrag"
down_revision = "001_notifications"
branch_labels = None
depends_on = None


def _array_type(dialect: str):
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import ARRAY
        return ARRAY(sa.String())
    return sa.JSON()


def _json_type(dialect: str):
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import JSONB
        return JSONB(astext_type=sa.Text())
    return sa.JSON()


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "rag_documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_name", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.String(length=1000), nullable=True),
        sa.Column("section", sa.String(length=500), nullable=True),
        sa.Column("heading", sa.String(length=500), nullable=True),
        sa.Column("heading_path", _array_type(dialect), nullable=True),
        sa.Column("structural_type", sa.String(length=50), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("parent_content", sa.Text(), nullable=True),
        sa.Column("doc_metadata", _json_type(dialect), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rag_documents_source_type", "rag_documents", ["source_type"])

    op.create_table(
        "knowledge_nodes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("aliases", _array_type(dialect), nullable=True),
        sa.Column("properties", _json_type(dialect), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_knowledge_nodes_entity_type", "knowledge_nodes", ["entity_type"])
    op.create_index("ix_knowledge_nodes_name", "knowledge_nodes", ["name"])

    op.create_table(
        "knowledge_edges",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("from_node", sa.String(length=36), nullable=False),
        sa.Column("to_node", sa.String(length=36), nullable=False),
        sa.Column("relation", sa.String(length=100), nullable=False),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("properties", _json_type(dialect), nullable=True),
        sa.ForeignKeyConstraint(["from_node"], ["knowledge_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_node"], ["knowledge_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ke_from", "knowledge_edges", ["from_node"])
    op.create_index("ke_to", "knowledge_edges", ["to_node"])

    op.create_table(
        "literature_index",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("pmid", sa.String(length=20), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("authors", _array_type(dialect), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("journal", sa.String(length=500), nullable=True),
        sa.Column("mesh_terms", _array_type(dialect), nullable=True),
        sa.Column("methods_tags", _array_type(dialect), nullable=True),
        sa.Column("disciplines", _array_type(dialect), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pmid"),
    )
    op.create_index("ix_literature_index_pmid", "literature_index", ["pmid"])

    if dialect == "postgresql":
        op.execute("ALTER TABLE rag_documents ADD COLUMN embedding vector(1536)")
        op.execute("ALTER TABLE knowledge_nodes ADD COLUMN embedding vector(1536)")
        op.execute("ALTER TABLE literature_index ADD COLUMN embedding vector(1536)")
        op.execute(
            """
            CREATE INDEX rag_docs_hnsw ON rag_documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        op.execute(
            """
            CREATE INDEX rag_docs_fts ON rag_documents
            USING gin(to_tsvector('english', content))
            """
        )
        op.execute(
            """
            CREATE INDEX lit_hnsw ON literature_index
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
    else:
        op.add_column("rag_documents", sa.Column("embedding", sa.Text(), nullable=True))
        op.add_column("knowledge_nodes", sa.Column("embedding", sa.Text(), nullable=True))
        op.add_column("literature_index", sa.Column("embedding", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS lit_hnsw")
        op.execute("DROP INDEX IF EXISTS rag_docs_fts")
        op.execute("DROP INDEX IF EXISTS rag_docs_hnsw")

    op.drop_index("ix_literature_index_pmid", table_name="literature_index")
    op.drop_table("literature_index")
    op.drop_index("ke_to", table_name="knowledge_edges")
    op.drop_index("ke_from", table_name="knowledge_edges")
    op.drop_table("knowledge_edges")
    op.drop_index("ix_knowledge_nodes_name", table_name="knowledge_nodes")
    op.drop_index("ix_knowledge_nodes_entity_type", table_name="knowledge_nodes")
    op.drop_table("knowledge_nodes")
    op.drop_index("ix_rag_documents_source_type", table_name="rag_documents")
    op.drop_table("rag_documents")
