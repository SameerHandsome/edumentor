"""
Qdrant collection definitions and initialization.
Creates curriculum_docs and user_memory collections with proper vector configs.
"""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import settings

logger = structlog.get_logger(__name__)

# nomic-embed-text produces 768-dim vectors
DENSE_DIM = 768

COLLECTION_CURRICULUM = "curriculum_docs"
COLLECTION_USER_MEMORY = "user_memory"
COLLECTION_USER_DOCS = "user_docs"


_qdrant_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    """
    Return a singleton AsyncQdrantClient pointed at Qdrant Cloud.

    The client is created once and reused for the process lifetime so we
    don't open a new HTTPS connection pool on every call.

    Qdrant Cloud requires:
      - url=  (full https:// URL including port 6333)
      - api_key= (from the Qdrant Cloud console)

    Do NOT use host= + port= for Cloud — that path skips TLS and will
    get a connection refused or silent auth failure.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=120,  # seconds — raised from default 5s for high-latency connections
        )
        logger.info("qdrant_client_created", url=settings.QDRANT_URL)
    return _qdrant_client


async def init_collections(client: AsyncQdrantClient) -> None:
    """
    Idempotently create both Qdrant collections if they don't exist.
    Both collections use:
      - dense vector  : nomic-embed-text (768-dim, Cosine)
      - sparse vector : BM25 compatible (named 'bm25')
    """
    existing = {c.name for c in (await client.get_collections()).collections}

    # ── curriculum_docs ──────────────────────────────────────────────────────
    if COLLECTION_CURRICULUM not in existing:
        await client.create_collection(
            collection_name=COLLECTION_CURRICULUM,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )
        # Create payload indexes for fast metadata filtering
        for field, schema in [
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("source", PayloadSchemaType.KEYWORD),
            ("topic_id", PayloadSchemaType.KEYWORD),
            ("subtopic_id", PayloadSchemaType.KEYWORD),
            ("difficulty", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("grade_level", PayloadSchemaType.INTEGER),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("created_at", PayloadSchemaType.DATETIME),
        ]:
            await client.create_payload_index(
                collection_name=COLLECTION_CURRICULUM,
                field_name=field,
                field_schema=schema,
            )
        logger.info("created_collection", name=COLLECTION_CURRICULUM)
    else:
        logger.info("collection_exists", name=COLLECTION_CURRICULUM)

    # ── user_memory ───────────────────────────────────────────────────────────
    if COLLECTION_USER_MEMORY not in existing:
        await client.create_collection(
            collection_name=COLLECTION_USER_MEMORY,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )
        # CRITICAL: user_id index ensures per-user isolation in every query
        for field, schema in [
            ("user_id", PayloadSchemaType.KEYWORD),  # PRIMARY isolation key
            ("doc_id", PayloadSchemaType.KEYWORD),  # dedup / delete by id
            ("type", PayloadSchemaType.KEYWORD),  # session_summary | preference | weak_area
            ("topic", PayloadSchemaType.KEYWORD),
            ("session_id", PayloadSchemaType.KEYWORD),
            ("created_at", PayloadSchemaType.DATETIME),
        ]:
            await client.create_payload_index(
                collection_name=COLLECTION_USER_MEMORY,
                field_name=field,
                field_schema=schema,
            )
        logger.info("created_collection", name=COLLECTION_USER_MEMORY)
    else:
        logger.info("collection_exists", name=COLLECTION_USER_MEMORY)

    # ── user_docs ─────────────────────────────────────────────────────────────
    # Private per-user uploaded documents. user_id is the isolation boundary.
    if COLLECTION_USER_DOCS not in existing:
        await client.create_collection(
            collection_name=COLLECTION_USER_DOCS,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )
        for field, schema in [
            ("user_id", PayloadSchemaType.KEYWORD),  # PRIMARY isolation key
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("filename", PayloadSchemaType.KEYWORD),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("created_at", PayloadSchemaType.DATETIME),
        ]:
            await client.create_payload_index(
                collection_name=COLLECTION_USER_DOCS,
                field_name=field,
                field_schema=schema,
            )
        logger.info("created_collection", name=COLLECTION_USER_DOCS)
    else:
        logger.info("collection_exists", name=COLLECTION_USER_DOCS)
