"""
User document ingestion — private per-user uploaded files.

Every chunk is stamped with user_id so retrieval is always scoped
to the uploading user. User A's documents are never visible to User B.

Deterministic chunk IDs (uuid5) make re-ingestion of the same file
idempotent — re-uploading overwrites rather than duplicating.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, SparseVector

from app.rag.bm25 import BM25Encoder
from app.rag.collections import COLLECTION_USER_DOCS
from app.rag.embeddings import embed_batch
from app.rag.ingestion import chunk_text  # reuse same chunking logic

logger = structlog.get_logger(__name__)

_CHUNK_NS = uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")  # namespace for user_docs chunk IDs


def _chunk_id(user_id: str, doc_id: str, chunk_idx: int) -> str:
    """Deterministic UUID5 — same user+doc+index always produces the same point ID."""
    return str(uuid.uuid5(_CHUNK_NS, f"{user_id}:{doc_id}:{chunk_idx}"))


async def ingest_user_document(
    client: AsyncQdrantClient,
    bm25_encoder: BM25Encoder,
    *,
    user_id: str,
    doc_id: str,
    filename: str,
    text: str,
    session_id: str = "",  # scopes retrieval to the uploading session
) -> int:
    """
    Chunk, embed, and upsert one user-uploaded document into user_docs.
    user_id is stamped on every chunk — the isolation boundary.
    Returns the number of chunks upserted.
    """
    # BUG FIX: chunk_text() calls OllamaEmbeddings synchronously (blocking I/O).
    # asyncio.to_thread offloads it so the event loop is never blocked.
    chunks = await asyncio.to_thread(chunk_text, text)
    if not chunks:
        logger.warning("user_doc_ingest_skipped_empty", doc_id=doc_id, user_id=user_id)
        return 0

    logger.info("ingesting_user_doc", doc_id=doc_id, user_id=user_id, chunks=len(chunks))

    dense_vectors = await embed_batch(chunks)
    now_iso = datetime.now(UTC).isoformat()

    points: list[PointStruct] = []
    for chunk_idx, (chunk_text_val, dense_vec) in enumerate(zip(chunks, dense_vectors)):
        sparse_indices, sparse_values = bm25_encoder.encode_document(chunk_text_val)
        points.append(
            PointStruct(
                id=_chunk_id(user_id, doc_id, chunk_idx),
                vector={
                    "dense": dense_vec,
                    "bm25": SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "user_id": user_id,  # isolation boundary — always present
                    "doc_id": doc_id,
                    "session_id": session_id,  # scopes retrieval to uploading session
                    "filename": filename,
                    "chunk_index": chunk_idx,
                    "content": chunk_text_val,
                    "created_at": now_iso,
                },
            )
        )

    batch_size = 64
    for i in range(0, len(points), batch_size):
        await client.upsert(
            collection_name=COLLECTION_USER_DOCS,
            points=points[i : i + batch_size],
        )

    logger.info(
        "user_doc_ingest_complete", doc_id=doc_id, user_id=user_id, total_chunks=len(chunks)
    )
    return len(chunks)


async def delete_user_document(
    client: AsyncQdrantClient,
    *,
    user_id: str,
    doc_id: str,
) -> None:
    """
    Delete all chunks of a user's document.
    Both user_id and doc_id are required — prevents cross-user deletes.
    """
    await client.delete(
        collection_name=COLLECTION_USER_DOCS,
        points_selector=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            ]
        ),
    )
    logger.info("user_doc_deleted", doc_id=doc_id, user_id=user_id)
