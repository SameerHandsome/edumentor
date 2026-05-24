"""
Curriculum document ingestion pipeline.

Chunks raw text, computes dense + sparse vectors, and upserts into
the curriculum_docs Qdrant collection with rich metadata payloads.

Payload schema (ALL fields indexed for pre-filtering):
{
    "doc_id":       str   — stable document identifier (caller-supplied)
    "chunk_index":  int   — zero-based position within the source document
    "source":       str   — filename / URL / textbook title
    "topic_id":     str   — FK → topics.id (UUID string)
    "subtopic_id":  str   — FK → topics.id for the leaf subtopic (optional)
    "difficulty":   str   — "beginner" | "intermediate" | "advanced"
    "language":     str   — ISO 639-1 code, e.g. "en"
    "doc_type":     str   — "textbook" | "lecture" | "worksheet" | "reference"
    "grade_level":  int   — 1–12+ (curriculum grade)
    "content":      str   — raw chunk text (returned with search results)
    "created_at":   str   — ISO-8601 UTC timestamp
}

NOTE: user_id is intentionally ABSENT from curriculum_docs — these are
      shared resources accessible to all users.  Access control lives
      entirely in user_memory, where every document IS scoped by user_id.
"""

from __future__ import annotations

# ── ALL imports at the top — never split by function definitions ─────────────
import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, SparseVector

from app.core.config import settings
from app.rag.bm25 import BM25Encoder
from app.rag.collections import COLLECTION_CURRICULUM
from app.rag.embeddings import embed_batch

logger = structlog.get_logger(__name__)

# ── chunking config ──────────────────────────────────────────────────────────
CHUNK_SIZE = 512  # characters (not tokens — fast, portable)
CHUNK_OVERLAP = 64  # character overlap between consecutive chunks

_CHUNK_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")  # fixed namespace for chunk IDs


def _chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Deterministic UUID5 — same doc+index always produces the same point ID."""
    return str(uuid.uuid5(_CHUNK_NS, f"{doc_id}:{chunk_idx}"))


def chunk_text(text: str) -> list[str]:
    """
    Splits text based purely on semantic meaning.
    Adapted to handle both standard text (punctuation) and code (newlines).

    NOTE: This function is SYNCHRONOUS — it calls OllamaEmbeddings internally
    via LangChain which is blocking I/O.  Always call it with:
        chunks = await asyncio.to_thread(chunk_text, text)
    Never call it directly from an async context.
    """
    if not text.strip():
        return []

    # Connect to your local embedding model
    embeddings = OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL
    )

    # Strictly Semantic Chunker
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80,
        # Split on punctuation OR newlines (\n)
        sentence_split_regex=r"(?<=[.?!])\s+|\n+",
    )

    # Process the document
    docs = text_splitter.create_documents([text])
    chunks = [doc.page_content for doc in docs if doc.page_content.strip()]

    return chunks


# ── ingestion ────────────────────────────────────────────────────────────────


async def ingest_document(
    client: AsyncQdrantClient,
    bm25_encoder: BM25Encoder,
    *,
    text: str,
    doc_id: str,
    source: str,
    topic_id: str,
    subtopic_id: str = "",
    difficulty: str = "intermediate",
    language: str = "en",
    doc_type: str = "textbook",
    grade_level: int = 10,
) -> int:
    """
    Chunk, embed (dense + sparse), and upsert one document into curriculum_docs.

    Returns the number of chunks upserted.

    chunk_text() calls OllamaEmbeddings synchronously (blocking I/O).
    We run it in a thread pool via asyncio.to_thread so the FastAPI event
    loop is never blocked during ingestion.
    """
    # BUG FIX: chunk_text() uses OllamaEmbeddings which is synchronous/blocking.
    # Running it directly in an async function would freeze the event loop.
    # asyncio.to_thread offloads it to a thread pool worker.
    chunks = await asyncio.to_thread(chunk_text, text)
    if not chunks:
        logger.warning("ingest_skipped_empty", doc_id=doc_id)
        return 0

    logger.info("ingesting_document", doc_id=doc_id, chunks=len(chunks))

    # Dense vectors — batch for efficiency
    dense_vectors = await embed_batch(chunks)

    now_iso = datetime.now(UTC).isoformat()

    points: list[PointStruct] = []
    for chunk_idx, (chunk_text_val, dense_vec) in enumerate(zip(chunks, dense_vectors)):
        # Sparse vector from BM25
        sparse_indices, sparse_values = bm25_encoder.encode_document(chunk_text_val)

        payload: dict[str, Any] = {
            "doc_id": doc_id,
            "chunk_index": chunk_idx,
            "source": source,
            "topic_id": topic_id,
            "subtopic_id": subtopic_id,
            "difficulty": difficulty,
            "language": language,
            "doc_type": doc_type,
            "grade_level": grade_level,
            "content": chunk_text_val,
            "created_at": now_iso,
        }

        points.append(
            PointStruct(
                id=_chunk_id(doc_id, chunk_idx),
                vector={
                    "dense": dense_vec,
                    "bm25": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
                payload=payload,
            )
        )

    # Upsert in small batches of 16 to avoid ReadTimeout on high-latency
    # connections (e.g. Pakistan → us-east4 GCP Qdrant Cloud).
    # Each batch is ~16 × 768-dim vectors — well within Qdrant's payload limit.
    batch_size = 16
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await client.upsert(
            collection_name=COLLECTION_CURRICULUM,
            points=batch,
        )
        logger.info(
            "upserted_batch",
            doc_id=doc_id,
            batch=f"{i}–{i + len(batch)}",
        )

    logger.info("ingest_complete", doc_id=doc_id, total_chunks=len(chunks))
    return len(chunks)


async def delete_document(
    client: AsyncQdrantClient,
    doc_id: str,
) -> None:
    """Remove all chunks belonging to a document from curriculum_docs."""
    await client.delete(
        collection_name=COLLECTION_CURRICULUM,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )
    logger.info("document_deleted", doc_id=doc_id)
