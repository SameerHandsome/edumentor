"""
Hybrid retriever — dense (nomic-embed-text) + sparse (BM25) fusion.

Retrieval flow per collection:
┌──────────────────────────────────────────────────────────┐
│ curriculum_docs                                          │
│  1. HyDE transform     → hypothetical dense vector      │
│  2. BM25 encode query  → sparse vector                  │
│  3. Qdrant hybrid search with pre-filter                 │
│  4. Merge + dedup by point_id (RRF score fusion)        │
│  5. Cross-encoder rerank → top-3 chunks                 │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│ user_memory                                              │
│  1. Multi-query transform → N dense vectors             │
│  2. BM25 encode query     → sparse vector               │
│  3. Qdrant hybrid search with MANDATORY user_id filter  │
│  4. Merge + dedup (RRF)                                 │
│  5. Cross-encoder rerank → top-3 memories               │
└──────────────────────────────────────────────────────────┘

RRF (Reciprocal Rank Fusion) combines dense and sparse result lists
without requiring score normalization.
"""

from __future__ import annotations

import uuid
from datetime import UTC
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    NamedSparseVector,
    NamedVector,
    SparseVector,
)

from app.rag.bm25 import BM25Encoder
from app.rag.collections import COLLECTION_CURRICULUM, COLLECTION_USER_MEMORY
from app.rag.embeddings import embed_batch, embed_text  # noqa: F401 (embed_batch used in tests)
from app.rag.filters import build_curriculum_filter, build_user_memory_filter
from app.rag.hyde import hyde_transform
from app.rag.multi_query import multi_query_transform
from app.rag.reranker import RankedResult, rerank

logger = structlog.get_logger(__name__)

# How many candidates to fetch from Qdrant before reranking
_PREFETCH_LIMIT = 20
# Final results returned to callers (before reranker trims to top_k)
_DEFAULT_TOP_K = 3
# RRF constant — 60 is the standard default
_RRF_K = 60


# ── RRF fusion ───────────────────────────────────────────────────────────────


def _rrf_fuse(
    *result_lists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    Each list item must have a "point_id" key.
    Returns a single deduplicated list sorted by fused RRF score desc.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            pid = item["point_id"]
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            payloads[pid] = item["payload"]

    fused = [
        {"point_id": pid, "payload": payloads[pid], "rrf_score": score}
        for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused


def _qdrant_hits_to_dicts(hits) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    """Convert Qdrant ScoredPoint objects to plain dicts for RRF."""
    return [{"point_id": str(hit.id), "payload": hit.payload or {}} for hit in hits]


# ── curriculum_docs retrieval ─────────────────────────────────────────────────


async def retrieve_curriculum(
    client: AsyncQdrantClient,
    bm25_encoder: BM25Encoder,
    query: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    topic_id: str | None = None,
    subtopic_ids: list[str] | None = None,
    difficulty: str | None = None,
    difficulties: list[str] | None = None,
    language: str | None = None,  # ← was "en"; None = no language filter
    doc_type: str | None = None,
    doc_types: list[str] | None = None,
    grade_level_min: int | None = None,
    grade_level_max: int | None = None,
    doc_ids: list[str] | None = None,
    source: str | None = None,
) -> list[RankedResult]:
    """
    Hybrid-retrieve then rerank chunks from curriculum_docs.

    Returns up to `top_k` RankedResult objects, richest first.
    """
    # 1. Query transforms
    dense_vec = await hyde_transform(query)
    sparse_indices, sparse_values = bm25_encoder.encode_query(query)

    # 2. Build optional pre-filter
    pre_filter: Filter | None = build_curriculum_filter(
        topic_id=topic_id,
        subtopic_ids=subtopic_ids,
        difficulty=difficulty,
        difficulties=difficulties,
        language=language,
        doc_type=doc_type,
        doc_types=doc_types,
        grade_level_min=grade_level_min,
        grade_level_max=grade_level_max,
        doc_ids=doc_ids,
        source=source,
    )

    # 3. Dense search
    dense_hits = await client.search(
        collection_name=COLLECTION_CURRICULUM,
        query_vector=NamedVector(name="dense", vector=dense_vec),
        query_filter=pre_filter,
        limit=_PREFETCH_LIMIT,
        with_payload=True,
    )

    # 4. Sparse (BM25) search
    sparse_hits = await client.search(
        collection_name=COLLECTION_CURRICULUM,
        query_vector=NamedSparseVector(
            name="bm25",
            vector=SparseVector(indices=sparse_indices, values=sparse_values),
        ),
        query_filter=pre_filter,
        limit=_PREFETCH_LIMIT,
        with_payload=True,
    )

    # 5. RRF fusion
    candidates = _rrf_fuse(
        _qdrant_hits_to_dicts(dense_hits),
        _qdrant_hits_to_dicts(sparse_hits),
    )

    logger.debug(
        "curriculum_candidates",
        dense=len(dense_hits),
        sparse=len(sparse_hits),
        fused=len(candidates),
    )

    # 6. Cross-encoder rerank
    return rerank(query, candidates, top_k=top_k)


# ── user_memory retrieval ─────────────────────────────────────────────────────


async def retrieve_user_memory(
    client: AsyncQdrantClient,
    bm25_encoder: BM25Encoder,
    query: str,
    user_id: str,  # REQUIRED — enforces per-user isolation
    *,
    top_k: int = _DEFAULT_TOP_K,
    memory_type: str | None = None,
    memory_types: list[str] | None = None,
    topic: str | None = None,
    session_id: str | None = None,
) -> list[RankedResult]:
    """
    Hybrid-retrieve then rerank memories from user_memory.

    user_id is always injected into the filter — one user can never
    see another user's memory entries.

    Returns up to `top_k` RankedResult objects.
    """
    # 1. Multi-query transform (original + N variants)
    dense_vecs = await multi_query_transform(query)

    # 2. MANDATORY user-scoped filter
    mem_filter: Filter = build_user_memory_filter(
        user_id=user_id,
        memory_type=memory_type,
        memory_types=memory_types,
        topic=topic,
        session_id=session_id,
    )

    # 3. Dense search — one pass per variant, collect into separate lists for RRF
    dense_ranked_lists = []
    for dense_vec in dense_vecs:
        hits = await client.search(
            collection_name=COLLECTION_USER_MEMORY,
            query_vector=NamedVector(name="dense", vector=dense_vec),
            query_filter=mem_filter,
            limit=_PREFETCH_LIMIT,
            with_payload=True,
        )
        dense_ranked_lists.append(_qdrant_hits_to_dicts(hits))

    # 4. Sparse search with original query
    sparse_indices, sparse_values = bm25_encoder.encode_query(query)
    sparse_hits = await client.search(
        collection_name=COLLECTION_USER_MEMORY,
        query_vector=NamedSparseVector(
            name="bm25",
            vector=SparseVector(indices=sparse_indices, values=sparse_values),
        ),
        query_filter=mem_filter,
        limit=_PREFETCH_LIMIT,
        with_payload=True,
    )

    # 5. RRF fusion — each dense pass + sparse as separate ranked lists
    candidates = _rrf_fuse(
        *dense_ranked_lists,
        _qdrant_hits_to_dicts(sparse_hits),
    )

    logger.debug(
        "memory_candidates",
        dense_queries=len(dense_vecs),
        fused=len(candidates),
        user_id=user_id,
    )

    # 6. Cross-encoder rerank
    return rerank(query, candidates, top_k=top_k)


# ── user_memory upsert (called by Celery session_summarize task) ─────────────


async def upsert_user_memory(
    client: AsyncQdrantClient,
    *,
    user_id: str,
    doc_id: str,
    memory_type: str,  # "session_summary" | "preference" | "weak_area"
    content: str,
    topic: str = "",
    session_id: str = "",
) -> None:
    """
    Embed and upsert a single memory entry into user_memory.
    Uses doc_id as the Qdrant point ID for stable upsert/overwrite semantics.
    Calling this repeatedly with the same doc_id safely overwrites the entry.

    BM25 sparse vector is intentionally empty for session memories.
    The Celery worker that calls this has no shared BM25 encoder with the
    API worker (which is fitted on the full corpus). Any sparse vector
    produced by a single-doc encoder would use an incompatible vocabulary,
    causing silent zero-hit sparse retrieval. Dense-only is correct here.
    """
    from datetime import datetime

    from qdrant_client.models import PointStruct

    dense_vec = await embed_text(content)
    # Empty sparse vector — dense-only retrieval for session memories
    sparse_indices: list = []
    sparse_values: list = []

    payload = {
        "user_id": user_id,
        "doc_id": doc_id,
        "type": memory_type,
        "content": content,
        "topic": topic,
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat(),
    }

    # Qdrant point IDs must be a pure UUID or unsigned int.
    # doc_id may be prefixed (e.g. "session_summary_<uuid>"), so we
    # convert it deterministically via uuid5 -- same input = same UUID.
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

    await client.upsert(
        collection_name=COLLECTION_USER_MEMORY,
        points=[
            PointStruct(
                id=point_id,  # deterministic UUID → safe to call repeatedly
                vector={
                    "dense": dense_vec,
                    "bm25": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
                payload=payload,
            )
        ],
    )
    logger.info(
        "user_memory_upserted",
        user_id=user_id,
        doc_id=doc_id,
        memory_type=memory_type,
    )


# ── user_docs retrieval ───────────────────────────────────────────────────────


async def retrieve_user_docs(
    client: AsyncQdrantClient,
    bm25_encoder: BM25Encoder,
    query: str,
    user_id: str,  # REQUIRED — enforces per-user isolation
    *,
    top_k: int = _DEFAULT_TOP_K,
    doc_id: str | None = None,
    filename: str | None = None,
    session_id: str | None = None,  # REQUIRED in practice — scopes to current session only
) -> list[RankedResult]:
    """
    Hybrid-retrieve then rerank chunks from the user's own uploaded documents.

    user_id is always injected into the filter — one user can never
    retrieve another user's uploaded documents.

    session_id should always be passed — without it, docs from previous
    sessions bleed into the current one.
    """
    from app.rag.collections import COLLECTION_USER_DOCS
    from app.rag.filters import build_user_docs_filter

    dense_vec = await embed_text(query)
    sparse_indices, sparse_values = bm25_encoder.encode_query(query)

    doc_filter = build_user_docs_filter(
        user_id, doc_id=doc_id, filename=filename, session_id=session_id
    )

    dense_hits = await client.search(
        collection_name=COLLECTION_USER_DOCS,
        query_vector=NamedVector(name="dense", vector=dense_vec),
        query_filter=doc_filter,
        limit=_PREFETCH_LIMIT,
        with_payload=True,
    )

    sparse_hits = await client.search(
        collection_name=COLLECTION_USER_DOCS,
        query_vector=NamedSparseVector(
            name="bm25",
            vector=SparseVector(indices=sparse_indices, values=sparse_values),
        ),
        query_filter=doc_filter,
        limit=_PREFETCH_LIMIT,
        with_payload=True,
    )

    candidates = _rrf_fuse(
        _qdrant_hits_to_dicts(dense_hits),
        _qdrant_hits_to_dicts(sparse_hits),
    )

    logger.debug(
        "user_docs_candidates",
        dense=len(dense_hits),
        sparse=len(sparse_hits),
        fused=len(candidates),
        user_id=user_id,
    )

    return rerank(query, candidates, top_k=top_k)
