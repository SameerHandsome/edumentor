"""Celery task — async user document ingestion."""

from __future__ import annotations

import asyncio
import json

import structlog

from app.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

DOC_STATUS_KEY = "doc:status:{doc_id}"
DOC_STATUS_TTL = 3600


@celery_app.task(
    name="app.tasks.doc_tasks.ingest_document",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    queue="session",
    ignore_result=True,  # we track status via our own Redis key, not Celery backend
)
def ingest_document(
    self, doc_id: str, user_id: str, filename: str, text: str, session_id: str = ""
) -> dict:
    """
    Chunk, embed, and upsert a user document in the background.
    Sets doc:status:{doc_id} = "processing" → "ready" | "failed" in Redis.
    """

    async def _run() -> dict:
        from app.core.redis_client import redis_client
        from app.rag.bm25 import BM25Encoder
        from app.rag.collections import get_qdrant_client
        from app.rag.user_docs_ingestion import ingest_user_document

        status_key = DOC_STATUS_KEY.format(doc_id=doc_id)

        try:
            client = await get_qdrant_client()
            encoder = BM25Encoder()
            encoder.fit([text])

            chunk_count = await ingest_user_document(
                client,
                encoder,
                user_id=user_id,
                doc_id=doc_id,
                filename=filename,
                text=text,
                session_id=session_id,
            )

            async with redis_client() as r:
                await r.setex(
                    status_key,
                    DOC_STATUS_TTL,
                    json.dumps({"status": "ready", "chunks": chunk_count}),
                )
            logger.info("doc_ingest_task_done", doc_id=doc_id, chunks=chunk_count)
            return {"status": "ready", "chunks": chunk_count}

        except Exception as exc:
            async with redis_client() as r:
                await r.setex(
                    status_key, DOC_STATUS_TTL, json.dumps({"status": "failed", "error": str(exc)})
                )
            logger.error("doc_ingest_task_failed", doc_id=doc_id, error=str(exc))
            raise

    return asyncio.run(_run())
