"""Celery session summarization — runs at session end, upserts user_memory."""

from __future__ import annotations

import asyncio
import uuid

import structlog

from app.tasks.celery_app import celery_app
from app.tasks.voice_tasks import _reset_singletons

logger = structlog.get_logger(__name__)


@celery_app.task(
    bind=True, name="app.tasks.session_tasks.summarize_session", max_retries=2, ignore_result=True
)
def summarize_session(self, session_id: str, user_id: str) -> dict:
    """
    1. Load full session messages from PostgreSQL
    2. Run memory_consolidation (memory_service)
    3. Upsert summary into user_memory Qdrant collection
    4. Update sessions.summary in PostgreSQL
    """

    async def _run():
        from sqlalchemy import select

        from app.agents.state import EduMentorState
        from app.core.database import AsyncSessionLocal
        from app.models.session import Session
        from app.rag.collections import get_qdrant_client
        from app.rag.retriever import upsert_user_memory
        from app.services.memory_service import memory_consolidation
        from app.services.session_service import (
            get_session_history_from_db,
            get_user_mastery,
            get_user_preferences,
        )

        # ── Idempotency guard ────────────────────────────────────────────────
        # summarize_session has no job_id — use the session record's summary
        # field as the idempotency marker.  If summary is already populated a
        # previous run completed; skip the expensive LLM + Qdrant upsert.
        async with AsyncSessionLocal() as _db:
            _sr = await _db.execute(select(Session).where(Session.id == uuid.UUID(session_id)))
            _sess = _sr.scalar_one_or_none()
            if _sess and _sess.summary:
                logger.info("session_already_summarized_skipping", session_id=session_id)
                return {"status": "done", "summary_length": len(_sess.summary)}
        # ── End idempotency guard ────────────────────────────────────────────

        async with AsyncSessionLocal() as db:
            history = await get_session_history_from_db(db, uuid.UUID(session_id), limit=50)
            prefs = await get_user_preferences(db, uuid.UUID(user_id))
            mastery = await get_user_mastery(db, uuid.UUID(user_id))

        # Resolve topic name for this session so memory can be filtered by topic
        topic_name = ""
        async with AsyncSessionLocal() as _tdb:
            from app.models.topic import Topic

            _sr2 = await _tdb.execute(select(Session).where(Session.id == uuid.UUID(session_id)))
            _sess2 = _sr2.scalar_one_or_none()
            if _sess2 and _sess2.topic_id:
                _tr = await _tdb.execute(select(Topic).where(Topic.id == _sess2.topic_id))
                _topic = _tr.scalar_one_or_none()
                topic_name = _topic.name if _topic else ""

        state = EduMentorState(
            session_id=session_id,
            user_id=user_id,
            user_query="",
            topic_name=topic_name,
            theta=mastery["theta"],
            student_level=mastery["level"],
            history=history,
            **prefs,
        )
        result = await memory_consolidation(state)
        summary = result.agent_response or ""  # guard against None if LLM call fails

        if summary:
            client = await get_qdrant_client()
            await upsert_user_memory(
                client,
                user_id=user_id,
                doc_id=f"session_summary_{session_id}",
                memory_type="session_summary",
                content=summary,
                topic=topic_name,  # ← now stored, enables topic filtering
                session_id=session_id,
            )
            async with AsyncSessionLocal() as db:
                res = await db.execute(select(Session).where(Session.id == uuid.UUID(session_id)))
                session = res.scalar_one_or_none()
                if session:
                    session.summary = summary
                await db.commit()
            logger.info("session_summarized", session_id=session_id, summary_len=len(summary))
        return {"status": "done", "summary_length": len(summary)}

    _reset_singletons()
    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error("summarize_failed", session_id=session_id, error=str(exc))
        _reset_singletons()
        self.retry(exc=exc, countdown=10)
        return {"status": "failed", "error": str(exc)}
