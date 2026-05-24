"""Celery voice processing task — STT → LangGraph → TTS → store."""

from __future__ import annotations

import asyncio
import base64
import time
import uuid

import structlog
from prometheus_client import Histogram

from app.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

try:
    VOICE_RTT = Histogram(
        "voice_round_trip_seconds",
        "Full voice round-trip latency",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )
except ValueError:
    from prometheus_client import REGISTRY

    VOICE_RTT = REGISTRY._names_to_collectors["voice_round_trip_seconds"]


def _reset_singletons() -> None:
    """
    Reset ALL module-level async singletons before creating a new event loop.

    Root cause: asyncpg connections (via SQLAlchemy) and redis-py async
    connections are both bound to the event loop that was active when they
    were first created.  When a Celery --pool=solo worker runs task N,
    closes its loop, then runs task N+1 with a new loop, every awaitable
    that touches an old connection raises:
        RuntimeError: Future attached to a different loop

    Fix: dispose/null-out every singleton here so they are recreated fresh
    inside the new loop.  Called before the loop is created AND after it is
    closed (in the finally block) so the NEXT task also starts clean.

    NOTE: redis-py's ConnectionPool.disconnect() is a coroutine — it must
    be called inside a running event loop.  We therefore null-out _pool
    here (synchronously) so get_redis_pool() will build a fresh one, and
    skip the await entirely.  The old pool's connections will be cleaned up
    by the GC / OS once the old loop is closed.
    """
    # 1. SQLAlchemy async engine
    try:
        from app.core.database import reset_for_celery_task

        reset_for_celery_task()
    except Exception:
        pass

    # 2. Redis async connection pool — just null it out; DO NOT call
    #    disconnect() here because it is a coroutine and there is no running
    #    loop at this point.  get_redis_pool() will lazily create a new pool
    #    bound to the fresh loop on first use inside _run().
    try:
        import app.core.redis_client as _rc

        _rc._pool = None
    except Exception:
        pass

    # 3. LangGraph RAG dependencies — qdrant_client and bm25_encoder are
    #    initialised in the FastAPI lifespan and never set in the Celery
    #    worker process.  Reset the graph singleton so get_graph() recompiles,
    #    and flag that RAG deps need re-injection before the next _run().
    try:
        import app.agents.graph as _graph_mod

        _graph_mod._graph = None  # force recompile on next get_graph()
        _graph_mod._qdrant_client = None  # will be re-injected inside _run()
        _graph_mod._bm25_encoder = None
    except Exception:
        pass


@celery_app.task(
    bind=True, name="app.tasks.voice_tasks.process_voice", max_retries=3, ignore_result=True
)
def process_voice(
    self, job_id: str, user_id: str, session_id: str, audio_b64: str, language: str = "en"
) -> dict:
    """
    1. Base64-decode audio bytes
    2. Transcribe via faster-whisper (STT)
    3. Run LangGraph with transcribed text
    4. Synthesize reply via Coqui TTS
    5. Store message in DB + Redis
    6. Update job status
    """
    start = time.perf_counter()

    # Reset BEFORE creating the loop so fresh connections bind to the new loop.
    _reset_singletons()

    async def _run():
        import base64 as b64

        from sqlalchemy import select

        import app.agents.graph as _graph_mod
        from app.agents.graph import get_graph, set_rag_dependencies
        from app.agents.state import EduMentorState
        from app.core.config import settings
        from app.core.database import AsyncSessionLocal
        from app.core.redis_client import CACHE_KEY_JOB, cache_set, get_history, push_message
        from app.models.job import Job
        from app.services.session_service import (
            get_user_mastery,
            get_user_preferences,
            save_message,
        )
        from app.services.stt import transcribe
        from app.services.tts import synthesize

        # ── Inject RAG deps if missing (Celery worker has no FastAPI lifespan) ──
        if _graph_mod._qdrant_client is None or _graph_mod._bm25_encoder is None:
            from app.rag.bm25 import BM25Encoder
            from app.rag.collections import COLLECTION_CURRICULUM, get_qdrant_client

            _qdrant = await get_qdrant_client()

            # Fit BM25 on the actual ingested curriculum_docs — same as FastAPI
            # lifespan does in main.py:_build_bm25_corpus_from_qdrant().
            # A hardcoded corpus leaves most domain terms OOV (out-of-vocabulary),
            # so encode_query() returns empty indices → sparse search returns
            # nothing → only dense retrieval runs → rag_chunks often empty.
            corpus: list[str] = []
            try:
                offset = None
                while True:
                    points, next_offset = await _qdrant.scroll(
                        collection_name=COLLECTION_CURRICULUM,
                        limit=250,
                        offset=offset,
                        with_payload=["content"],
                        with_vectors=False,
                    )
                    for p in points:
                        content = (p.payload or {}).get("content", "")
                        if content.strip():
                            corpus.append(content[:500])
                    if next_offset is None:
                        break
                    offset = next_offset
                logger.info("voice_task_bm25_corpus_loaded", docs=len(corpus))
            except Exception as _bm25_exc:
                logger.warning("voice_task_bm25_scroll_failed", error=str(_bm25_exc))

            if not corpus:
                # Qdrant not yet ingested or unreachable — use a richer fallback
                # corpus that covers typical student questions so BM25 is not
                # completely blind.  Dense retrieval still runs in parallel so
                # this only affects the sparse half of the hybrid search.
                corpus = [
                    # Mathematics
                    "mathematics algebra geometry calculus equations variables functions derivatives integrals limits",
                    "trigonometry vectors matrices linear algebra probability statistics",
                    # Biology
                    "biology cell nucleus photosynthesis mitosis meiosis genetics dna rna protein synthesis",
                    "evolution ecology ecosystem anatomy physiology respiration digestion",
                    # Physics
                    "physics motion force energy momentum velocity acceleration Newton laws thermodynamics",
                    "electricity magnetism circuits waves optics quantum mechanics",
                    # Chemistry
                    "chemistry atoms molecules bonds reactions periodic table elements compounds",
                    "organic chemistry acids bases redox reactions thermochemistry electrochemistry",
                    # Coding / CS
                    "coding programming python algorithms data structures variables loops functions",
                    "neural networks machine learning deep learning backpropagation gradient descent",
                    "tokenization NLP natural language processing BERT GPT transformers attention",
                    "forward propagation activation functions weights biases layers training inference",
                    # General study
                    "student learning knowledge understanding concepts practice problem solving",
                ]
                logger.info("voice_task_bm25_fallback_corpus_used", docs=len(corpus))

            _bm25 = BM25Encoder()
            _bm25.fit(corpus)
            set_rag_dependencies(_qdrant, _bm25)
            logger.info(
                "voice_task_rag_deps_initialised",
                vocab_size=_bm25.vocab_size(),
                corpus_docs=len(corpus),
            )

        # ── Idempotency guard ────────────────────────────────────────────────
        from app.core.multi_layer_cache import ml_get_job as _ml_get_job

        cached = await _ml_get_job(job_id)
        if cached and cached.get("status") == "done":
            logger.info("voice_task_already_done_skipping", job_id=job_id)
            return cached["result"]
        async with AsyncSessionLocal() as _db:
            _jr = await _db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
            _job = _jr.scalar_one_or_none()
            if _job and _job.status == "done":
                logger.info("voice_task_already_done_db_skipping", job_id=job_id)
                return _job.result

        # ── Mark job processing + store celery_task_id ───────────────────────
        async with AsyncSessionLocal() as _db:
            _jr2 = await _db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
            _job2 = _jr2.scalar_one_or_none()
            if _job2:
                _job2.status = "processing"
                _job2.celery_task_id = self.request.id or ""
            await _db.commit()

        audio_bytes = b64.b64decode(audio_b64)
        text = await transcribe(audio_bytes, language=language)
        if not text:
            return {"status": "failed", "error": "transcription_empty"}

        async with AsyncSessionLocal() as db:
            prefs = await get_user_preferences(db, uuid.UUID(user_id))
            mastery = await get_user_mastery(db, uuid.UUID(user_id))

        history = await get_history(session_id)
        if not history:
            async with AsyncSessionLocal() as db:
                from app.services.session_service import get_session_history_from_db

                history = await get_session_history_from_db(db, uuid.UUID(session_id))

        # ── Fetch session context (topic + user_docs flag) ────────────────────
        # The text route does this inline; the voice task must replicate it so
        # retrieval_node gets: topic_id (CRAG filter), topic_name (memory scope),
        # and has_user_docs (gates user_docs retrieval). Without these, all three
        # retrieval paths are degraded or fully skipped.
        topic_id_str = ""
        topic_name = ""
        has_user_docs = False
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select as _select

            from app.models.session import Session as _Session
            from app.models.topic import Topic as _Topic
            from app.models.user_document import UserDocument as _UserDoc

            sess_res = await db.execute(
                _select(_Session).where(_Session.id == uuid.UUID(session_id))
            )
            sess_row = sess_res.scalar_one_or_none()
            if sess_row and sess_row.topic_id:
                topic_id_str = str(sess_row.topic_id)
                topic_res = await db.execute(_select(_Topic).where(_Topic.id == sess_row.topic_id))
                topic_obj = topic_res.scalar_one_or_none()
                topic_name = topic_obj.name if topic_obj else ""

            doc_res = await db.execute(
                _select(_UserDoc).where(_UserDoc.user_id == uuid.UUID(user_id)).limit(1)
            )
            has_user_docs = doc_res.scalar_one_or_none() is not None

        state = EduMentorState(
            session_id=session_id,
            user_id=user_id,
            user_query=text,
            topic_id=topic_id_str,
            topic_name=topic_name,
            has_user_docs=has_user_docs,
            theta=mastery["theta"],
            student_level=mastery["level"],
            history=history,
            **prefs,
        )

        graph = get_graph()
        raw_result = await graph.ainvoke(state)

        # graph.ainvoke() returns a raw dict — cast to EduMentorState.
        # Without this, raw_result.agent_response raises AttributeError.
        result_state: EduMentorState = (
            EduMentorState(**raw_result) if isinstance(raw_result, dict) else raw_result
        )

        reply_text = result_state.agent_response
        audio_bytes_out = await synthesize(reply_text, language=language)
        audio_out_b64 = base64.b64encode(audio_bytes_out).decode() if audio_bytes_out else ""

        async with AsyncSessionLocal() as db:
            await save_message(db, uuid.UUID(session_id), "user", text, trace_id="")
            msg = await save_message(
                db,
                uuid.UUID(session_id),
                "assistant",
                reply_text,
                agent_type=result_state.agent_type,
                trace_id=result_state.langsmith_trace_id,
            )
            job_result_data = {
                "text": reply_text,
                "audio_b64": audio_out_b64,
                "message_id": str(msg.id),
                "agent_type": result_state.agent_type,
            }
            job_result = await db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
            job = job_result.scalar_one_or_none()
            if job:
                job.status = "done"
                job.result = job_result_data
            await db.commit()

        await cache_set(
            CACHE_KEY_JOB.format(job_id=job_id),
            {"status": "done", "result": job_result_data},
            settings.CACHE_JOB_TTL,
        )

        await push_message(session_id, "user", text)
        await push_message(session_id, "assistant", reply_text)

        elapsed = time.perf_counter() - start
        VOICE_RTT.observe(elapsed)
        logger.info("voice_task_done", job_id=job_id, duration_ms=round(elapsed * 1000))
        return {"status": "done", "text": reply_text}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error("voice_task_failed", job_id=job_id, error=str(exc))
        _exc = exc  # capture before Python clears the name after the except block

        async def _mark_failed():
            _reset_singletons()
            from sqlalchemy import select

            from app.core.config import settings as _s
            from app.core.database import AsyncSessionLocal
            from app.core.redis_client import CACHE_KEY_JOB, cache_set
            from app.models.job import Job

            err_str = str(_exc)[:500]
            try:
                async with AsyncSessionLocal() as _db:
                    _jr = await _db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
                    _job = _jr.scalar_one_or_none()
                    if _job and _job.status != "done":
                        _job.status = "failed"
                        _job.error = err_str
                    await _db.commit()
                await cache_set(
                    CACHE_KEY_JOB.format(job_id=job_id),
                    {"status": "failed", "error": err_str},
                    _s.CACHE_JOB_TTL,
                )
            except Exception as db_exc:
                logger.error("voice_task_failed_db_update_error", job_id=job_id, error=str(db_exc))

        asyncio.run(_mark_failed())
        _reset_singletons()

        if self.request.retries < self.max_retries:
            self.retry(exc=_exc, countdown=5)
        return {"status": "failed", "error": str(_exc)}
