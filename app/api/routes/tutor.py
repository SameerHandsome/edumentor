"""
Tutor routes — start/end session, voice (202+job), text, SSE stream, feedback, history.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import AsyncIterator
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as _BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import get_graph
from app.agents.state import EduMentorState
from app.api.routes.deps import get_current_user_id
from app.core.config import settings
from app.core.database import get_db
from app.core.multi_layer_cache import (
    ml_get_job,
    ml_get_mastery,
    ml_get_text,
    ml_set_mastery,
    ml_set_text,
)
from app.core.redis_client import (
    FEEDBACK_BAD_SESSIONS_SET,
    FEEDBACK_BAD_SESSIONS_THRESHOLD,
    FEEDBACK_SESSION_THUMBSDOWN,
    FEEDBACK_THUMBSDOWN_PER_SESSION,
    RATE_LIMIT_KEY,
    RATE_LIMIT_KEY_TEXT,
    RATE_LIMIT_KEY_UPLOAD,
    acquire_inflight_lock,
    check_rate_limit,
    get_history,
    push_message,
    release_inflight_lock,
)
from app.models.feedback import Feedback
from app.models.job import Job
from app.models.session import Message, Session
from app.models.topic import Topic
from app.models.user_document import UserDocument
from app.rag.user_docs_ingestion import delete_user_document
from app.schemas.tutor import (
    FeedbackRequest,
    JobStatusResponse,
    MessageResponse,
    SessionResponse,
    StartSessionRequest,
    StartSessionResponse,
    TextRequest,
    TextResponse,
    VoiceResponse,
)
from app.services.session_service import (
    create_session,
    end_session,
    get_session_history_from_db,
    get_user_mastery,
    get_user_preferences,
    save_message,
)
from app.tasks.voice_tasks import process_voice

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/tutor", tags=["tutor"])


@router.post("/start", response_model=StartSessionResponse, status_code=201)
async def start_session(
    body: StartSessionRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    session = await create_session(db, UUID(user_id), body.topic_id)
    if body.session_goal:
        from app.models.user import UserPreference

        pref_res = await db.execute(
            select(UserPreference).where(UserPreference.user_id == UUID(user_id))
        )
        pref = pref_res.scalar_one_or_none()
        if pref:
            pref.session_goal = body.session_goal
    await db.commit()
    logger.info("session_started", session_id=str(session.id), user_id=user_id)
    return StartSessionResponse(session_id=session.id)


@router.post("/end", status_code=202)
async def end_session_route(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == UUID(user_id))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await end_session(db, session_id)
    await db.commit()
    from app.tasks.session_tasks import summarize_session

    summarize_session.delay(str(session_id), user_id)
    return {"status": "ending", "session_id": str(session_id)}


@router.post("/voice", response_model=VoiceResponse, status_code=202)
async def voice_input(
    session_id: UUID = Form(...),
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    allowed, retry_after = await check_rate_limit(
        key=RATE_LIMIT_KEY.format(user_id=user_id),
        limit=settings.VOICE_RATE_LIMIT,
        window=settings.RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429, detail={"error": "rate_limit_exceeded", "retry_after": retry_after}
        )

    audio_bytes = await audio.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()

    job_id = uuid.uuid4()
    job = Job(id=job_id, user_id=UUID(user_id), job_type="voice_process", status="pending")
    db.add(job)
    await db.commit()

    process_voice.delay(str(job_id), user_id, str(session_id), audio_b64, language)
    logger.info("voice_job_queued", job_id=str(job_id))
    return VoiceResponse(job_id=job_id)


@router.post("/text", response_model=TextResponse)
async def text_input(
    body: TextRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    import hashlib

    allowed, retry_after = await check_rate_limit(
        key=RATE_LIMIT_KEY_TEXT.format(user_id=user_id),
        limit=settings.TEXT_RATE_LIMIT,
        window=settings.RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429, detail={"error": "rate_limit_exceeded", "retry_after": retry_after}
        )

    prefs = await get_user_preferences(db, UUID(user_id))

    mastery = await ml_get_mastery(user_id)
    if not mastery:
        mastery = await get_user_mastery(db, UUID(user_id))
        await ml_set_mastery(user_id, mastery)

    history = await get_history(str(body.session_id))
    if not history:
        history = await get_session_history_from_db(db, body.session_id)

    sess_res = await db.execute(
        select(Session).where(Session.id == body.session_id, Session.user_id == UUID(user_id))
    )
    session_row = sess_res.scalar_one_or_none()
    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found")

    # Resolve topic name for memory scoping
    topic_name = ""
    topic_id_str = ""
    if session_row and session_row.topic_id:
        from app.models.topic import Topic as _Topic

        topic_res = await db.execute(select(_Topic).where(_Topic.id == session_row.topic_id))
        topic_obj = topic_res.scalar_one_or_none()
        topic_name = topic_obj.name if topic_obj else ""
        topic_id_str = str(session_row.topic_id)

    # Check if user has any uploaded docs (cheap DB count, avoids Qdrant round-trip)
    from app.models.user_document import UserDocument as _UserDoc

    doc_count_res = await db.execute(
        select(_UserDoc)
        .where(
            _UserDoc.user_id == UUID(user_id),
            _UserDoc.session_id
            == body.session_id,  # current session only — prevents cross-session doc bleed
        )
        .limit(1)
    )
    has_user_docs = doc_count_res.scalar_one_or_none() is not None

    query_hash = hashlib.md5(body.message.encode()).hexdigest()
    cached_response = await ml_get_text(user_id, str(body.session_id), query_hash)
    if cached_response:
        return TextResponse(
            session_id=body.session_id,
            reply=cached_response["reply"],
            agent_type=cached_response["agent_type"],
            trace_id=cached_response.get("trace_id"),
        )

    # ── In-flight deduplication lock ─────────────────────────────────────────
    # Prevents duplicate message inserts when the user re-submits while the
    # LLM is still processing (slow local Ollama, network lag, page refresh).
    # SET NX EX is atomic — only one request wins the lock; the rest get 409.
    lock_acquired = await acquire_inflight_lock(user_id, str(body.session_id), query_hash)
    if not lock_acquired:
        raise HTTPException(
            status_code=409,
            detail="A response to this message is already being generated. Please wait.",
        )

    state = EduMentorState(
        session_id=str(body.session_id),
        user_id=user_id,
        user_query=body.message,
        topic_id=topic_id_str,
        topic_name=topic_name,
        has_user_docs=has_user_docs,
        theta=mastery["theta"],
        student_level=mastery["level"],
        history=history,
        **prefs,
    )
    graph = get_graph()
    try:
        raw_result = await graph.ainvoke(state)
    except Exception:
        await release_inflight_lock(user_id, str(body.session_id), query_hash)
        raise

    result_state: EduMentorState = (
        EduMentorState(**raw_result) if isinstance(raw_result, dict) else raw_result
    )

    if result_state.intent == "quiz":
        await save_message(db, body.session_id, "user", body.message)
        await db.commit()
        await release_inflight_lock(user_id, str(body.session_id), query_hash)
        await push_message(str(body.session_id), "user", body.message)

        # ── Bug 3a fix: extract the actual topic from the user's raw message ──
        # result_state.topic_id holds the session's topic UUID (set at session
        # start), NOT the topic the user mentioned in this message.
        # e.g. "make a quiz on XGBoost" → we must extract "XGBoost", not return
        # the existing session UUID that maps to a different subject.
        #
        # Extraction strategy (in priority order):
        #   1. Parse common quiz-request patterns from the raw message
        #   2. Fall back to result_state.topic_id if it's a plain text name
        #   3. Fall back to empty string (auto-creates a generic topic row)
        import re as _re

        _quiz_patterns = [
            # "quiz on X", "quiz about X", "test on X", "practice X"
            r"(?:quiz|test|exam|practice|questions?)\s+(?:on|about|for|over|covering|regarding)\s+(.+)",
            # "make me a X quiz", "create a X test"
            r"(?:make|create|give|generate|start)\s+(?:me\s+)?(?:a\s+)?(.+?)\s+(?:quiz|test|exam|questions?)",
            # "I want to be quizzed on X"
            r"(?:quiz|test)\s+me\s+(?:on|about)\s+(.+)",
        ]
        extracted_topic: str = ""
        msg_lower = body.message.lower().strip()
        for pattern in _quiz_patterns:
            m = _re.search(pattern, msg_lower)
            if m:
                extracted_topic = m.group(1).strip().rstrip("?.!")
                # Capitalise first letter for display
                extracted_topic = extracted_topic[:1].upper() + extracted_topic[1:]
                break

        # Decide raw_topic: prefer extracted topic; fall back to state value
        # only when it looks like a plain text name (not a UUID).
        state_topic_val = (result_state.topic_id or "").strip()
        is_uuid = False
        try:
            UUID(state_topic_val)
            is_uuid = True
        except (ValueError, AttributeError):
            pass

        raw_topic = extracted_topic or (state_topic_val if not is_uuid else "")
        resolved_topic_id: str | None = None
        topic_display_name: str = raw_topic

        # If state gave a UUID and we have no better text name, keep the UUID
        if not raw_topic and is_uuid:
            resolved_topic_id = state_topic_val

        # Case 1: already resolved to a UUID above
        if not resolved_topic_id and is_uuid and not extracted_topic:
            resolved_topic_id = state_topic_val

        # Case 2: text name — fuzzy match existing topics
        if not resolved_topic_id and raw_topic:
            from sqlalchemy import func as sqlfunc

            topic_res = await db.execute(
                select(Topic).where(sqlfunc.lower(Topic.name).contains(raw_topic.lower())).limit(1)
            )
            matched = topic_res.scalar_one_or_none()
            if matched:
                resolved_topic_id = str(matched.id)
                topic_display_name = matched.name

        # Case 3: no match — auto-create a new topic row for this arbitrary topic
        if not resolved_topic_id:
            new_topic_id = uuid.uuid4()
            slug_base = (
                _re.sub(r"[^a-z0-9]+", "-", (raw_topic or "custom").lower()).strip("-") or "custom"
            )
            slug = f"{slug_base}-{str(new_topic_id)[:8]}"
            new_topic = Topic(
                id=new_topic_id,
                name=raw_topic or "Custom Topic",
                slug=slug,
                description=f"Auto-created from chat quiz intent: {raw_topic}",
                grade_level=10,
                order_index=9999,
            )
            db.add(new_topic)
            await db.flush()
            resolved_topic_id = str(new_topic_id)
            topic_display_name = raw_topic or "Custom Topic"
            logger.info(
                "topic_auto_created_from_chat", topic_id=resolved_topic_id, name=topic_display_name
            )

        logger.info(
            "quiz_redirect",
            user_id=user_id,
            session_id=str(body.session_id),
            topic_id=resolved_topic_id,
            topic_name=topic_display_name,
            extracted_from_message=bool(extracted_topic),
        )

        return TextResponse(
            session_id=body.session_id,
            reply=f"Sure! Let me set up a quiz on **{topic_display_name}** for you.",
            agent_type="quiz",
            quiz_redirect=True,
            topic_id=resolved_topic_id,
            topic_name=topic_display_name,
        )

    # ── Sanity-check the reply before it reaches the user ────────────────────
    # If the model failed and the state object leaked into agent_response,
    # we catch it here instead of shipping raw JSON to the frontend.
    reply = (result_state.agent_response or "").strip()
    _is_state_leak = reply.startswith("{") and any(
        k in reply for k in ("user_query", "agent_type", "session_id", "rag_chunks")
    )
    _is_empty = len(reply) < 5
    if _is_state_leak or _is_empty:
        logger.error(
            "agent_response_corrupted",
            is_state_leak=_is_state_leak,
            is_empty=_is_empty,
            preview=reply[:120],
        )
        reply = "I had trouble generating a response. Please try again."

    await save_message(db, body.session_id, "user", body.message)
    asst_msg = await save_message(
        db,
        body.session_id,
        "assistant",
        reply,
        agent_type=result_state.agent_type,
        trace_id=result_state.langsmith_trace_id,
    )
    await db.commit()

    # Release the in-flight lock now that messages are committed.
    # Do this BEFORE push_message so the lock is gone even if push fails.
    await release_inflight_lock(user_id, str(body.session_id), query_hash)

    await push_message(str(body.session_id), "user", body.message)
    await push_message(str(body.session_id), "assistant", reply)

    response_payload = {
        "reply": reply,
        "agent_type": result_state.agent_type,
        "trace_id": result_state.langsmith_trace_id,
    }
    await ml_set_text(user_id, str(body.session_id), query_hash, response_payload)

    return TextResponse(
        session_id=body.session_id,
        reply=reply,
        agent_type=result_state.agent_type,
        trace_id=result_state.langsmith_trace_id,
        message_id=asst_msg.id,
    )


@router.post("/feedback", status_code=201)
async def submit_feedback(
    body: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Record a thumbs-up (rating=5) or thumbs-down (rating=1) on an assistant reply.

    Thumbs-down tracking (DPO trigger):
      - Per-session Redis counter tracks how many 👎 this student gave in this session.
      - When a student hits FEEDBACK_THUMBSDOWN_PER_SESSION (3) in one session:
          → their user_id is added to the global FEEDBACK_BAD_SESSIONS_SET.
      - When that set reaches FEEDBACK_BAD_SESSIONS_THRESHOLD (5) unique students:
          → export_dpo_pairs Celery task is triggered immediately (instead of
            waiting for the weekly cron), writing a JSONL of prompt+rejected pairs.
          → the set is cleared so the counter resets for the next training round.
    """
    if body.rating not in (1, 5):
        raise HTTPException(
            status_code=422, detail="Rating must be 1 (thumbs-down) or 5 (thumbs-up)"
        )

    # ── Save feedback row ────────────────────────────────────────────────────
    fb = Feedback(
        user_id=UUID(user_id),
        session_id=body.session_id,
        message_id=body.message_id,
        langsmith_trace_id=body.langsmith_trace_id,
        rating=body.rating,
        comment=body.comment,
        is_dpo_candidate=body.rating == 1,
    )
    db.add(fb)
    await db.commit()

    triggered = False

    # ── Thumbs-down path ─────────────────────────────────────────────────────
    if body.rating == 1:
        from app.core.redis_client import redis_client

        async with redis_client() as r:
            session_key = FEEDBACK_SESSION_THUMBSDOWN.format(session_id=str(body.session_id))

            # Increment counter for this session; set 24h TTL on first write
            count = await r.incr(session_key)
            if count == 1:
                await r.expire(session_key, 86400)

            logger.info(
                "thumbsdown_recorded",
                user_id=user_id,
                session_id=str(body.session_id),
                count=count,
                threshold=FEEDBACK_THUMBSDOWN_PER_SESSION,
            )

            # Student has hit 3 👎 in this session — mark them as a "bad session" student
            if count == FEEDBACK_THUMBSDOWN_PER_SESSION:
                await r.sadd(FEEDBACK_BAD_SESSIONS_SET, user_id)
                bad_count = await r.scard(FEEDBACK_BAD_SESSIONS_SET)

                logger.info(
                    "bad_session_student_added",
                    user_id=user_id,
                    bad_student_count=bad_count,
                    trigger_threshold=FEEDBACK_BAD_SESSIONS_THRESHOLD,
                )

                # 5 unique students have each had 3 bad replies — trigger export
                if bad_count >= FEEDBACK_BAD_SESSIONS_THRESHOLD:
                    await r.delete(FEEDBACK_BAD_SESSIONS_SET)  # reset for next round
                    from app.tasks.mlops_tasks import export_dpo_pairs

                    export_dpo_pairs.delay()
                    triggered = True
                    logger.info("dpo_export_triggered", reason="5_students_3_thumbsdown_each")

    return {
        "status": "recorded",
        "triggered_export": triggered,
    }


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Session)
        .where(Session.user_id == UUID(user_id))
        .order_by(Session.started_at.desc())
        .limit(20)
    )
    return [SessionResponse.from_session(s) for s in result.scalars().all()]


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == UUID(user_id))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse.from_session(session)


@router.get("/history", response_model=list[MessageResponse])
async def get_history_route(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    sess_res = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == UUID(user_id))
    )
    session_row = sess_res.scalar_one_or_none()
    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found")
    result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    return [MessageResponse.model_validate(m) for m in result.scalars().all()]


@router.get("/stream")
async def stream_response(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    import json

    from app.services.tts import synthesize_stream

    sess_res = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == UUID(user_id))
    )
    if not sess_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found")

    msg_res = await db.execute(
        select(Message)
        .where(Message.session_id == session_id, Message.role == "assistant")
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    msg = msg_res.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="No assistant message found")

    async def event_generator() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'text', 'content': msg.content})}\n\n"
        async for audio_chunk in synthesize_stream(msg.content):
            chunk_b64 = base64.b64encode(audio_chunk).decode()
            yield f"data: {json.dumps({'type': 'audio', 'content': chunk_b64})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/upload-doc", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    session_id: UUID = Form(...),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    allowed, retry_after = await check_rate_limit(
        key=RATE_LIMIT_KEY_UPLOAD.format(user_id=user_id),
        limit=settings.UPLOAD_RATE_LIMIT,
        window=settings.RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429, detail={"error": "rate_limit_exceeded", "retry_after": retry_after}
        )

    import json as _json
    import uuid as _uuid

    allowed_types = {"text/plain", "application/pdf", "text/markdown"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    raw_bytes = await file.read()

    if file.content_type == "application/pdf":
        try:
            import io

            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF extraction failed: {exc}")
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Document appears to be empty or unreadable.")

    doc_id = str(_uuid.uuid4())
    filename = file.filename or "upload"

    # Mark as processing in Redis immediately so the status endpoint can respond
    from app.core.redis_client import redis_client
    from app.tasks.doc_tasks import DOC_STATUS_KEY, DOC_STATUS_TTL, ingest_document

    async with redis_client() as r:
        await r.setex(
            DOC_STATUS_KEY.format(doc_id=doc_id),
            DOC_STATUS_TTL,
            _json.dumps({"status": "processing"}),
        )

    # Save DB record immediately (chunk_count=0 until task finishes)
    doc_record = UserDocument(
        user_id=UUID(user_id),
        session_id=session_id,
        doc_id=doc_id,
        filename=filename,
        chunk_count=0,
    )
    db.add(doc_record)
    await db.commit()

    # Dispatch to Celery — returns immediately, embedding happens in background
    ingest_document.delay(doc_id, user_id, filename, text, str(session_id))

    logger.info("user_doc_queued", user_id=user_id, doc_id=doc_id, filename=filename)
    return {"doc_id": doc_id, "filename": filename, "status": "processing"}


@router.get("/doc-status/{doc_id}")
async def doc_status(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Poll this endpoint to check if a document has finished indexing."""
    import json as _json

    result = await db.execute(
        select(UserDocument).where(
            UserDocument.doc_id == doc_id, UserDocument.user_id == UUID(user_id)
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found.")

    from app.core.redis_client import redis_client
    from app.tasks.doc_tasks import DOC_STATUS_KEY

    async with redis_client() as r:
        raw = await r.get(DOC_STATUS_KEY.format(doc_id=doc_id))
    if not raw:
        # Key expired or never set — treat as ready
        return {"doc_id": doc_id, "status": "ready"}
    data = _json.loads(raw)
    return {"doc_id": doc_id, **data}


@router.get("/my-docs")
async def list_my_documents(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserDocument)
        .where(UserDocument.user_id == UUID(user_id))
        .order_by(UserDocument.created_at.desc())
    )
    docs = result.scalars().all()
    return [
        {
            "doc_id": d.doc_id,
            "filename": d.filename,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/upload-doc/{doc_id}", status_code=200)
async def delete_document_route(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserDocument).where(
            UserDocument.doc_id == doc_id, UserDocument.user_id == UUID(user_id)
        )
    )
    doc_record = result.scalar_one_or_none()
    if not doc_record:
        raise HTTPException(status_code=404, detail="Document not found.")

    from app.rag.collections import get_qdrant_client

    client = await get_qdrant_client()
    await delete_user_document(client, user_id=user_id, doc_id=doc_id)
    await db.delete(doc_record)
    await db.commit()
    return {"status": "deleted", "doc_id": doc_id}


@router.get("/job/{job_id}/status", response_model=JobStatusResponse)
async def job_status(
    job_id: UUID, user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
):
    cached = await ml_get_job(str(job_id))
    if cached:
        return JobStatusResponse(
            job_id=job_id, status=cached["status"], result=cached.get("result")
        )

    result = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == UUID(user_id)))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job.id, status=job.status, result=job.result, error=job.error)


# ── Session management: rename + delete ───────────────────────────────────────


class _RenameBody(_BaseModel):
    name: str


@router.patch("/sessions/{session_id}/rename", status_code=200)
async def rename_session(
    session_id: UUID,
    body: _RenameBody,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == UUID(user_id))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    name = body.name.strip()[:120]
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    meta = dict(session.metadata_ or {})
    meta["display_name"] = name
    session.metadata_ = meta
    await db.commit()
    return {"status": "renamed", "session_id": str(session_id), "name": name}


@router.delete("/sessions/{session_id}", status_code=200)
async def delete_session(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == UUID(user_id))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()
    return {"status": "deleted", "session_id": str(session_id)}
