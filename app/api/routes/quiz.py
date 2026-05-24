"""
Quiz HTTP routes — generate questions and submit answers.

Endpoints
---------
GET  /quiz/generate          — fetch IRT-matched questions from DB (cached)
POST /quiz/submit            — score answer inline with await (rate-limited)
GET  /quiz/result/{job_id}   — poll scoring result (multi-layer cache)
GET  /quiz/history           — past attempts for current user

Design notes
------------
* Rate limiting  : QUIZ_RATE_LIMIT / RATE_LIMIT_WINDOW_SECONDS applied to
                   both /generate and /submit via the _quiz_rate_limit dep.
* Scoring        : /submit scores the answer directly with `await score_attempt()`
                   — no Celery.  This eliminates the event-loop bug (asyncio.run()
                   inside a process that already owns a loop).  Returns the full
                   result immediately (200 OK) instead of a 202 + job_id poll.
                   A Job row is still written (status="done") so /result/{job_id}
                   keeps working for any client that still calls it.
* Caching        : /generate uses ml_get_quiz / ml_set_quiz (L1→L2→DB).
                   /result   uses ml_get_job              (L1→L2→DB).
* Quiz generation: The quiz_agent (DSPy + Groq primary, direct Groq httpx
                   fallback) generates questions and returns them as JSON in
                   state.agent_response.  The route parses that JSON and
                   persists the question to quiz_questions so it can be scored
                   later via score_attempt.  If the DB has fewer questions than
                   num_questions, the LLM is called on-the-fly to fill the gap.
"""

from __future__ import annotations

import json
import uuid
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

# Move agent imports to module level — no reason to defer them inside the
# route function, and top-level imports make missing-module errors surface
# at startup rather than on the first request.
from app.agents.quiz_agent import quiz_agent
from app.agents.state import EduMentorState
from app.api.routes.deps import get_current_user_id
from app.core.config import settings
from app.core.database import get_db
from app.core.multi_layer_cache import (
    ml_get_job,
    ml_get_mastery,
    ml_get_quiz,
    ml_set_mastery,
    ml_set_quiz,
)
from app.core.redis_client import RATE_LIMIT_KEY_QUIZ, check_rate_limit
from app.models.job import Job
from app.models.quiz import QuizAttempt, QuizQuestion
from app.models.topic import Topic
from app.schemas.quiz import QuizHistoryItem, QuizQuestionOut, QuizSubmitSchema
from app.services.session_service import get_user_mastery, get_user_preferences

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/quiz", tags=["quiz"])


# ── Topic auto-creation helper ───────────────────────────────────────────────


async def get_or_create_topic(db: AsyncSession, topic_id: UUID, topic_name: str = "") -> UUID:
    """
    Ensure a topic row exists for the given UUID.
    If not found (arbitrary/custom topic), insert a minimal placeholder row
    so quiz_questions FK constraint is satisfied.
    Returns the same topic_id.
    """
    result = await db.execute(select(Topic).where(Topic.id == topic_id))
    if result.scalar_one_or_none():
        return topic_id

    # Auto-create a placeholder topic
    display_name = topic_name or f"Custom Topic ({str(topic_id)[:8]})"
    slug = str(topic_id)  # UUID as slug — guaranteed unique
    new_topic = Topic(
        id=topic_id,
        name=display_name,
        slug=slug,
        description="Auto-created for custom quiz topic",
        grade_level=10,
        order_index=9999,
    )
    db.add(new_topic)
    await db.flush()
    logger.info("topic_auto_created", topic_id=str(topic_id), name=display_name)
    return topic_id


# ── Rate-limit dependency ────────────────────────────────────────────────────


async def _quiz_rate_limit(user_id: str = Depends(get_current_user_id)) -> str:
    """Raise 429 when the user exceeds QUIZ_RATE_LIMIT per window."""
    allowed, retry_after = await check_rate_limit(
        RATE_LIMIT_KEY_QUIZ.format(user_id=user_id),
        settings.QUIZ_RATE_LIMIT,
        settings.RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "retry_after": retry_after},
        )
    return user_id


# ── Generate questions ───────────────────────────────────────────────────────


@router.get("/generate", response_model=list[QuizQuestionOut])
async def generate_quiz(
    session_id: UUID = Query(..., description="Active session UUID"),
    num_questions: int = Query(default=10, ge=1, le=20),
    topic_id: UUID | None = Query(
        default=None, description="Topic UUID (omit when using topic_name)"
    ),
    topic_name: str | None = Query(
        default=None, description="Arbitrary topic name — auto-creates topic if needed"
    ),
    user_id: str = Depends(_quiz_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate IRT-difficulty-matched questions for the current student.

    Flow
    ----
    1. Load mastery (L1 → L2 → DB) to get current theta for this topic.
    2. Check the multi-layer quiz cache.  Cache hit → return immediately.
    3. Complete all DB reads (topic upsert + user prefs) then commit — this
       releases the asyncpg connection before the Groq call so Neon never
       sees an idle connection and closes it mid-request.
    4. Cache miss → call quiz_agent (DSPy + Groq, ~2-5 s).
    5. Persist the generated question to quiz_questions (so score_quiz can
       look it up by ID later).
    6. Backfill from DB for any remaining num_questions slots using the IRT
       difficulty window filter.
    7. Write results to both cache layers and return.

    Cache key: (user_id, topic_id, theta_bucket) where theta is rounded to
    1 d.p. so small fluctuations reuse the cache across several questions.
    Cache is invalidated by score_quiz after every answered question.
    """
    # ── 0. Resolve topic — support both UUID and arbitrary name ────────────────
    # If topic_id is provided, use it directly (after ensuring the row exists).
    # If only topic_name is provided, look it up or auto-create a new topic row.
    if topic_id is None and not topic_name:
        raise HTTPException(
            status_code=422,
            detail="Either topic_id (UUID) or topic_name (string) is required.",
        )

    if topic_id is None:
        # topic_name flow — find existing or create new
        import re as _re

        from sqlalchemy import func as sqlfunc

        topic_res = await db.execute(
            select(Topic)
            .where(
                sqlfunc.lower(Topic.name).contains(topic_name.lower())  # type: ignore[union-attr]
            )
            .limit(1)
        )
        matched_topic = topic_res.scalar_one_or_none()
        if matched_topic:
            topic_id = matched_topic.id
        else:
            new_tid = uuid.uuid4()
            slug_base = _re.sub(r"[^a-z0-9]+", "-", topic_name.lower()).strip("-") or "custom"
            slug = f"{slug_base}-{str(new_tid)[:8]}"
            new_topic = Topic(
                id=new_tid,
                name=topic_name,
                slug=slug,
                description=f"Auto-created from quiz request: {topic_name}",
                grade_level=10,
                order_index=9999,
            )
            db.add(new_topic)
            await db.flush()
            topic_id = new_tid
            logger.info("topic_auto_created_in_generate", topic_id=str(topic_id), name=topic_name)

    topic_id_str = str(topic_id)

    # ── 1. Load mastery (L1 → L2 → DB) ─────────────────────────────────────
    mastery = await ml_get_mastery(user_id)
    if not mastery:
        mastery = await get_user_mastery(db, UUID(user_id))
        await ml_set_mastery(user_id, mastery)

    theta: float = mastery["theta"]
    theta_bucket = f"{round(theta, 1):.1f}"

    # ── 2. Cache check ───────────────────────────────────────────────────────
    cached = await ml_get_quiz(user_id, topic_id_str, theta_bucket, num_questions)
    if cached:
        logger.info(
            "quiz_generate_cache_hit",
            user_id=user_id,
            topic_id=topic_id_str,
            theta_bucket=theta_bucket,
        )
        return cached

    # ── 3. Finish all DB reads, then commit to release the connection ────────
    # All reads (topic upsert + user prefs) must complete BEFORE db.commit().
    # commit() releases the asyncpg connection back to the pool. After this
    # point no DB work happens until step 5, so the connection is free while
    # Groq generates the question (2-5 s) — no idle-timeout risk from Neon.
    await get_or_create_topic(db, topic_id, topic_name=topic_name or "")
    prefs = await get_user_preferences(db, UUID(user_id))
    await db.commit()

    # ── 4. Call quiz_agent (Groq — typically 2-5 s) ──────────────────────────
    state = EduMentorState(
        session_id=str(session_id),
        user_id=user_id,
        topic_id=topic_id_str,
        topic_name=topic_name or "",  # human label → quiz_agent passes this to LLM, not the UUID
        theta=theta,
        student_level=mastery["level"],
        **prefs,
    )
    result_state: EduMentorState = await quiz_agent(state)

    # ── 5. Persist the generated question (fresh transaction) ────────────────
    generated_questions: list[dict] = []

    if result_state.agent_response:
        try:
            q_data = json.loads(result_state.agent_response)

            new_q = QuizQuestion(
                id=uuid.uuid4(),
                topic_id=topic_id,
                question_text=q_data.get("question", ""),
                choices=q_data.get("choices", {}),
                correct_answer=q_data.get("correct_answer", "A").upper()[:1],
                explanation=q_data.get("explanation", ""),
                difficulty_b=float(q_data.get("difficulty_b", theta + 0.2)),
                created_by="quiz_agent",
            )
            db.add(new_q)
            await db.commit()
            await db.refresh(new_q)

            generated_questions.append(
                {
                    "id": str(new_q.id),
                    "question_text": new_q.question_text,
                    "choices": new_q.choices,
                    "difficulty_b": new_q.difficulty_b,
                }
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "quiz_agent_parse_failed", error=str(exc), raw=result_state.agent_response[:200]
            )

    # ── 6. Backfill remaining slots from DB using IRT window ─────────────────
    # Bug 3b fix: pass the FULL list of already-generated IDs to NOT IN so
    # DB backfill never returns a duplicate of the LLM-generated question.
    slots_needed = num_questions - len(generated_questions)
    if slots_needed > 0:
        window = settings.IRT_DIFFICULTY_WINDOW
        existing_ids = [UUID(q["id"]) for q in generated_questions]

        db_result = await db.execute(
            select(QuizQuestion)
            .where(
                and_(
                    QuizQuestion.topic_id == topic_id,
                    QuizQuestion.difficulty_b >= theta - window,
                    QuizQuestion.difficulty_b <= theta + window,
                    QuizQuestion.id.notin_(existing_ids) if existing_ids else True,
                )
            )
            .order_by((QuizQuestion.difficulty_b - theta) * (QuizQuestion.difficulty_b - theta))
            .limit(slots_needed)
        )
        db_questions = db_result.scalars().all()
        generated_questions.extend(
            [
                {
                    "id": str(q.id),
                    "question_text": q.question_text,
                    "choices": q.choices,
                    "difficulty_b": q.difficulty_b,
                }
                for q in db_questions
            ]
        )

    # ── 6b. Fill remaining slots by generating additional questions via LLM. ──
    # DSPy is skipped here — it adds latency and offers no benefit for bulk
    # sequential generation.  _generate_via_groq_direct is faster, more
    # reliable, and accepts an exclusion list so duplicates are rare.
    still_needed = num_questions - len(generated_questions)
    if still_needed > 0:
        from app.agents.quiz_agent import _generate_via_groq_direct

        topic_label = topic_name or topic_id_str or "general knowledge"
        logger.info("quiz_llm_topup", topic=topic_label, still_needed=still_needed, theta=theta)

        # Retry loop — duplicates and failures don't count toward the quota.
        # b_target is varied each attempt so the LLM sees a fresh difficulty
        # signal and is nudged toward generating a distinct question.
        filled = 0
        max_attempts = still_needed * 5  # 5 tries per needed question is plenty
        attempts = 0
        while filled < still_needed and attempts < max_attempts:
            attempts += 1
            # Cycle b_offset: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 then repeats
            b_offset = 0.2 + ((attempts - 1) % 6) * 0.1
            b_target = theta + b_offset

            # Always pass the full exclusion list so Groq avoids repeats
            existing_texts_list = [q["question_text"] for q in generated_questions]

            q_data = await _generate_via_groq_direct(
                topic_label,
                theta,
                b_target,
                exclude_questions=existing_texts_list,
            )
            if not q_data:
                logger.warning("quiz_llm_topup_failed", attempt=attempts)
                continue  # one failure doesn't kill the loop — try again

            # Dedup: skip if same question text already in the set
            new_text = q_data.get("question", "").strip().lower()
            existing_texts = {q["question_text"].strip().lower() for q in generated_questions}
            if new_text in existing_texts:
                logger.warning(
                    "quiz_llm_topup_duplicate_skipped", attempt=attempts, topic=topic_label
                )
                continue

            try:
                new_q = QuizQuestion(
                    id=uuid.uuid4(),
                    topic_id=topic_id,
                    question_text=q_data.get("question", ""),
                    choices=q_data.get("choices", {}),
                    correct_answer=q_data.get("correct_answer", "A").upper()[:1],
                    explanation=q_data.get("explanation", ""),
                    difficulty_b=float(q_data.get("difficulty_b", b_target)),
                    created_by="quiz_agent_topup",
                )
                db.add(new_q)
                await db.commit()
                await db.refresh(new_q)
                generated_questions.append(
                    {
                        "id": str(new_q.id),
                        "question_text": new_q.question_text,
                        "choices": new_q.choices,
                        "difficulty_b": new_q.difficulty_b,
                    }
                )
                filled += 1
            except Exception as exc:
                logger.warning("quiz_llm_topup_persist_failed", error=str(exc))
                break

    if not generated_questions:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_questions_available",
                "hint": (
                    f"No questions found for topic {topic_id} near "
                    f"difficulty [{theta - settings.IRT_DIFFICULTY_WINDOW:.2f}, "
                    f"{theta + settings.IRT_DIFFICULTY_WINDOW:.2f}]. "
                    "The quiz agent also failed to generate one. "
                    "Check GROQ_API_KEY is set and the topic_id is valid."
                ),
            },
        )

    # ── 7. Write to both cache layers ────────────────────────────────────────
    await ml_set_quiz(user_id, topic_id_str, theta_bucket, generated_questions, num_questions)
    logger.info(
        "quiz_generated",
        user_id=user_id,
        topic_id=topic_id_str,
        theta=theta,
        count=len(generated_questions),
    )
    return generated_questions


# ── Submit answer ────────────────────────────────────────────────────────────


@router.post("/submit", status_code=200)
async def submit_quiz_answer(
    body: QuizSubmitSchema,
    user_id: str = Depends(_quiz_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a quiz answer and score it directly (no Celery).

    Scores inline with `await` — eliminates the event-loop bug that occurred
    when the Celery worker called asyncio.new_event_loop() inside a process
    that already owns an event loop.  Returns the full result immediately (200).

    Idempotency: job row is written with status="done" + result so the existing
    GET /quiz/result/{job_id} polling endpoint keeps working for any client
    that still calls it.
    """
    from app.core.multi_layer_cache import ml_delete_mastery, ml_set_job
    from app.services.irt import score_attempt

    q_res = await db.execute(select(QuizQuestion).where(QuizQuestion.id == body.question_id))
    if not q_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Question not found")

    job_id = uuid.uuid4()
    job_id_str = str(job_id)

    # Create job record for audit trail / history compatibility
    job = Job(
        id=job_id,
        user_id=UUID(user_id),
        job_type="quiz_score",
        status="pending",
    )
    db.add(job)
    await db.commit()

    # Score directly — no Celery, no event-loop conflict
    result = await score_attempt(
        db,
        user_id=user_id,
        question_id=str(body.question_id),
        session_id=str(body.session_id),
        selected_answer=body.selected_answer,
        time_taken_seconds=body.time_taken_seconds,
    )

    # Persist result on the job row
    job_res = await db.execute(select(Job).where(Job.id == job_id))
    job_obj = job_res.scalar_one_or_none()
    if job_obj:
        job_obj.status = "done"
        job_obj.result = result
    await db.commit()

    # Invalidate mastery cache so next /generate uses fresh theta
    await ml_delete_mastery(user_id)
    await ml_set_job(job_id_str, {"status": "done", "result": result})

    logger.info(
        "quiz_scored_inline",
        job_id=job_id_str,
        user_id=user_id,
        question_id=str(body.question_id),
        is_correct=result["is_correct"],
    )
    # Return job_id alongside the result so the frontend poll path works:
    # quiz.js does const { job_id } = await res.json() then polls /quiz/result/{job_id}.
    # The result is already in cache (ml_set_job above) so the poll resolves instantly.
    return {"job_id": job_id_str, "status": "done", "result": result}


# ── Poll result ──────────────────────────────────────────────────────────────


@router.get("/result/{job_id}")
async def quiz_result(
    job_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the scoring result for a submitted answer.

    Lookup order: L1 (in-process TTLCache) → L2 (Redis) → PostgreSQL.
    The score_quiz worker writes to both cache layers on completion so most
    polls are served without hitting PostgreSQL.
    """
    cached = await ml_get_job(str(job_id))
    if cached:
        return cached

    result = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == UUID(user_id)))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job_id": job.id, "status": job.status, "result": job.result}


# ── History ──────────────────────────────────────────────────────────────────


@router.get("/history", response_model=list[QuizHistoryItem])
async def quiz_history(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return past quiz attempts for the current user, newest first."""
    result = await db.execute(
        select(QuizAttempt)
        .where(QuizAttempt.user_id == UUID(user_id))
        .order_by(QuizAttempt.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()
