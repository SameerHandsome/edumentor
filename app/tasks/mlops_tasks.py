"""
MLOps Celery tasks:
- export_dpo_pairs: export low-rated assistant replies as (prompt, rejected) JSONL
                    for DPO fine-tuning on Colab.  Triggered on-demand when 5
                    students each give 3 thumbs-down, OR by the weekly beat schedule.
- run_drift_detection: Evidently AI drift check on student input distribution.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC

import structlog

from app.tasks.celery_app import celery_app
from app.tasks.voice_tasks import _reset_singletons

logger = structlog.get_logger(__name__)

# JSONL is written here.  Mount this path or scp it to your laptop before
# opening the Colab notebook.
DPO_EXPORT_PATH = os.environ.get(
    "DPO_EXPORT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "dpo_pairs.jsonl"),
)


@celery_app.task(name="app.tasks.mlops_tasks.export_dpo_pairs", max_retries=3, ignore_result=True)
def export_dpo_pairs() -> dict:
    """
    Export feedback rows with rating=1 (thumbs-down) as DPO training pairs.

    For every thumbed-down assistant message the task fetches:
      - The assistant message text                    → "rejected"
      - The immediately preceding user message text  → "prompt"
      - Topic name and agent_type                    → metadata

    Writes / appends to DPO_EXPORT_PATH as newline-delimited JSON.
    Marks exported rows as is_dpo_candidate=True so they are never
    exported twice.

    The "chosen" field is left null — the Colab notebook
    (notebooks/dpo_finetune.ipynb) generates it via Groq (free tier)
    before running DPO fine-tuning with Unsloth on Phi-3.5-mini.
    """

    async def _run():
        from sqlalchemy import select, update

        from app.core.database import AsyncSessionLocal
        from app.models.feedback import Feedback
        from app.models.session import Message, Session
        from app.models.topic import Topic

        async with AsyncSessionLocal() as db:
            # Lock un-exported thumbs-down rows atomically
            result = await db.execute(
                select(Feedback)
                .where(
                    Feedback.rating == 1,
                    Feedback.is_dpo_candidate == False,  # noqa: E712
                    Feedback.message_id.isnot(None),  # need message_id to fetch context
                )
                .with_for_update()
            )
            low_rated = result.scalars().all()

            if not low_rated:
                logger.info("dpo_export_skipped", reason="no_new_thumbsdown_rows")
                return {"exported": 0}

            pairs = []
            for fb in low_rated:
                # Fetch the thumbed-down assistant message
                asst_res = await db.execute(select(Message).where(Message.id == fb.message_id))
                asst_msg = asst_res.scalar_one_or_none()
                if not asst_msg:
                    continue

                # Fetch the user message that immediately preceded it
                user_res = await db.execute(
                    select(Message)
                    .where(
                        Message.session_id == asst_msg.session_id,
                        Message.role == "user",
                        Message.created_at < asst_msg.created_at,
                    )
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )
                user_msg = user_res.scalar_one_or_none()
                if not user_msg:
                    continue

                # Fetch topic name for context
                topic_name = ""
                if fb.session_id:
                    sess_res = await db.execute(select(Session).where(Session.id == fb.session_id))
                    sess = sess_res.scalar_one_or_none()
                    if sess and sess.topic_id:
                        topic_res = await db.execute(select(Topic).where(Topic.id == sess.topic_id))
                        topic = topic_res.scalar_one_or_none()
                        if topic:
                            topic_name = topic.name

                pairs.append(
                    {
                        "prompt": user_msg.content,
                        "rejected": asst_msg.content,
                        "chosen": None,  # filled by Colab notebook
                        "topic": topic_name,
                        "agent_type": asst_msg.agent_type or "",
                        "session_id": str(fb.session_id),
                        "message_id": str(fb.message_id),
                        "trace_id": fb.langsmith_trace_id or "",
                    }
                )

            # Mark all as exported (single UPDATE)
            if pairs:
                ids = [fb.id for fb in low_rated if fb.message_id is not None]
                await db.execute(
                    update(Feedback).where(Feedback.id.in_(ids)).values(is_dpo_candidate=True)
                )
            await db.commit()

        # Write JSONL — append so previous exports are not lost
        if pairs:
            os.makedirs(os.path.dirname(os.path.abspath(DPO_EXPORT_PATH)), exist_ok=True)
            with open(DPO_EXPORT_PATH, "a", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info("dpo_pairs_exported", count=len(pairs), path=DPO_EXPORT_PATH)
        return {"exported": len(pairs), "path": DPO_EXPORT_PATH}

    _reset_singletons()
    try:
        return asyncio.run(_run())
    finally:
        _reset_singletons()


@celery_app.task(name="app.tasks.mlops_tasks.run_drift_detection", ignore_result=True)
def run_drift_detection() -> dict:
    """
    Run Evidently AI drift detection on recent student inputs vs reference window.
    Triggers retraining via GitHub Actions webhook if drift exceeds threshold.
    """

    async def _run():
        from datetime import datetime, timedelta

        import httpx
        from sqlalchemy import select

        from app.core.database import AsyncSessionLocal
        from app.models.session import Message

        try:
            import pandas as pd
            from evidently.metric_preset import TextOverviewPreset  # type: ignore
            from evidently.report import Report  # type: ignore

            _evidently_available = True
        except ImportError:
            _evidently_available = False
            logger.warning("evidently_not_installed")

        from app.core.config import settings

        now = datetime.now(UTC)
        ref_start = now - timedelta(days=settings.EVIDENTLY_REFERENCE_WINDOW_DAYS * 2)
        ref_end = now - timedelta(days=settings.EVIDENTLY_REFERENCE_WINDOW_DAYS)
        cur_start = ref_end

        async with AsyncSessionLocal() as db:
            ref_res = await db.execute(
                select(Message.content).where(
                    Message.role == "user",
                    Message.created_at >= ref_start,
                    Message.created_at < ref_end,
                )
            )
            cur_res = await db.execute(
                select(Message.content).where(
                    Message.role == "user",
                    Message.created_at >= cur_start,
                )
            )
            reference_texts = [r[0] for r in ref_res.fetchall()]
            current_texts = [r[0] for r in cur_res.fetchall()]

        if len(reference_texts) < 50 or len(current_texts) < 10:
            return {"status": "skipped", "reason": "insufficient_data"}

        drift_score = 0.0
        if _evidently_available:
            import pandas as pd

            ref_df = pd.DataFrame({"text": reference_texts})
            cur_df = pd.DataFrame({"text": current_texts})
            report = Report(metrics=[TextOverviewPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            result_dict = report.as_dict()
            # Extract drift score from Evidently result
            try:
                drift_score = result_dict["metrics"][0]["result"].get("drift_score", 0.0)
            except (KeyError, IndexError):
                drift_score = 0.0
        else:
            # Simple length-distribution drift proxy
            ref_avg = sum(len(t) for t in reference_texts) / len(reference_texts)
            cur_avg = sum(len(t) for t in current_texts) / len(current_texts)
            drift_score = abs(ref_avg - cur_avg) / (ref_avg + 1)

        drift_detected = drift_score > settings.DRIFT_THRESHOLD
        logger.info(
            "drift_detection",
            drift_score=drift_score,
            threshold=settings.DRIFT_THRESHOLD,
            drift_detected=drift_detected,
        )

        if drift_detected and settings.GITHUB_WEBHOOK_URL:
            async with httpx.AsyncClient() as client:
                await client.post(
                    settings.GITHUB_WEBHOOK_URL,
                    headers={
                        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
                        "Accept": "application/vnd.github+json",
                    },
                    json={
                        "event_type": "drift_detected",
                        "client_payload": {"drift_score": drift_score},
                    },
                )
            logger.info("retraining_webhook_triggered", drift_score=drift_score)

        async with AsyncSessionLocal() as db:
            import uuid as _uuid

            from sqlalchemy import select as _select

            from app.models.job import Job
            from app.models.user import User

            admin_res = await db.execute(_select(User).limit(1))
            admin = admin_res.scalar_one_or_none()
            if admin and drift_detected:
                drift_job = Job(
                    id=_uuid.uuid4(),
                    user_id=admin.id,
                    job_type="drift_retraining",
                    status="triggered",
                    result={"drift_score": drift_score},
                )
                db.add(drift_job)
                await db.commit()

        return {"status": "done", "drift_score": drift_score, "drift_detected": drift_detected}

    _reset_singletons()
    try:
        return asyncio.run(_run())
    finally:
        _reset_singletons()


# Periodic schedule (configure in Celery beat)
celery_app.conf.beat_schedule = {
    "export-dpo-weekly": {
        "task": "app.tasks.mlops_tasks.export_dpo_pairs",
        "schedule": 604800.0,  # 7 days in seconds
    },
    "drift-detection-weekly": {
        "task": "app.tasks.mlops_tasks.run_drift_detection",
        "schedule": 604800.0,
    },
}
