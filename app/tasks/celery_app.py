"""Celery application factory — CloudAMQP broker, no result backend.

Job status is tracked entirely via the `jobs` table in Postgres and the
multi-layer app cache (L1 TTLCache + L2 Upstash Redis).  Celery's own
result backend is disabled — it was an extra write to the same Redis that
provided no value and crashed the worker when the connection was lost.
"""

from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue

from app.core.config import settings

celery_app = Celery(
    "edumentor",
    broker=settings.CELERY_BROKER_URL,
    backend=None,
    include=[
        "app.tasks.voice_tasks",
        "app.tasks.session_tasks",
        "app.tasks.mlops_tasks",
        "app.tasks.doc_tasks",
    ],
)

# ── Dead-letter exchange — exhausted tasks land here instead of vanishing ──
_DLX = Exchange("dlx", type="direct", durable=True)


def _q(name: str, max_length: int) -> Queue:
    return Queue(
        name,
        Exchange(name, type="direct", durable=True),
        routing_key=name,
        durable=True,
        queue_arguments={
            "x-max-length": max_length,
            "x-overflow": "reject-publish",
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": f"dlq.{name}",
        },
    )


def _dlq(name: str) -> Queue:
    return Queue(
        f"dlq.{name}",
        _DLX,
        routing_key=f"dlq.{name}",
        durable=True,
    )


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_ignore_result=True,  # no result backend — status lives in Postgres + app cache
    task_store_errors_even_if_ignored=False,
    task_queues=(
        _q("voice", max_length=500),
        _q("session", max_length=500),
        _q("mlops", max_length=200),
        _dlq("voice"),
        _dlq("session"),
        _dlq("mlops"),
    ),
    task_routes={
        "app.tasks.voice_tasks.*": {"queue": "voice"},
        "app.tasks.session_tasks.*": {"queue": "session"},
        "app.tasks.mlops_tasks.*": {"queue": "mlops"},
    },
)
