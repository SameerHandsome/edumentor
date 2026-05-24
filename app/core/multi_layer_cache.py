"""
Multi-layer cache — L1 (in-process TTLCache) + L2 (Redis).

Lookup order:
    1. L1 (cachetools TTLCache, in-process, sub-millisecond)
       → hit: return immediately, no Redis call.
    2. L2 (Redis, shared across all workers/replicas)
       → hit: backfill L1, return value.
    3. miss: caller computes the value, then writes both L1 and L2.

Why two layers?
    • L1 eliminates Redis round-trips for hot keys polled repeatedly within the
      same process (e.g. job-status polling, mastery reads on every request).
    • L2 keeps caches coherent across all FastAPI workers and Celery processes.

Thread-safety:
    cachetools TTLCache is NOT thread-safe by itself.  We protect every access
    with threading.Lock (safe for both sync and asyncio, because asyncio runs in
    a single OS thread and we never await while holding the lock).

L1 TTLs are intentionally shorter than L2 TTLs so stale process-local data
is evicted well before the Redis copy expires.

Public API
----------
    ml_cache_get(key)              -> Any | None
    ml_cache_set(key, value, l2_ttl, l1_ttl=None)
    ml_cache_delete(key)

    # Pre-configured helpers (match existing redis_client cache-key constants)
    ml_get_topics()
    ml_set_topics(value)
    ml_get_mastery(user_id)
    ml_set_mastery(user_id, value)
    ml_get_job(job_id)
    ml_set_job(job_id, value)
    ml_get_text(user_id, session_id, query_hash)
    ml_set_text(user_id, session_id, query_hash, value)
    ml_get_quiz(user_id, topic_id, theta_bucket, num_questions)
    ml_set_quiz(user_id, topic_id, theta_bucket, num_questions, value)
"""

from __future__ import annotations

import datetime
import threading
import uuid
from typing import Any

import structlog
from cachetools import TTLCache

from app.core.config import settings
from app.core.redis_client import (
    CACHE_KEY_JOB,
    CACHE_KEY_MASTERY,
    CACHE_KEY_QUIZ,
    CACHE_KEY_TEXT_RESPONSE,
    CACHE_KEY_TOPICS,
    cache_delete,
    cache_get,
    cache_set,
)

logger = structlog.get_logger(__name__)

# ── L1 TTL constants (seconds) — always shorter than the L2 Redis TTLs ──────
_L1_TTL_TOPICS = 120  # L2 = 3600s
_L1_TTL_MASTERY = 30  # L2 = 300s
_L1_TTL_JOB = 1  # L2 = 2s
_L1_TTL_TEXT = 30  # L2 = 300s
_L1_TTL_QUIZ = 30  # L2 = 300s

# ── L1 cache instances ───────────────────────────────────────────────────────
_L1_TOPICS: TTLCache = TTLCache(maxsize=10, ttl=_L1_TTL_TOPICS)
_L1_MASTERY: TTLCache = TTLCache(maxsize=2048, ttl=_L1_TTL_MASTERY)
_L1_JOB: TTLCache = TTLCache(maxsize=4096, ttl=_L1_TTL_JOB)
_L1_TEXT: TTLCache = TTLCache(maxsize=2048, ttl=_L1_TTL_TEXT)
_L1_QUIZ: TTLCache = TTLCache(maxsize=2048, ttl=_L1_TTL_QUIZ)
_L1_GENERIC: TTLCache = TTLCache(maxsize=1024, ttl=30)

_L1_LOCK = threading.Lock()


def _json_serializable(obj: Any) -> Any:
    """
    Recursively convert a value so it is safe to pass to json.dumps.

    Handles the types Pydantic's model_dump() leaves as native Python objects:
      • uuid.UUID      → str
      • datetime / date / time → ISO-format str
      • dict / list    → recurse
    Everything else is returned unchanged (json.dumps handles int, float,
    str, bool, None natively).
    """
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime.datetime | datetime.date | datetime.time):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_serializable(i) for i in obj]
    return obj


def _pick_l1(key: str) -> TTLCache:
    if key == CACHE_KEY_TOPICS:
        return _L1_TOPICS
    if key.startswith("cache:mastery:"):
        return _L1_MASTERY
    if key.startswith("cache:job:"):
        return _L1_JOB
    if key.startswith("cache:text:"):
        return _L1_TEXT
    if key.startswith("cache:quiz:"):
        return _L1_QUIZ
    return _L1_GENERIC


async def ml_cache_get(key: str) -> Any | None:
    l1 = _pick_l1(key)
    with _L1_LOCK:
        if key in l1:
            logger.debug("cache_hit_l1", key=key)
            return l1[key]
    value = await cache_get(key)
    if value is not None:
        logger.debug("cache_hit_l2", key=key)
        with _L1_LOCK:
            l1[key] = value
        return value
    logger.debug("cache_miss", key=key)
    return None


async def ml_cache_set(key: str, value: Any, l2_ttl: int, l1_ttl: int | None = None) -> None:
    l1 = _pick_l1(key)
    # Normalise value before storing in L1 and L2 so both layers hold the
    # same JSON-safe representation (avoids UUID/datetime surprises on reads).
    safe_value = _json_serializable(value)
    with _L1_LOCK:
        l1[key] = safe_value
    await cache_set(key, safe_value, l2_ttl)
    logger.debug("cache_set_both_layers", key=key, l2_ttl=l2_ttl)


async def ml_cache_delete(key: str) -> None:
    l1 = _pick_l1(key)
    with _L1_LOCK:
        l1.pop(key, None)
    await cache_delete(key)
    logger.debug("cache_deleted_both_layers", key=key)


# ── Pre-configured helpers ────────────────────────────────────────────────────


async def ml_get_topics() -> Any | None:
    return await ml_cache_get(CACHE_KEY_TOPICS)


async def ml_set_topics(value: Any) -> None:
    await ml_cache_set(CACHE_KEY_TOPICS, value, settings.CACHE_TOPICS_TTL)


async def ml_get_mastery(user_id: str) -> Any | None:
    return await ml_cache_get(CACHE_KEY_MASTERY.format(user_id=user_id))


async def ml_set_mastery(user_id: str, value: Any) -> None:
    await ml_cache_set(CACHE_KEY_MASTERY.format(user_id=user_id), value, settings.CACHE_MASTERY_TTL)


async def ml_delete_mastery(user_id: str) -> None:
    await ml_cache_delete(CACHE_KEY_MASTERY.format(user_id=user_id))


async def ml_get_job(job_id: str) -> Any | None:
    return await ml_cache_get(CACHE_KEY_JOB.format(job_id=job_id))


async def ml_set_job(job_id: str, value: Any) -> None:
    await ml_cache_set(
        CACHE_KEY_JOB.format(job_id=job_id), value, settings.CACHE_JOB_TTL, l1_ttl=_L1_TTL_JOB
    )


async def ml_get_text(user_id: str, session_id: str, query_hash: str) -> Any | None:
    return await ml_cache_get(
        CACHE_KEY_TEXT_RESPONSE.format(
            user_id=user_id, session_id=session_id, query_hash=query_hash
        )
    )


async def ml_set_text(user_id: str, session_id: str, query_hash: str, value: Any) -> None:
    await ml_cache_set(
        CACHE_KEY_TEXT_RESPONSE.format(
            user_id=user_id, session_id=session_id, query_hash=query_hash
        ),
        value,
        settings.CACHE_TEXT_TTL,
    )


async def ml_get_quiz(
    user_id: str, topic_id: str, theta_bucket: str, num_questions: int = 10
) -> Any | None:
    key = (
        CACHE_KEY_QUIZ.format(user_id=user_id, topic_id=topic_id, theta_bucket=theta_bucket)
        + f":n{num_questions}"
    )
    return await ml_cache_get(key)


async def ml_set_quiz(
    user_id: str, topic_id: str, theta_bucket: str, value: Any, num_questions: int = 10
) -> None:
    key = (
        CACHE_KEY_QUIZ.format(user_id=user_id, topic_id=topic_id, theta_bucket=theta_bucket)
        + f":n{num_questions}"
    )
    await ml_cache_set(key, value, settings.CACHE_QUIZ_TTL)
