"""Upstash Redis client — connection pool + helpers for history and rate limiting."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

_pool: aioredis.ConnectionPool | None = None


def get_redis_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=50,  # raised from 20 — prevents pool saturation under load
            decode_responses=True,
        )
    return _pool


def get_redis() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=get_redis_pool())


@asynccontextmanager
async def redis_client() -> AsyncIterator[aioredis.Redis]:
    """
    Context manager that always returns the connection to the pool via aclose().
    Use this instead of get_redis() directly to prevent connection leaks.

    Without aclose(), each get_redis() call holds a reference inside
    redis.asyncio even after the coroutine exits. After ~100 requests the
    pool reaches max_connections and all reads/writes silently fail.
    """
    r = get_redis()
    try:
        yield r
    finally:
        await r.aclose()


# ── Session history (last 5 messages) ───────────────────────────────────────

SESSION_HISTORY_KEY = "session_history:{session_id}"
MAX_HISTORY = 5


async def push_message(session_id: str, role: str, content: str) -> None:
    async with redis_client() as r:
        key = SESSION_HISTORY_KEY.format(session_id=session_id)
        msg = json.dumps({"role": role, "content": content})
        async with r.pipeline(transaction=True) as pipe:
            pipe.rpush(key, msg)
            pipe.ltrim(key, -MAX_HISTORY, -1)
            pipe.expire(key, 86400)  # 24h TTL
            await pipe.execute()


async def get_history(session_id: str) -> list[dict]:
    async with redis_client() as r:
        key = SESSION_HISTORY_KEY.format(session_id=session_id)
        raw = await r.lrange(key, 0, -1)
        return [json.loads(m) for m in raw]


async def clear_history(session_id: str) -> None:
    async with redis_client() as r:
        await r.delete(SESSION_HISTORY_KEY.format(session_id=session_id))


# ── Rate limiting (sliding window) ──────────────────────────────────────────

RATE_LIMIT_KEY = "rate_limit:{user_id}:voice"
RATE_LIMIT_KEY_TEXT = "rate_limit:{user_id}:text"
RATE_LIMIT_KEY_LOGIN = "rate_limit:{ip}:login"
RATE_LIMIT_KEY_SIGNUP = "rate_limit:{ip}:signup"
RATE_LIMIT_KEY_QUIZ = "rate_limit:{user_id}:quiz"
RATE_LIMIT_KEY_UPLOAD = "rate_limit:{user_id}:upload"
RATE_LIMIT_KEY_RELOAD = "rate_limit:{user_id}:reload"

_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local cutoff = now - window * 1000
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return 0
end
return redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')[2]
"""


async def check_rate_limit(key: str, limit: int, window: int) -> tuple[bool, int]:
    """
    Generic sliding window rate limiter.
    Returns (allowed: bool, retry_after_seconds: int).
    key    — full Redis key e.g. rate_limit:{user_id}:text
    limit  — max requests allowed in window
    window — window size in seconds
    """
    import time

    async with redis_client() as r:
        now_ms = int(time.time() * 1000)
        result = await r.eval(_SLIDING_WINDOW_LUA, 1, key, now_ms, window, limit)  # type: ignore[misc]
        if result == 0:
            return True, 0
        oldest_ms = int(result)
        retry_after = max(0, window - int((now_ms - oldest_ms) / 1000))
        return False, retry_after


# ── Generic cache helpers ────────────────────────────────────────────────────


async def cache_set(key: str, value: Any, ttl: int) -> None:
    """Store JSON-serializable value in Redis with TTL."""
    async with redis_client() as r:
        await r.setex(key, ttl, json.dumps(value))


async def cache_get(key: str) -> Any | None:
    """Return cached value or None if missing/expired."""
    async with redis_client() as r:
        raw = await r.get(key)
        return json.loads(raw) if raw else None


async def cache_delete(key: str) -> None:
    """Invalidate a cache key."""
    async with redis_client() as r:
        await r.delete(key)


# ── Cache keys ───────────────────────────────────────────────────────────────

CACHE_KEY_TOPICS = "cache:topics:all"
CACHE_KEY_MASTERY = "cache:mastery:{user_id}"
CACHE_KEY_JOB = "cache:job:{job_id}:status"
CACHE_KEY_TEXT_RESPONSE = "cache:text:{user_id}:{session_id}:{query_hash}"
CACHE_KEY_VOICE_JOB = "cache:voice:{user_id}:{session_id}:{audio_hash}"
CACHE_KEY_QUIZ = "cache:quiz:{user_id}:{topic_id}:{theta_bucket}"

# ── Feedback / DPO trigger keys ──────────────────────────────────────────────
# Per-session thumbs-down counter.  Expires after 24 h so stale sessions
# don't accumulate counts across days.
# Value: integer (INCR).  TTL: 86400 s.
FEEDBACK_SESSION_THUMBSDOWN = "feedback:thumbsdown:session:{session_id}"

# Global set of user_ids that have completed 3 thumbs-down in one session.
# When this set reaches FEEDBACK_BAD_SESSIONS_THRESHOLD unique users the
# export task is triggered and the set is cleared.
# Members: user_id strings.  No TTL — cleared programmatically after trigger.
FEEDBACK_BAD_SESSIONS_SET = "feedback:bad_sessions_set"

# Threshold constants (import these wherever needed)
FEEDBACK_THUMBSDOWN_PER_SESSION = 3  # how many 👎 in one session = "bad session"
FEEDBACK_BAD_SESSIONS_THRESHOLD = 5  # how many unique students  = trigger export

# ── In-flight request deduplication ─────────────────────────────────────────
# Prevents duplicate message inserts when a slow LLM response (Ollama can take
# minutes locally) causes the user to re-submit the same message.
#
# Key pattern: inflight:text:{user_id}:{session_id}:{query_hash}
# TTL: 600 s (10 min) — longer than any realistic LLM timeout.
# Value: "1" (presence-only flag).
#
# Usage:
#   acquired = await acquire_inflight_lock(user_id, session_id, query_hash)
#   if not acquired:
#       return 409 / early return   ← duplicate request
#   try:
#       ... run LLM + save messages ...
#   finally:
#       await release_inflight_lock(user_id, session_id, query_hash)

INFLIGHT_KEY_TEXT = "inflight:text:{user_id}:{session_id}:{query_hash}"
INFLIGHT_TTL = 600  # 10 minutes safety expiry


async def acquire_inflight_lock(user_id: str, session_id: str, query_hash: str) -> bool:
    """
    Atomically set an in-flight lock for this (user, session, query) triple.
    Returns True  if the lock was acquired (this request should proceed).
    Returns False if another request is already in-flight (caller should return 409).
    Uses SET NX EX — atomic, no race condition.
    """
    key = INFLIGHT_KEY_TEXT.format(user_id=user_id, session_id=session_id, query_hash=query_hash)
    async with redis_client() as r:
        result = await r.set(key, "1", nx=True, ex=INFLIGHT_TTL)
        return result is not None  # SET NX returns None if key already exists


async def release_inflight_lock(user_id: str, session_id: str, query_hash: str) -> None:
    """Release the in-flight lock after the request completes (success or error)."""
    key = INFLIGHT_KEY_TEXT.format(user_id=user_id, session_id=session_id, query_hash=query_hash)
    async with redis_client() as r:
        await r.delete(key)
