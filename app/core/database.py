"""Async SQLAlchemy engine + session factory for Neon PostgreSQL."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


def _make_engine():
    return create_async_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=500,
        pool_timeout=30,
        echo=settings.DEBUG,
    )


def _make_session_factory(eng):
    return async_sessionmaker(
        bind=eng,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


engine = _make_engine()
AsyncSessionLocal = _make_session_factory(engine)


def reset_for_celery_task() -> None:
    """
    Dispose the module-level engine and recreate both the engine and the
    session factory.  Must be called BEFORE creating a new event loop in a
    Celery task so that all async connections bind to the fresh loop.

    Background: asyncpg connections are bound to the event loop that was
    current when they were created.  When a Celery task closes its loop and
    the next task opens a new one, any reused connection raises
    'Future attached to a different loop'.  Disposing and recreating the
    engine forces SQLAlchemy to open fresh connections on demand.
    """
    global engine, AsyncSessionLocal
    try:
        # dispose() closes all checked-in (idle) connections synchronously.
        # Checked-out connections are left to GC — safe since the old loop
        # is already closed and those connections are unusable anyway.
        engine.sync_engine.dispose()
    except Exception:
        pass
    engine = _make_engine()
    AsyncSessionLocal = _make_session_factory(engine)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
