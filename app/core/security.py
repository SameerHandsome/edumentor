"""JWT creation, verification, and password hashing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import bcrypt
import structlog
from jose import JWTError, jwt

from app.core.config import settings

logger = structlog.get_logger(__name__)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: str | UUID, extra: dict | None = None) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "type": "access",
        **(extra or {}),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str | UUID) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, expected_type: str = "access") -> dict:
    """Decode and validate a JWT. Raises JWTError on failure."""
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    if payload.get("type") != expected_type:
        raise JWTError(f"Expected token type '{expected_type}'")
    return payload


def get_user_id_from_token(token: str) -> str | None:
    try:
        payload = decode_token(token)
        return payload.get("sub")
    except JWTError:
        return None
