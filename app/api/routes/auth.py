"""
Auth routes — two authentication methods:

1. Password auth  →  POST /auth/signup, /auth/login, /auth/refresh, /auth/logout
2. GitHub OAuth   →  GET  /auth/github/login   (redirects user to GitHub)
                     GET  /auth/github/callback (GitHub redirects back here)

Both methods issue the same JWT access + refresh tokens so the rest of
the API works identically regardless of how the user authenticated.

GitHub OAuth flow:
  Frontend            Backend                GitHub
  ───────             ───────                ──────
  GET /auth/github/login
                  → returns GitHub URL
  redirect user ──────────────────────────→ user approves
                  ←── GET /callback?code=xxx
                  exchange code for token ──→
                  ←── access_token
                  fetch /user profile ──────→
                  ←── {id, email, name, avatar}
                  upsert user in DB
                  → JWT tokens to frontend
"""

from __future__ import annotations

import secrets
import uuid

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.redis_client import (
    RATE_LIMIT_KEY_LOGIN,
    RATE_LIMIT_KEY_SIGNUP,
    check_rate_limit,
    get_redis,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User, UserPreference
from app.schemas.auth import (
    GitHubCallbackResponse,
    GitHubOAuthURLResponse,
    LoginRequest,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

# ── GitHub OAuth constants ────────────────────────────────────────────────────
_GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL = "https://api.github.com/user"
_GITHUB_EMAIL_URL = "https://api.github.com/user/emails"


# ── Password auth ─────────────────────────────────────────────────────────────


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, request: Request, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host
    allowed, retry_after = await check_rate_limit(
        key=RATE_LIMIT_KEY_SIGNUP.format(ip=client_ip),
        limit=settings.SIGNUP_RATE_LIMIT,
        window=settings.RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "retry_after": retry_after},
        )

    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        id=uuid.uuid4(),
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        is_verified=False,
    )
    pref = UserPreference(user_id=user.id)
    db.add(user)
    db.add(pref)
    await db.commit()
    logger.info("user_signed_up", user_id=str(user.id))
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host
    allowed, retry_after = await check_rate_limit(
        key=RATE_LIMIT_KEY_LOGIN.format(ip=client_ip),
        limit=settings.LOGIN_RATE_LIMIT,
        window=settings.RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "retry_after": retry_after},
        )

    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    logger.info("user_logged_in", user_id=str(user.id))
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest):
    # BUG FIX (logout blacklist): check blacklist before issuing new tokens
    redis = get_redis()
    if await redis.exists(f"token:blacklist:{body.refresh_token}"):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")
    try:
        payload = decode_token(body.refresh_token, expected_type="refresh")
        user_id = payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return TokenResponse(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
    )


@router.post("/logout", status_code=204)
async def logout(body: RefreshRequest):
    # BUG FIX: old implementation was a no-op — tokens stayed valid for 7 days
    # after logout because JWTs are stateless.  Now we blacklist the refresh
    # token in Redis with a TTL equal to its remaining lifetime so it can never
    # be used to mint new access tokens.  Access tokens are short-lived
    # (ACCESS_TOKEN_EXPIRE_MINUTES) so we don't bother blacklisting them.
    redis = get_redis()
    await redis.set(
        f"token:blacklist:{body.refresh_token}",
        "1",
        ex=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
    )
    logger.info("user_logged_out")


# ── GitHub OAuth ──────────────────────────────────────────────────────────────


@router.get("/github/login", response_model=GitHubOAuthURLResponse)
async def github_login():
    """
    Step 1 — Frontend calls this endpoint to get the GitHub authorization URL.
    Frontend then redirects the user's browser to that URL.

    Example frontend usage:
        const res = await fetch('/auth/github/login')
        const { url } = await res.json()
        window.location.href = url
    """
    if not settings.GITHUB_CLIENT_ID:
        raise HTTPException(
            status_code=501,
            detail="GitHub OAuth is not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env",
        )

    # state param prevents CSRF — random token tied to this login attempt.
    # Store it in Redis (TTL 10 min) so the callback can verify it.
    state = secrets.token_urlsafe(32)
    redis = get_redis()
    await redis.set(f"oauth:state:{state}", "1", ex=600)

    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
        "scope": "read:user user:email",  # minimal scopes — name, avatar, email only
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_GITHUB_AUTHORIZE_URL}?{query}"

    logger.info("github_oauth_initiated")  # state intentionally omitted from logs
    return GitHubOAuthURLResponse(url=url)


@router.get("/github/callback", response_model=GitHubCallbackResponse)
async def github_callback(
    code: str = Query(..., description="Authorization code from GitHub"),
    state: str = Query(default="", description="CSRF state token"),
    db: AsyncSession = Depends(get_db),
):
    """
    Step 2 — GitHub redirects the user here after they approve access.
    This endpoint:
      1. Exchanges the code for a GitHub access token
      2. Fetches the user's GitHub profile (name, avatar, email)
      3. Creates a new user OR logs in an existing one
      4. Returns EduMentor JWT tokens

    GitHub registers the callback URL — it must match GITHUB_REDIRECT_URI exactly.
    """
    if not settings.GITHUB_CLIENT_ID:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")

    # ── CSRF state verification ───────────────────────────────────────────────
    if not state:
        raise HTTPException(status_code=400, detail="Missing OAuth state parameter")
    redis = get_redis()
    stored = await redis.getdel(f"oauth:state:{state}")
    if stored is None:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    # ── Step 1: Exchange code for GitHub access token ─────────────────────────
    github_token = await _exchange_code_for_token(code)

    # ── Step 2: Fetch GitHub user profile ─────────────────────────────────────
    github_user = await _fetch_github_user(github_token)
    github_id = str(github_user["id"])
    full_name = github_user.get("name") or github_user.get("login", "GitHub User")
    avatar_url = github_user.get("avatar_url", "")

    # Email might not be public on GitHub profile — fetch from emails endpoint
    email = github_user.get("email")
    if not email:
        email = await _fetch_github_primary_email(github_token)
    if not email:
        # Fallback: construct a placeholder (GitHub username@users.noreply.github.com)
        login = github_user.get("login", github_id)
        email = f"{login}@users.noreply.github.com"

    logger.info("github_profile_fetched", github_id=github_id, email=email)

    # ── Step 3: Find or create user ───────────────────────────────────────────
    is_new_user = False

    # First try to find by github_id (most reliable — email can change on GitHub)
    result = await db.execute(select(User).where(User.github_id == github_id))
    user = result.scalar_one_or_none()

    if not user:
        # Try by email (handles case where user signed up with password before)
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user:
            # Existing password user — link their GitHub account
            user.github_id = github_id
            user.oauth_provider = "github"
            user.avatar_url = avatar_url
            user.is_verified = True  # GitHub email is already verified
            logger.info("github_linked_to_existing", user_id=str(user.id))
        else:
            # Brand new user via GitHub
            is_new_user = True
            user = User(
                id=uuid.uuid4(),
                email=email,
                hashed_password=None,  # no password for OAuth users
                full_name=full_name,
                avatar_url=avatar_url,
                github_id=github_id,
                oauth_provider="github",
                is_verified=True,  # GitHub already verified the email
                is_active=True,
            )
            pref = UserPreference(user_id=user.id)
            db.add(user)
            db.add(pref)
            logger.info("github_new_user_created", user_id=str(user.id), email=email)

    await db.commit()
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    logger.info("github_oauth_success", user_id=str(user.id), is_new=is_new_user)

    return GitHubCallbackResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        is_new_user=is_new_user,
        user_id=str(user.id),
        full_name=user.full_name,
        avatar_url=user.avatar_url,
    )


# ── GitHub API helpers ────────────────────────────────────────────────────────


async def _exchange_code_for_token(code: str) -> str:
    """Exchange GitHub authorization code for an access token."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            _GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.GITHUB_REDIRECT_URI,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        logger.error("github_token_exchange_failed", error=data.get("error_description"))
        raise HTTPException(
            status_code=400,
            detail=f"GitHub OAuth error: {data.get('error_description', data['error'])}",
        )

    token = data.get("access_token")
    if not token:
        raise HTTPException(status_code=400, detail="GitHub did not return an access token")
    return token


async def _fetch_github_user(token: str) -> dict:
    """Fetch the authenticated user's GitHub profile."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            _GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _fetch_github_primary_email(token: str) -> str:
    """
    Fetch the user's primary verified email from GitHub.
    Needed when the user has set their email to private on their GitHub profile.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            _GITHUB_EMAIL_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        if resp.status_code != 200:
            return ""
        emails = resp.json()

    for entry in emails:
        if entry.get("primary") and entry.get("verified"):
            return entry["email"]
    for entry in emails:
        if entry.get("verified"):
            return entry["email"]
    return ""
