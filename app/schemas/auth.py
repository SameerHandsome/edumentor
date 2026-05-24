"""Auth request/response schemas — password auth + GitHub OAuth."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1, max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


# ── GitHub OAuth schemas ──────────────────────────────────────────────────────


class GitHubOAuthURLResponse(BaseModel):
    """Returned by GET /auth/github/login — frontend redirects user here."""

    url: str


class GitHubCallbackResponse(BaseModel):
    """Returned after successful GitHub OAuth flow."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    is_new_user: bool  # True = account just created, False = existing user logged in
    user_id: str
    full_name: str
    avatar_url: str
