"""Central settings — pydantic-settings, loaded once at startup."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    APP_NAME: str = "EduMentor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ALLOWED_ORIGINS: list[str] = ["*"]

    DATABASE_URL: str = Field(..., description="asyncpg Neon URL")
    SYNC_DATABASE_URL: str = Field(..., description="psycopg2 Neon URL for Alembic")

    REDIS_URL: str = Field(...)
    REDIS_TOKEN: str = Field(...)

    QDRANT_URL: str = Field(...)
    QDRANT_API_KEY: str = Field(...)

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "edumentor:latest"  # Default value
    OLLAMA_FALLBACK_MODEL: str = "qwen3.5:0.8b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT_SECONDS: int = 120

    JWT_SECRET: str = Field(...)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    CELERY_BROKER_URL: str = Field(...)
    CELERY_RESULT_BACKEND: str | None = Field(default=None)

    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "edumentor"
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    WANDB_API_KEY: str = ""
    WANDB_PROJECT: str = "edumentor"

    # Rate limiting
    VOICE_RATE_LIMIT: int = 10
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    SIGNUP_RATE_LIMIT: int = 5
    LOGIN_RATE_LIMIT: int = 5
    TEXT_RATE_LIMIT: int = 30
    QUIZ_RATE_LIMIT: int = 20
    UPLOAD_RATE_LIMIT: int = 10
    RELOAD_RATE_LIMIT: int = 3
    RELOAD_RATE_LIMIT_WINDOW: int = 3600  # 1 hour window for model reload

    # Caching TTLs (seconds)
    CACHE_TOPICS_TTL: int = 3600  # 1 hour
    CACHE_MASTERY_TTL: int = 300  # 5 minutes
    CACHE_JOB_TTL: int = 2  # 2 seconds
    CACHE_TEXT_TTL: int = 300  # 5 minutes
    CACHE_VOICE_TTL: int = 300  # 5 minutes
    CACHE_QUIZ_TTL: int = 300  # 5 minutes

    WHISPER_MODEL: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    COQUI_MODEL_NAME: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    COQUI_SPEAKER_WAV: str = "assets/speaker_reference.wav"
    COQUI_LANGUAGE: str = "en"

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    IRT_DEFAULT_THETA: float = 0.0
    IRT_DEFAULT_B: float = 0.0
    IRT_DIFFICULTY_WINDOW: float = 0.5

    DSPY_LM_MODEL: str = "ollama/edumentor-phi3.5"
    DSPY_MAX_TOKENS: int = 768

    # Groq — used by quiz_agent only; other agents keep using Ollama
    GROQ_API_KEY: str = Field(default="", description="Groq API key for quiz agent")
    QUIZ_LM_MODEL: str = "groq/meta-llama/llama-4-scout-17b-16e-instruct"

    PROMETHEUS_PORT: int = 9090
    EVIDENTLY_REFERENCE_WINDOW_DAYS: int = 7
    DRIFT_THRESHOLD: float = 0.15

    GITHUB_WEBHOOK_URL: str = ""
    GITHUB_TOKEN: str = ""

    # Shadow testing
    SHADOW_MODEL_ENABLED: bool = False
    SHADOW_MODEL_NAME: str = ""

    # GitHub OAuth
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    GITHUB_REDIRECT_URI: str = "http://localhost:8000/auth/github/callback"
    FRONTEND_URL: str = "http://localhost:3000"

    # Tools
    TAVILY_API_KEY: str = ""
    # JSON list of MCP server configs — see agents/tools/mcp_tools.py for schema.
    # Accepts a Python list OR a JSON string from the .env file:
    #   MCP_SERVERS='[{"name":"calc","type":"stdio","command":"python","args":["-m","my_mcp"]}]'
    MCP_SERVERS: list[Any] = []

    @field_validator("MCP_SERVERS", mode="before")
    @classmethod
    def _parse_mcp_servers(cls, v: Any) -> list[Any]:
        if isinstance(v, str):
            import json

            return json.loads(v)
        return v if v is not None else []


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings: Settings = get_settings()
