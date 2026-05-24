"""MLOps admin routes — health, metrics, model reload."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.routes.deps import require_admin
from app.core.config import settings
from app.core.redis_client import RATE_LIMIT_KEY_RELOAD, check_rate_limit

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["mlops"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "edumentor"}


@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/model/status")
async def model_status(_: str = Depends(require_admin)):
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            models = resp.json().get("models", [])
            loaded = any(m["name"].startswith("edumentor") for m in models)
        return {
            "model": settings.OLLAMA_MODEL,
            "loaded": loaded,
            "models": [m["name"] for m in models],
        }
    except Exception as exc:
        return {"model": settings.OLLAMA_MODEL, "loaded": False, "error": str(exc)}


@router.post("/model/reload")
async def model_reload(user_id: str = Depends(require_admin)):
    """Pull latest GGUF into Ollama — called by retraining CI/CD pipeline."""
    allowed, retry_after = await check_rate_limit(
        key=RATE_LIMIT_KEY_RELOAD.format(user_id=user_id),
        limit=settings.RELOAD_RATE_LIMIT,
        window=settings.RELOAD_RATE_LIMIT_WINDOW,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "retry_after": retry_after},
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/pull",
                json={"name": settings.OLLAMA_MODEL, "stream": False},
            )
            resp.raise_for_status()
        logger.info("model_reloaded", model=settings.OLLAMA_MODEL)
        return {"status": "reloaded", "model": settings.OLLAMA_MODEL}
    except Exception as exc:
        logger.error("model_reload_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
