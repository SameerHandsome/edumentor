"""Request metrics + logging middleware using centralized metrics registry."""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog
from fastapi import Request, Response

from app.core.metrics import HTTP_ERRORS_TOTAL, HTTP_REQUEST_DURATION, HTTP_REQUESTS_TOTAL

logger = structlog.get_logger(__name__)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    status = str(response.status_code)

    HTTP_REQUEST_DURATION.labels(
        method=request.method, endpoint=endpoint, status_code=status
    ).observe(duration)
    HTTP_REQUESTS_TOTAL.labels(method=request.method, endpoint=endpoint, status_code=status).inc()
    if response.status_code >= 500:
        HTTP_ERRORS_TOTAL.labels(endpoint=endpoint).inc()

    logger.info(
        "request",
        method=request.method,
        path=endpoint,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response
