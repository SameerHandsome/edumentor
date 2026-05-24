"""
Cross-encoder reranker.

After hybrid retrieval returns a candidate set, the reranker scores each
(query, passage) pair with a cross-encoder model for precise relevance
ordering.  Only the top-k reranked results reach the prompt.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  — small (22 MB), CPU-fast (~5 ms per pair on modern hardware),
    calibrated for passage-level relevance.

The model is loaded lazily on first use and cached for the process lifetime.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import structlog

logger = structlog.get_logger(__name__)

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_TOP_K = 3


class RankedResult(NamedTuple):
    payload: dict[str, Any]
    score: float  # cross-encoder logit (higher = more relevant)
    point_id: str


@functools.lru_cache(maxsize=1)
def _load_cross_encoder():  # type: ignore[return]
    """
    Load the cross-encoder once and cache it.
    Import is deferred so the module can be imported even if sentence-transformers
    is not yet installed (useful for unit tests with mocks).
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        model = CrossEncoder(_RERANKER_MODEL)
        logger.info("cross_encoder_loaded", model=_RERANKER_MODEL)
        return model
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for the reranker. "
            "Install with: pip install sentence-transformers"
        ) from exc


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = _DEFAULT_TOP_K,
) -> list[RankedResult]:
    """
    Re-score `candidates` against `query` and return the top-k.

    Parameters
    ----------
    query      : the user's original (un-transformed) query text
    candidates : list of dicts, each must have keys:
                   - "payload"  : dict with at least "content" key
                   - "point_id" : Qdrant point UUID string
    top_k      : number of results to return after reranking

    Returns
    -------
    List of RankedResult sorted descending by cross-encoder score.
    """
    if not candidates:
        return []

    model = _load_cross_encoder()

    pairs = [(query, c["payload"].get("content", "")) for c in candidates]

    scores: list[float] = model.predict(pairs).tolist()

    ranked = sorted(
        [
            RankedResult(
                payload=c["payload"],
                score=float(s),
                point_id=c["point_id"],
            )
            for c, s in zip(candidates, scores)
        ],
        key=lambda r: r.score,
        reverse=True,
    )

    logger.debug(
        "rerank_complete",
        candidates=len(candidates),
        top_k=top_k,
        top_score=ranked[0].score if ranked else None,
    )
    return ranked[:top_k]


async def prewarm_reranker() -> None:
    """
    Load the cross-encoder model at app startup so the first real query
    doesn't pay the 1–3 s model-load penalty.

    Call this from the FastAPI lifespan with asyncio.to_thread because
    CrossEncoder.__init__ downloads and loads model weights synchronously.

    Example (in app/main.py lifespan):
        import asyncio
        from app.rag.reranker import prewarm_reranker
        await prewarm_reranker()
    """
    import asyncio as _asyncio

    await _asyncio.to_thread(_load_cross_encoder)
    logger.info("reranker_prewarmed", model=_RERANKER_MODEL)
