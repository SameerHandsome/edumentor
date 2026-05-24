"""
EduMentor FastAPI application entry point.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server

from app.api.routes import auth, curriculum, mlops, quiz, tutor, user
from app.core.config import settings
from app.core.middleware import metrics_middleware

logger = structlog.get_logger(__name__)

# Configure LangSmith tracing
if settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT


async def _build_bm25_corpus_from_qdrant(qdrant_client) -> list[str]:
    """
    Scroll curriculum_docs and collect chunk content for BM25 fitting.

    Why: fitting BM25 on the ACTUAL ingested text gives a vocabulary that
    covers every domain-specific term in your PDFs (nucleus, mitosis, chloroplast,
    etc.) rather than a small hardcoded seed that leaves most tokens OOV.

    We take the first 500 chars of each stored chunk — enough for full vocabulary
    coverage without pulling megabytes of payload over the network.

    Falls back to a minimal hardcoded corpus if the collection is empty so
    the server still starts cleanly before any documents are ingested.
    """
    from app.rag.collections import COLLECTION_CURRICULUM

    samples: list[str] = []
    offset = None
    try:
        while True:
            points, next_offset = await qdrant_client.scroll(
                collection_name=COLLECTION_CURRICULUM,
                limit=250,
                offset=offset,
                with_payload=["content"],
                with_vectors=False,
            )
            for p in points:
                content = (p.payload or {}).get("content", "")
                if content.strip():
                    samples.append(content[:500])
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        logger.warning("bm25_qdrant_scroll_failed", error=str(exc))

    if not samples:
        logger.warning("bm25_collection_empty_using_fallback_corpus")
        # Minimal fallback so the server starts before any ingestion has run
        samples = [
            "mathematics algebra geometry calculus equations variables polynomials",
            "biology cell nucleus atom photosynthesis mitosis genetics dna protein",
            "physics motion force energy momentum velocity acceleration gravity",
            "chemistry atoms molecules bonds reactions periodic table elements",
            "student learning knowledge understanding concept practice problem",
        ]
    return samples


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize DB, Qdrant, BM25, LangGraph. Shutdown: cleanup."""
    logger.info("edumentor_starting", version=settings.APP_VERSION)

    # Initialize Qdrant collections
    from app.rag.collections import get_qdrant_client, init_collections

    qdrant = await get_qdrant_client()
    await init_collections(qdrant)

    # Build BM25 vocabulary from the actual ingested Qdrant content so every
    # domain-specific token (nucleus, mitosis, chloroplast, etc.) is in-vocab.
    # This replaces the old hardcoded seed corpus which left most real tokens OOV.
    from app.rag.bm25 import BM25Encoder

    bm25_corpus = await _build_bm25_corpus_from_qdrant(qdrant)
    bm25 = BM25Encoder()
    bm25.fit(bm25_corpus)
    logger.info("bm25_fitted", vocab_size=bm25.vocab_size(), corpus_docs=len(bm25_corpus))

    # Pre-warm the cross-encoder reranker so the first real query doesn't
    # pay the 1-3s model-load penalty. CrossEncoder.__init__ is synchronous
    # so we run it in a thread.
    from app.rag.reranker import prewarm_reranker

    await prewarm_reranker()

    # Inject RAG deps into LangGraph
    from app.agents.graph import get_graph, set_rag_dependencies

    set_rag_dependencies(qdrant, bm25)
    get_graph()  # compile once

    # Load MCP tools (local stdio + remote SSE/HTTP servers)
    from app.agents.tools.mcp_tools import init_mcp_tools

    await init_mcp_tools()

    # Start Prometheus metrics server on side port
    try:
        start_http_server(settings.PROMETHEUS_PORT)
        logger.info("prometheus_started", port=settings.PROMETHEUS_PORT)
    except OSError:
        logger.warning("prometheus_port_busy", port=settings.PROMETHEUS_PORT)

    logger.info("edumentor_ready")
    yield
    logger.info("edumentor_shutting_down")
    await qdrant.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(metrics_middleware)

# Routers
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(curriculum.router)
app.include_router(tutor.router)
app.include_router(quiz.router)
app.include_router(mlops.router, prefix="/mlops")


# Public health (no auth)
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
