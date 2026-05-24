"""
RAG retrieval node — CRAG + Self-RAG enhanced.

Flow:
  1. Self-RAG: assess if retrieval is even needed (skip for chitchat)
  2. CRAG: retrieve + score → correct / ambiguous / re-retrieve if irrelevant
  3. user_memory: always retrieved semantically (multi-query, user-scoped)
     — Uses a SUBJECT-ORIENTED memory query (not the raw user question) so
       session summaries ("user studied X") score high against relevant past
       sessions even when stored without a topic tag.
  4. user_docs: retrieved when user has uploaded documents
"""

from __future__ import annotations

import re as _re

import structlog
from langsmith import traceable

from app.agents.state import EduMentorState
from app.rag.crag import crag_retrieve
from app.rag.retriever import retrieve_user_docs, retrieve_user_memory
from app.rag.self_rag import assess_retrieval_need

logger = structlog.get_logger(__name__)

# ── Score threshold calibration ───────────────────────────────────────────────
#
# ms-marco-MiniLM-L-6-v2 is a passage-retrieval cross-encoder trained to score
# (question, answer-passage) pairs.  Session summaries are NOT answer passages
# — they are subject-area descriptors like "user studied image classification
# and CNNs".  When the raw user question is used as the query, the model gives
# low scores even for clearly relevant summaries.
#
# Fix (two-part):
#   1. Build a SUBJECT QUERY from the user's message.  E.g.:
#        "What is the difference between X and Y?" → "X and Y"
#        "Explain gradient descent"               → "gradient descent"
#      This matches the vocabulary of stored summaries much better.
#   2. Keep a low threshold (-0.5) because summaries are short and the
#      cross-encoder logit range for (subject, summary) pairs is lower than
#      for (question, passage) pairs.  Truly off-topic subjects (biology query
#      vs physics summary) still score well below -1.0, so -0.5 safely
#      excludes them.
#
# Logit calibration for (subject-phrase, session-summary) pairs:
#   > 1.0   : near-exact topic overlap
#   0.0–1.0 : same subject area
#  -0.5–0.0 : weakly related (edge of same domain)
#  < -0.5   : different domain → excluded
_MEMORY_SCORE_THRESHOLD = -0.5

# Subject-extraction patterns — progressively more general.
# We try each in order and return the first non-empty match.
_SUBJECT_PATTERNS: list[_re.Pattern[str]] = [
    # "what is/are/does X" → X
    _re.compile(
        r"^what\s+(?:is|are|does|do|was|were|has|have)\s+(?:the\s+)?(?:difference\s+between\s+)?(.+?)[\?\.]?$",
        _re.I,
    ),
    # "explain X" / "describe X" / "define X"
    _re.compile(
        r"^(?:explain|describe|define|tell\s+me\s+about|teach\s+me)\s+(.+?)[\?\.]?$", _re.I
    ),
    # "how does/do X work" → X
    _re.compile(r"^how\s+(?:does|do)\s+(.+?)\s+work[\?\.]?$", _re.I),
    # "difference between X and Y" → "X and Y"
    _re.compile(r"difference\s+between\s+(.+)", _re.I),
    # "X vs Y" → "X vs Y"
    _re.compile(r"(.+?\s+vs\.?\s+.+?)[\?\.]?$", _re.I),
]

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "in",
        "of",
        "to",
        "for",
        "on",
        "at",
        "by",
        "with",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "who",
        "can",
        "could",
        "would",
        "should",
        "will",
        "i",
        "me",
        "my",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "please",
        "tell",
        "me",
        "about",
        "give",
        "some",
        "any",
        "also",
    }
)


def _build_memory_query(user_query: str, topic_name: str) -> str:
    """
    Convert the user's raw question into a subject-phrase that matches
    the vocabulary of stored session summaries.

    Priority:
      1. topic_name if set (most reliable signal)
      2. Regex extraction of subject from question
      3. Noun-phrase heuristic: drop question words + stopwords
      4. Fall back to the original query (safe but less optimal)
    """
    # 1. If the session has an explicit topic, use it directly
    if topic_name and topic_name.strip():
        return f"session about {topic_name.strip()}"

    q = user_query.strip()

    # 2. Pattern match
    for pattern in _SUBJECT_PATTERNS:
        m = pattern.match(q)
        if m:
            subject = m.group(1).strip().rstrip("?.!")
            if len(subject) >= 3:
                return subject

    # 3. Noun-phrase heuristic: strip leading question words and stopwords
    tokens = q.lower().split()
    kept = [t.rstrip("?.!,;") for t in tokens if t.rstrip("?.!,;") not in _STOPWORDS]
    if len(kept) >= 2:
        return " ".join(kept)

    # 4. Fallback
    return q


@traceable(name="retrieval_node", project_name="edumentor")
async def retrieval_node(state: EduMentorState, qdrant_client, bm25_encoder) -> EduMentorState:
    """
    CRAG + Self-RAG retrieval for curriculum_docs, user_memory, and user_docs.
    """
    chunks: list[str] = []
    session_summary = ""
    user_doc_chunks: list[str] = []
    crag_status = "skipped"

    # ── Self-RAG: should we retrieve at all? ─────────────────────────────────
    needs_retrieval = await assess_retrieval_need(state.user_query)

    if needs_retrieval:
        # ── CRAG: curriculum retrieval with quality scoring ───────────────────
        try:
            results, crag_status = await crag_retrieve(
                qdrant_client,
                bm25_encoder,
                state.user_query,
                top_k=3,
                topic_id=state.topic_id or None,
            )
            chunks = [r.payload.get("content", "") for r in results]
            logger.info("crag_retrieval", status=crag_status, chunks=len(chunks))
        except Exception as exc:
            logger.error("crag_failed", error=str(exc))
    else:
        logger.debug("self_rag_skipped_retrieval", query=state.user_query[:60])

    # ── user_memory: subject-query retrieval — no topic exact-match filter ────
    #
    # ROOT CAUSE of empty session_summary:
    #   The cross-encoder (ms-marco-MiniLM-L-6-v2) is trained on (question,
    #   passage) pairs. When the raw user question is scored against a session
    #   summary like "User studied image classification and object detection",
    #   the model gives a low logit (~0.1-0.3) even though the content is
    #   clearly relevant — because summaries are not answer passages.
    #
    # FIX: Build a subject-phrase from the user query (e.g. "object detection
    #   and image classification") and use THAT as the memory retrieval query.
    #   Subject phrases score much higher against same-topic summaries (0.5-2.0)
    #   while cross-domain pairs (biology query vs coding summary) score < -1.0.
    #
    # The threshold is lowered to -0.5 because (subject, summary) logits are
    # systematically lower than (question, passage) logits due to the shorter
    # summary length.
    try:
        if state.user_id:
            # Build a subject-oriented query for memory retrieval
            memory_query = _build_memory_query(
                state.user_query,
                topic_name=state.topic_name or "",
            )
            logger.debug(
                "memory_query_built",
                original=state.user_query[:80],
                memory_query=memory_query,
            )

            memory_results = await retrieve_user_memory(
                qdrant_client,
                bm25_encoder,
                memory_query,  # ← subject phrase, not raw question
                user_id=state.user_id,
                memory_types=["session_summary", "weak_area"],
                # topic= intentionally omitted — semantic similarity + threshold handles relevance
                top_k=5,  # fetch 1 extra to give selection more choice
            )

            if memory_results:
                # Step 1: drop candidates below score threshold
                candidates = [r for r in memory_results if r.score > _MEMORY_SCORE_THRESHOLD]

                if candidates:
                    chosen = None
                    current_topic = (state.topic_name or "").strip().lower()

                    # Step 2: prefer a memory whose stored topic matches current session
                    # e.g. "object detection" query → ML/coding session summary, not biology
                    if current_topic:
                        for r in candidates:
                            stored_topic = (r.payload.get("topic") or "").strip().lower()
                            if stored_topic and (
                                current_topic in stored_topic or stored_topic in current_topic
                            ):
                                chosen = r
                                logger.info(
                                    "memory_topic_match",
                                    stored_topic=stored_topic,
                                    current_topic=current_topic,
                                    score=r.score,
                                )
                                break

                    # Step 3: if no topic match, try subject-keyword match on content
                    # This handles the case where topic_name is "" but the summary
                    # content clearly belongs to the same subject area.
                    if chosen is None:
                        subject_keywords = set(memory_query.lower().split())
                        subject_keywords -= _STOPWORDS
                        for r in candidates:
                            content_lower = (r.payload.get("content") or "").lower()
                            # If 2+ subject keywords appear in the summary content, it's relevant
                            matches = sum(1 for kw in subject_keywords if kw in content_lower)
                            if matches >= 2:
                                chosen = r
                                logger.info(
                                    "memory_keyword_match",
                                    keyword_matches=matches,
                                    score=r.score,
                                    topic=r.payload.get("topic", ""),
                                )
                                break

                    # Step 4: fall back to highest-scoring candidate if nothing matched
                    if chosen is None:
                        chosen = candidates[0]

                    session_summary = chosen.payload.get("content", "")
                    logger.info(
                        "memory_retrieved",
                        score=chosen.score,
                        memory_type=chosen.payload.get("type", ""),
                        topic=chosen.payload.get("topic", ""),
                        chars=len(session_summary),
                        memory_query=memory_query,
                    )
                else:
                    top = memory_results[0]
                    logger.debug(
                        "memory_below_threshold",
                        score=top.score,
                        threshold=_MEMORY_SCORE_THRESHOLD,
                        candidates_checked=len(memory_results),
                        memory_query=memory_query,
                    )
    except Exception as exc:
        logger.error("memory_retrieval_failed", error=str(exc))

    # ── user_docs: only retrieve when the user actually has uploaded docs ─────
    try:
        if state.user_id and needs_retrieval and state.has_user_docs:
            doc_results = await retrieve_user_docs(
                qdrant_client,
                bm25_encoder,
                state.user_query,
                user_id=state.user_id,
                session_id=state.session_id,
                top_k=2,
            )
            user_doc_chunks = [
                r.payload.get("content", "")
                for r in doc_results
                if len(r.payload.get("content", "").strip()) > 30  # drop garbage chunks
            ]
            logger.debug("user_docs_retrieved", chunks=len(user_doc_chunks))
    except Exception as exc:
        logger.error("user_docs_retrieval_failed", error=str(exc))

    # Inject CRAG status into RAG chunks as a header comment for the prompt
    if crag_status == "ambiguous" and chunks:
        chunks[0] = f"[Note: context confidence is moderate]\n{chunks[0]}"
    elif crag_status == "web_corrected" and chunks:
        chunks[0] = f"[Note: retrieved from web search — verify if critical]\n{chunks[0]}"

    return state.model_copy(
        update={
            "rag_chunks": chunks,
            "session_summary": session_summary,
            "user_doc_chunks": user_doc_chunks,
        }
    )
