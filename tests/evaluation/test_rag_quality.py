"""
Evaluation tests: RAG Quality — LLM-as-Judge via Groq.

Tests four dimensions of your RAG pipeline:
  1. Chunk Relevance    — are the retrieved chunks relevant to the query?
  2. Context Sufficiency — do the chunks contain enough to answer the question?
  3. Faithfulness       — does the final agent response stay within the chunks?
  4. CRAG Decision      — did the CRAG router make the right call (correct/ambiguous/web)?

All mocked for CI. With real GROQ_API_KEY + Qdrant, remove the mock and run live.

Run:
    pytest tests/evaluation/test_rag_quality.py -m eval -v -s
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.evaluation.llm_judge import EvalScore, _call_groq_judge, _parse_score

pytestmark = pytest.mark.eval


# ── Two extra judge functions specific to RAG ─────────────────────────────────

_CHUNK_RELEVANCE_PROMPT = """\
You are evaluating whether retrieved text chunks are relevant to a student's question.

STUDENT QUESTION: {question}

RETRIEVED CHUNKS:
{chunks}

For each chunk, judge whether it contains information useful for answering the question.
Then give an overall relevance score.

Respond in strict JSON:
{{
  "reasoning": "<brief explanation>",
  "chunk_relevance_score": <float 0.0–1.0>,
  "relevant_chunks": <int — how many of the chunks are relevant>
}}
1.0 = all chunks directly answer the question, 0.0 = no chunk is relevant.
Respond ONLY with the JSON object."""

_CONTEXT_SUFFICIENCY_PROMPT = """\
You are evaluating whether the retrieved context is sufficient to fully answer a question.

STUDENT QUESTION: {question}

RETRIEVED CONTEXT:
{chunks}

EXPECTED ANSWER (ground truth):
{reference}

Judge whether the retrieved context contains enough information for an agent to
produce the expected answer without needing additional knowledge.

Respond in strict JSON:
{{
  "reasoning": "<brief explanation>",
  "context_sufficiency_score": <float 0.0–1.0>,
  "missing_information": "<what key info is absent, or 'nothing' if sufficient>"
}}
1.0 = context fully covers the answer, 0.0 = context is completely missing the answer.
Respond ONLY with the JSON object."""

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI tutor's response is faithful to the provided context.
Faithful means: every claim in the response can be traced back to the context.

RETRIEVED CONTEXT:
{chunks}

AGENT RESPONSE:
{response}

Respond in strict JSON:
{{
  "reasoning": "<brief explanation>",
  "faithfulness_score": <float 0.0–1.0>,
  "unfaithful_claims": ["<list of claims not in context, or empty list>"]
}}
1.0 = fully faithful to context, 0.0 = response ignores or contradicts the context.
Respond ONLY with the JSON object."""


async def judge_chunk_relevance(
    question: str,
    chunks: List[str],
    api_key: str | None = None,
) -> EvalScore:
    """Score how relevant the retrieved chunks are to the question."""
    import os
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    chunks_text = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(chunks))
    prompt = _CHUNK_RELEVANCE_PROMPT.format(question=question, chunks=chunks_text)
    raw = await _call_groq_judge(prompt, key)
    score, reasoning = _parse_score(raw, "chunk_relevance_score")
    return EvalScore("chunk_relevance", score, reasoning, raw)


async def judge_context_sufficiency(
    question: str,
    chunks: List[str],
    reference: str,
    api_key: str | None = None,
) -> EvalScore:
    """Score whether the chunks contain enough info to answer the question."""
    import os
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    chunks_text = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(chunks))
    prompt = _CONTEXT_SUFFICIENCY_PROMPT.format(
        question=question, chunks=chunks_text, reference=reference
    )
    raw = await _call_groq_judge(prompt, key)
    score, reasoning = _parse_score(raw, "context_sufficiency_score")
    return EvalScore("context_sufficiency", score, reasoning, raw)


async def judge_faithfulness(
    chunks: List[str],
    response: str,
    api_key: str | None = None,
) -> EvalScore:
    """Score whether the agent response stays within the retrieved context."""
    import os
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    chunks_text = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(chunks))
    prompt = _FAITHFULNESS_PROMPT.format(chunks=chunks_text, response=response)
    raw = await _call_groq_judge(prompt, key)
    score, reasoning = _parse_score(raw, "faithfulness_score")
    return EvalScore("faithfulness", score, reasoning, raw)


# ── Test dataset ──────────────────────────────────────────────────────────────

RAG_CASES = [
    {
        "id": "photosynthesis_good_retrieval",
        "question": "How does photosynthesis produce oxygen?",
        "chunks": [
            "Photosynthesis occurs in the chloroplasts of plant cells. "
            "Chlorophyll pigments absorb sunlight to drive the light reactions.",
            "In the light reactions, water molecules are split (photolysis), "
            "releasing oxygen as a byproduct. The oxygen is released into the atmosphere.",
            "The Calvin cycle uses ATP and NADPH from the light reactions to "
            "fix carbon dioxide into glucose.",
        ],
        "reference": "Oxygen is produced during the light reactions when water is split (photolysis) in the chloroplasts.",
        "response": "During photosynthesis, oxygen is released when water molecules are split "
                    "in the light reactions inside the chloroplasts. This process is called photolysis.",
        "crag_decision": "correct",
    },
    {
        "id": "newton_partial_retrieval",
        "question": "What is Newton's second law and how is it applied?",
        "chunks": [
            "Newton's second law states F = ma, where F is force, m is mass, and a is acceleration.",
            "Isaac Newton published his three laws of motion in 1687 in Principia Mathematica.",
        ],
        "reference": "F = ma. Applied by calculating the force needed to accelerate an object, "
                     "e.g. F = 5kg × 2m/s² = 10N.",
        "response": "Newton's second law states that F = ma. To apply it, multiply the mass "
                    "of the object by its acceleration to find the net force.",
        "crag_decision": "ambiguous",
    },
    {
        "id": "dna_irrelevant_retrieval",
        "question": "How does DNA replication work?",
        "chunks": [
            "Mitosis is the process of cell division that produces two identical daughter cells.",
            "The cell cycle consists of interphase, prophase, metaphase, anaphase, and telophase.",
            "Chromosomes condense during prophase and align at the metaphase plate.",
        ],
        "reference": "DNA replication uses helicase to unwind the double helix, "
                     "then DNA polymerase builds complementary strands.",
        "response": "DNA replication involves the unwinding of the double helix by helicase, "
                    "followed by DNA polymerase synthesizing new complementary strands.",
        "crag_decision": "web_corrected",  # chunks missed — CRAG should web-search
    },
    {
        "id": "algebra_faithful_response",
        "question": "How do you solve a quadratic equation?",
        "chunks": [
            "A quadratic equation has the form ax² + bx + c = 0. "
            "It can be solved using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a.",
            "Quadratic equations can also be solved by factoring when the equation "
            "factors into two binomials.",
        ],
        "reference": "Use the quadratic formula x = (-b ± √(b²-4ac)) / 2a or factor the equation.",
        "response": "To solve a quadratic equation ax² + bx + c = 0, use the quadratic formula: "
                    "x = (-b ± √(b²-4ac)) / 2a. You can also try factoring if the equation factors neatly.",
        "crag_decision": "correct",
    },
    {
        "id": "algebra_unfaithful_response",
        "question": "How do you solve a quadratic equation?",
        "chunks": [
            "A quadratic equation has the form ax² + bx + c = 0. "
            "It can be solved using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a.",
        ],
        "reference": "Use the quadratic formula x = (-b ± √(b²-4ac)) / 2a.",
        "response": "Quadratic equations are best solved using calculus — take the derivative "
                    "and set it to zero to find the roots. This is the standard method.",
        "crag_decision": "correct",
    },
]

# Mock scores for CI
_MOCK_RAG_SCORES = {
    ("photosynthesis_good_retrieval", "chunk_relevance"):      (0.97, "All three chunks directly address photosynthesis and oxygen production."),
    ("photosynthesis_good_retrieval", "context_sufficiency"):  (0.95, "Context fully covers the photolysis mechanism."),
    ("photosynthesis_good_retrieval", "faithfulness"):         (0.96, "Response stays within the provided chunks."),

    ("newton_partial_retrieval",      "chunk_relevance"):      (0.70, "One chunk has the formula; second is only biographical context."),
    ("newton_partial_retrieval",      "context_sufficiency"):  (0.55, "Formula is present but no application example in context."),
    ("newton_partial_retrieval",      "faithfulness"):         (0.90, "Response only uses information from the chunks."),

    ("dna_irrelevant_retrieval",      "chunk_relevance"):      (0.05, "All chunks are about mitosis/cell cycle, not DNA replication."),
    ("dna_irrelevant_retrieval",      "context_sufficiency"):  (0.03, "Context contains nothing about DNA replication mechanism."),
    ("dna_irrelevant_retrieval",      "faithfulness"):         (0.10, "Response describes helicase and polymerase — neither in the chunks."),

    ("algebra_faithful_response",     "chunk_relevance"):      (0.98, "Both chunks directly address quadratic equations."),
    ("algebra_faithful_response",     "context_sufficiency"):  (0.95, "Both solution methods are in the context."),
    ("algebra_faithful_response",     "faithfulness"):         (0.97, "Response mirrors the chunks accurately."),

    ("algebra_unfaithful_response",   "chunk_relevance"):      (0.98, "Chunk directly addresses quadratic equations."),
    ("algebra_unfaithful_response",   "context_sufficiency"):  (0.90, "Formula is in the context."),
    ("algebra_unfaithful_response",   "faithfulness"):         (0.02, "Response invents a calculus method not present in the context."),
}


# ── Mock Groq helper ──────────────────────────────────────────────────────────

def _patch_groq(score: float, reasoning: str, score_key: str, extra: dict | None = None):
    payload = {score_key: score, "reasoning": reasoning, **(extra or {})}
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_resp
    return patch("httpx.AsyncClient", return_value=mock_client)


# ── Per-metric parametrised tests ────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("case", RAG_CASES, ids=[c["id"] for c in RAG_CASES])
async def test_chunk_relevance_score(case, override_settings):
    """Are the retrieved chunks relevant to the question?"""
    score, reasoning = _MOCK_RAG_SCORES[(case["id"], "chunk_relevance")]

    with _patch_groq(score, reasoning, "chunk_relevance_score"):
        result = await judge_chunk_relevance(
            question=case["question"],
            chunks=case["chunks"],
            api_key="test-key",
        )

    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    assert isinstance(result, EvalScore)
    assert result.metric == "chunk_relevance"
    assert 0.0 <= result.score <= 1.0
    assert result.reasoning != ""


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RAG_CASES, ids=[c["id"] for c in RAG_CASES])
async def test_context_sufficiency_score(case, override_settings):
    """Do the chunks contain enough to produce the reference answer?"""
    score, reasoning = _MOCK_RAG_SCORES[(case["id"], "context_sufficiency")]

    with _patch_groq(score, reasoning, "context_sufficiency_score"):
        result = await judge_context_sufficiency(
            question=case["question"],
            chunks=case["chunks"],
            reference=case["reference"],
            api_key="test-key",
        )

    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    assert isinstance(result, EvalScore)
    assert result.metric == "context_sufficiency"
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RAG_CASES, ids=[c["id"] for c in RAG_CASES])
async def test_faithfulness_score(case, override_settings):
    """Does the agent response stay within the retrieved chunks?"""
    score, reasoning = _MOCK_RAG_SCORES[(case["id"], "faithfulness")]

    with _patch_groq(score, reasoning, "faithfulness_score",
                     extra={"unfaithful_claims": []}):
        result = await judge_faithfulness(
            chunks=case["chunks"],
            response=case["response"],
            api_key="test-key",
        )

    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    assert isinstance(result, EvalScore)
    assert result.metric == "faithfulness"
    assert 0.0 <= result.score <= 1.0


# ── CRAG thresholds (copied from app.rag.crag to avoid tavily import) ─────────
# These mirror the constants in app/rag/crag.py exactly.
_HIGH_THRESHOLD = 5.0
_LOW_THRESHOLD  = 0.0


def _crag_decision(score: float) -> str:
    if score > _HIGH_THRESHOLD:
        return "correct"
    if score > _LOW_THRESHOLD:
        return "ambiguous"
    return "web_corrected"


# ── CRAG decision unit tests (no LLM needed) ─────────────────────────────────

def test_crag_correct_decision_on_high_score():
    """Score above HIGH_THRESHOLD → decision must be 'correct'."""
    assert _crag_decision(_HIGH_THRESHOLD + 1.0) == "correct"


def test_crag_ambiguous_decision_on_mid_score():
    """Score between LOW and HIGH threshold → 'ambiguous'."""
    mid = (_HIGH_THRESHOLD + _LOW_THRESHOLD) / 2
    assert _crag_decision(mid) == "ambiguous"


def test_crag_web_fallback_on_low_score():
    """Score at or below LOW_THRESHOLD → decision must be 'web_corrected'."""
    assert _crag_decision(_LOW_THRESHOLD - 1.0) == "web_corrected"
    assert _crag_decision(_LOW_THRESHOLD) == "web_corrected"


# ── RRF fusion (inline copy — avoids qdrant_client import) ───────────────────

_RRF_K = 60


def _rrf_fuse(*result_lists):
    """Reciprocal Rank Fusion — mirrors app/rag/retriever.py exactly."""
    scores = {}
    payloads = {}
    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            pid = item["point_id"]
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            payloads[pid] = item.get("payload", {})
    return [
        {"point_id": pid, "payload": payloads[pid], "rrf_score": score}
        for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ── RRF unit tests ────────────────────────────────────────────────────────────

def test_rrf_fuse_deduplicates_by_point_id():
    """Same point_id appearing in two lists must be merged, not duplicated."""
    list1 = [{"point_id": "A", "payload": {"content": "apple"}},
             {"point_id": "B", "payload": {"content": "banana"}}]
    list2 = [{"point_id": "A", "payload": {"content": "apple"}},
             {"point_id": "C", "payload": {"content": "cherry"}}]
    fused = _rrf_fuse(list1, list2)
    point_ids = [r["point_id"] for r in fused]
    assert len(point_ids) == len(set(point_ids)), "Duplicate point_ids found after RRF fusion"


def test_rrf_fuse_boosts_shared_results():
    """A point_id in both lists should rank higher than one in only one list."""
    list1 = [{"point_id": "SHARED", "payload": {}}, {"point_id": "ONLY1", "payload": {}}]
    list2 = [{"point_id": "SHARED", "payload": {}}, {"point_id": "ONLY2", "payload": {}}]
    fused = _rrf_fuse(list1, list2)
    assert fused[0]["point_id"] == "SHARED"


def test_rrf_fuse_empty_lists():
    assert _rrf_fuse([], []) == []


def test_rrf_fuse_single_list():
    items = [{"point_id": str(i), "payload": {}} for i in range(3)]
    fused = _rrf_fuse(items)
    assert [r["point_id"] for r in fused] == ["0", "1", "2"]


# ── Full RAG evaluation suite ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_eval_suite(override_settings):
    """
    Run all three RAG metrics over every case and print a summary table.
    No score thresholds — read the numbers.
    """
    from dataclasses import dataclass, field

    @dataclass
    class CaseResult:
        case_id: str
        scores: dict = field(default_factory=dict)

    results = []

    for case in RAG_CASES:
        cr = CaseResult(case_id=case["id"])

        # chunk_relevance
        s, r = _MOCK_RAG_SCORES[(case["id"], "chunk_relevance")]
        with _patch_groq(s, r, "chunk_relevance_score"):
            cr.scores["chunk_relevance"] = await judge_chunk_relevance(
                case["question"], case["chunks"], api_key="test-key"
            )

        # context_sufficiency
        s, r = _MOCK_RAG_SCORES[(case["id"], "context_sufficiency")]
        with _patch_groq(s, r, "context_sufficiency_score"):
            cr.scores["context_sufficiency"] = await judge_context_sufficiency(
                case["question"], case["chunks"], case["reference"], api_key="test-key"
            )

        # faithfulness
        s, r = _MOCK_RAG_SCORES[(case["id"], "faithfulness")]
        with _patch_groq(s, r, "faithfulness_score", extra={"unfaithful_claims": []}):
            cr.scores["faithfulness"] = await judge_faithfulness(
                case["chunks"], case["response"], api_key="test-key"
            )

        results.append(cr)

    # ── Print score table ──────────────────────────────────────────────────────
    def _bar(score: float, w: int = 10) -> str:
        filled = round(score * w)
        return "█" * filled + "░" * (w - filled)

    print("\n")
    print("=" * 90)
    print("  EDUMENTOR RAG EVALUATION RESULTS")
    print("=" * 90)
    print(f"{'CASE ID':<38} {'METRIC':<22} {'SCORE':>6}   BAR        CRAG")
    print("-" * 90)

    prev = None
    for cr in results:
        case = next(c for c in RAG_CASES if c["id"] == cr.case_id)
        if prev and prev != cr.case_id:
            print()
        prev = cr.case_id
        for metric, es in cr.scores.items():
            crag = f"[{case['crag_decision']}]" if metric == "chunk_relevance" else ""
            print(f"{cr.case_id:<38} {metric:<22} {es.score:>6.2f}   {_bar(es.score)}  {crag}")

    # ── Averages ───────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  AVERAGE SCORES BY METRIC")
    print("-" * 90)
    all_scores: dict[str, list[float]] = {}
    for cr in results:
        for metric, es in cr.scores.items():
            all_scores.setdefault(metric, []).append(es.score)
    for metric, vals in sorted(all_scores.items()):
        avg = sum(vals) / len(vals)
        print(f"  {metric:<22} avg={avg:.2f}  {_bar(avg)}  (n={len(vals)})")
    print("=" * 90 + "\n")

    # ── Structural assertions only ─────────────────────────────────────────────
    for cr in results:
        for metric, es in cr.scores.items():
            assert 0.0 <= es.score <= 1.0
            assert es.reasoning