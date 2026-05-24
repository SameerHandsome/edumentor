"""
LLM-as-Judge evaluator — Groq only (llama-3.3-70b-versatile).

No thresholds. No pass/fail. Just scores.
Every judge function returns an EvalScore with a float score (0.0–1.0)
and the LLM's reasoning. You read the numbers and decide what they mean.

Usage
-----
    from tests.evaluation.llm_judge import judge_hallucination

    result = await judge_hallucination(
        question="What is photosynthesis?",
        context="Photosynthesis converts light into glucose using chlorophyll.",
        response="Photosynthesis converts light energy into chemical energy in plants.",
    )
    print(result.score)     # e.g. 0.95
    print(result.reasoning) # "Response stays within the provided context..."
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

# ── Constants ──────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL   = "llama-3.3-70b-versatile"
JUDGE_TIMEOUT = 30.0


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class EvalScore:
    metric:       str    # e.g. "hallucination"
    score:        float  # 0.0 – 1.0  (1.0 = best quality)
    reasoning:    str    # LLM's chain-of-thought explanation
    raw_response: str    # full JSON string from Groq (for debugging)


# ── Prompt templates ──────────────────────────────────────────────────────────

_HALLUCINATION_PROMPT = """\
You are an expert fact-checker.  Evaluate whether the RESPONSE contains any
information NOT supported by the CONTEXT.

QUESTION: {question}

CONTEXT (ground-truth reference):
{context}

RESPONSE TO EVALUATE:
{response}

Only penalise for fabrication — not for paraphrasing or omission.

Respond in strict JSON:
{{
  "reasoning": "<one or two sentences explaining your judgment>",
  "hallucination_score": <float 0.0–1.0>
}}
1.0 = fully grounded, 0.0 = severe fabrication.
Respond ONLY with the JSON object."""

_CORRECTNESS_PROMPT = """\
You are an expert educational evaluator.  Rate how CORRECT and COMPLETE the
RESPONSE is compared to the REFERENCE ANSWER.

QUESTION: {question}

REFERENCE ANSWER:
{reference}

RESPONSE TO EVALUATE:
{response}

Respond in strict JSON:
{{
  "reasoning": "<one or two sentences>",
  "correctness_score": <float 0.0–1.0>
}}
1.0 = fully correct and complete, 0.0 = completely wrong.
Respond ONLY with the JSON object."""

_RELEVANCE_PROMPT = """\
You are evaluating whether an AI tutor's RESPONSE is relevant to the QUESTION.

QUESTION: {question}

RESPONSE:
{response}

Respond in strict JSON:
{{
  "reasoning": "<one or two sentences>",
  "relevance_score": <float 0.0–1.0>
}}
1.0 = completely on-topic, 0.0 = off-topic or evasive.
Respond ONLY with the JSON object."""

_SOCRATIC_QUALITY_PROMPT = """\
You are evaluating a Socratic tutoring response.  A good Socratic response:
  1. Does NOT give a direct answer.
  2. Asks a guiding question or provides a hint.
  3. Is encouraging and appropriate for the student's level.

STUDENT QUESTION: {question}

TUTOR RESPONSE:
{response}

Respond in strict JSON:
{{
  "reasoning": "<brief explanation>",
  "socratic_quality_score": <float 0.0–1.0>
}}
1.0 = perfect Socratic guidance, 0.0 = gives direct answer (rule violation).
Respond ONLY with the JSON object."""

_AUDIO_CLARITY_PROMPT = """\
You are evaluating whether a TTS (text-to-speech) script is appropriate for
being read aloud.  A good TTS script:
  1. Uses short, complete sentences (≤ 20 words each is ideal).
  2. Avoids markdown, LaTeX, special symbols (*, #, $, \\frac).
  3. Sounds natural when spoken.
  4. Does NOT exceed 4 sentences (voice-optimized).

TTS SCRIPT TO EVALUATE:
{response}

Respond in strict JSON:
{{
  "reasoning": "<brief explanation>",
  "audio_clarity_score": <float 0.0–1.0>,
  "tts_issues": ["<specific issues found, or empty list>"]
}}
1.0 = perfect for TTS, 0.0 = unreadable / full of symbols.
Respond ONLY with the JSON object."""


# ── Core Groq call ────────────────────────────────────────────────────────────

async def _call_groq_judge(prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=JUDGE_TIMEOUT) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_score(raw: str, score_key: str) -> tuple[float, str]:
    """Parse JSON from judge. Returns (score, reasoning). Never raises."""
    try:
        parsed   = json.loads(raw)
        score    = float(parsed.get(score_key, 0.0))
        reasoning = parsed.get("reasoning", "")
        return max(0.0, min(1.0, score)), reasoning
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return 0.0, f"Parse error: {exc} | raw={raw[:200]}"


# ── Public judge functions ─────────────────────────────────────────────────────

async def judge_hallucination(
    question: str,
    context: str,
    response: str,
    api_key: str | None = None,
) -> EvalScore:
    """Score how grounded the response is. 1.0 = no hallucination."""
    key    = api_key or os.environ.get("GROQ_API_KEY", "")
    prompt = _HALLUCINATION_PROMPT.format(
        question=question, context=context, response=response
    )
    raw            = await _call_groq_judge(prompt, key)
    score, reason  = _parse_score(raw, "hallucination_score")
    return EvalScore("hallucination", score, reason, raw)


async def judge_correctness(
    question: str,
    reference: str,
    response: str,
    api_key: str | None = None,
) -> EvalScore:
    """Score how correct the response is vs. a reference. 1.0 = fully correct."""
    key    = api_key or os.environ.get("GROQ_API_KEY", "")
    prompt = _CORRECTNESS_PROMPT.format(
        question=question, reference=reference, response=response
    )
    raw            = await _call_groq_judge(prompt, key)
    score, reason  = _parse_score(raw, "correctness_score")
    return EvalScore("correctness", score, reason, raw)


async def judge_relevance(
    question: str,
    response: str,
    api_key: str | None = None,
) -> EvalScore:
    """Score how on-topic the response is. 1.0 = perfectly relevant."""
    key    = api_key or os.environ.get("GROQ_API_KEY", "")
    prompt = _RELEVANCE_PROMPT.format(question=question, response=response)
    raw            = await _call_groq_judge(prompt, key)
    score, reason  = _parse_score(raw, "relevance_score")
    return EvalScore("relevance", score, reason, raw)


async def judge_socratic_quality(
    question: str,
    response: str,
    api_key: str | None = None,
) -> EvalScore:
    """Score Socratic pedagogy compliance. 1.0 = guides, 0.0 = gives answer directly."""
    key    = api_key or os.environ.get("GROQ_API_KEY", "")
    prompt = _SOCRATIC_QUALITY_PROMPT.format(question=question, response=response)
    raw            = await _call_groq_judge(prompt, key)
    score, reason  = _parse_score(raw, "socratic_quality_score")
    return EvalScore("socratic_quality", score, reason, raw)


async def judge_audio_clarity(
    response: str,
    api_key: str | None = None,
) -> EvalScore:
    """Score TTS suitability. 1.0 = clean for speech, 0.0 = full of symbols/markdown."""
    key    = api_key or os.environ.get("GROQ_API_KEY", "")
    prompt = _AUDIO_CLARITY_PROMPT.format(response=response)
    raw            = await _call_groq_judge(prompt, key)
    score, reason  = _parse_score(raw, "audio_clarity_score")
    return EvalScore("audio_clarity", score, reason, raw)
