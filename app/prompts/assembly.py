"""
7-Layer Prompt Assembly — applied to ALL agents.

Layer order (strict):
1. SYSTEM PROMPT        — agent role, rules, student_level, theta
2. USER PREFERENCES     — explanation_style, weak_topics, session_goal ← PostgreSQL
3. SESSION SUMMARY      — RAG retrieved from user_memory Qdrant
4. RAG CONTENT          — top-3 reranked chunks from curriculum_docs
5. COT INSTRUCTION      — step-by-step reasoning scaffold
6. LAST 5 MESSAGES      — Upstash Redis → fallback PostgreSQL messages table
7. CURRENT USER QUERY   — transcribed text or typed message
"""

from __future__ import annotations

# Rough char budget for the system prompt (4096 ctx - ~800 for history+query)
_SYSTEM_CHAR_BUDGET = 2400
# Max chars per RAG chunk injected into the prompt
_MAX_CHUNK_CHARS = 400


def _trim(text: str, max_chars: int) -> str:
    """Hard-trim text to max_chars, appending ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ── Per-agent system prompts ──────────────────────────────────────────────────

SOCRATIC_SYSTEM = """You are EduMentor, a Socratic AI tutor. Your role is to GUIDE students to discover answers themselves — you NEVER give direct answers.

Rules (non-negotiable):
- Always respond with a guiding question or hint that leads the student toward the answer.
- Never state the answer directly, even if the student begs.
- Adapt complexity to student_level={student_level} (IRT theta={theta:.2f}).
- Use the student's preferred explanation style: {explanation_style}.
- Keep voice responses under 4 sentences.
- If the student is stuck, break the problem into smaller steps.
- Acknowledge emotions ("That's a tough one!") before guiding.
- Language: respond in {language}."""

QUIZ_SYSTEM = """You are EduMentor Quiz Generator. Generate well-structured multiple-choice questions.

Rules:
- Generate questions at difficulty level matching IRT theta={theta:.2f} (b-parameter target: {b_target:.2f}).
- Each question must have exactly 4 choices (A, B, C, D) with one correct answer.
- Include a concise explanation for the correct answer.
- Questions must test conceptual understanding, not just memorization.
- student_level={student_level}, language={language}."""

EXPLAINER_SYSTEM = """You are EduMentor Concept Explainer. You explain concepts clearly using analogies first.

Rules:
- Always start with an analogy the student can relate to.
- Use few-shot Chain-of-Thought: analogy → concept → example → summary.
- Limit voice output to under 4 sentences.
- Adapt to explanation_style={explanation_style} and student_level={student_level}.
- IRT theta={theta:.2f} — gauge depth accordingly.
- Language: {language}."""

INTENT_SYSTEM = """You are EduMentor Orchestrator. Classify the student's intent from their message.

Classify into exactly one of:
- "socratic"    : student is asking a question, needs guidance, or exploring a concept
- "quiz"        : student explicitly wants a quiz, test, or practice questions
- "explain"     : student wants an explanation, definition, or clarification
- "feedback"    : student is giving feedback on a response
- "meta"        : off-topic, small talk, or session management

Respond with ONLY the intent label, nothing else."""

MEMORY_CONSOLIDATION_SYSTEM = """You are EduMentor Memory Consolidator. Summarize a tutoring session.

Create a concise summary (3-5 sentences) covering:
1. Topics discussed
2. Concepts the student demonstrated understanding of
3. Concepts the student struggled with (weak areas)
4. Student's engagement style and pace
5. Recommended next topics

Output as plain text. No bullet points."""


# ── 7-layer assembly ──────────────────────────────────────────────────────────


def assemble_prompt(
    *,
    agent_type: str,  # "socratic"|"quiz"|"explain"|"intent"|"memory"
    query: str,  # Layer 7 — current user message
    student_level: str = "intermediate",
    theta: float = 0.0,
    explanation_style: str = "step_by_step",
    language: str = "en",
    weak_topics: list[str] | None = None,
    session_goal: str = "",
    session_summary: str = "",  # Layer 3 — from user_memory RAG
    user_doc_chunks: list[str] | None = None,  # Layer 3.5 — user's own uploaded docs
    rag_chunks: list[str] | None = None,  # Layer 4 — from curriculum_docs RAG
    history: list[dict] | None = None,  # Layer 6 — last 5 messages
    b_target: float = 0.0,  # For quiz agent IRT targeting
) -> list[dict]:
    """
    Assemble the full 8-layer prompt as a messages list for Ollama chat API.
    Returns: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    weak_topics = weak_topics or []
    rag_chunks = rag_chunks or []
    user_doc_chunks = user_doc_chunks or []
    history = history or []

    # ── Layer 1: System prompt ────────────────────────────────────────────────
    system_templates = {
        "socratic": SOCRATIC_SYSTEM,
        "quiz": QUIZ_SYSTEM,
        "explain": EXPLAINER_SYSTEM,
        "intent": INTENT_SYSTEM,
        "memory": MEMORY_CONSOLIDATION_SYSTEM,
    }
    system_template = system_templates.get(agent_type, SOCRATIC_SYSTEM)
    system_content = system_template.format(
        student_level=student_level,
        theta=theta,
        explanation_style=explanation_style,
        language=language,
        b_target=b_target,
    )

    # ── Layer 2: User preferences ─────────────────────────────────────────────
    if weak_topics or session_goal:
        prefs_lines = []
        if session_goal:
            prefs_lines.append(f"Session goal: {session_goal}")
        if weak_topics:
            prefs_lines.append(f"Student's weak areas: {', '.join(weak_topics)}")
        system_content += "\n\n[STUDENT PROFILE]\n" + "\n".join(prefs_lines)

    # ── Layer 3: Session summary from user_memory ─────────────────────────────
    if session_summary:
        system_content += f"\n\n[RECENT SESSION CONTEXT]\n{_trim(session_summary, 300)}"

    # ── Layer 3.5: User's own uploaded documents ──────────────────────────────
    if user_doc_chunks:
        docs_text = "\n---\n".join(_trim(c, _MAX_CHUNK_CHARS) for c in user_doc_chunks)
        system_content += f"\n\n[STUDENT'S OWN UPLOADED NOTES]\n{docs_text}"

    # ── Layer 4: RAG content from curriculum_docs ─────────────────────────────
    if rag_chunks:
        chunks_text = "\n---\n".join(_trim(c, _MAX_CHUNK_CHARS) for c in rag_chunks)
        system_content += f"\n\n[CURRICULUM REFERENCE MATERIAL]\n{chunks_text}"

    # ── Layer 5: CoT instruction ──────────────────────────────────────────────
    cot_instruction = _get_cot_instruction(agent_type)
    if cot_instruction:
        system_content += f"\n\n[REASONING APPROACH]\n{cot_instruction}"

    # Build messages list
    messages: list[dict] = [{"role": "system", "content": system_content}]

    # ── Layer 6: Last 3 messages (trimmed from 5 to fit small model ctx) ──────
    for msg in history[-3:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # ── Layer 7: Current query ────────────────────────────────────────────────
    messages.append({"role": "user", "content": query})

    return messages


def _get_cot_instruction(agent_type: str) -> str:
    cot_map = {
        "socratic": (
            "Think step by step before responding:\n"
            "1. What concept is the student trying to understand?\n"
            "2. What prerequisite knowledge do they need?\n"
            "3. What question would expose their current understanding gap?\n"
            "4. How can I guide them with a question rather than an answer?"
        ),
        "quiz": (
            "Generate the question by:\n"
            "1. Identify the core concept to test.\n"
            "2. Create a clear question stem.\n"
            "3. Make one clearly correct answer.\n"
            "4. Make three plausible distractors.\n"
            "5. Write a 1-2 sentence explanation."
        ),
        "explain": (
            "Structure your explanation as:\n"
            "1. Analogy (relate to something familiar)\n"
            "2. Core concept (precise definition)\n"
            "3. Concrete example\n"
            "4. One-sentence summary"
        ),
        "intent": "",
        "memory": "",
    }
    return cot_map.get(agent_type, "")
