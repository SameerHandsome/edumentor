"""Socratic Tutor Agent — NEVER gives direct answers."""

from __future__ import annotations

import structlog
from langsmith import traceable

from app.agents.ollama_client import ollama_chat
from app.agents.state import EduMentorState
from app.prompts.assembly import assemble_prompt

logger = structlog.get_logger(__name__)


@traceable(name="socratic_tutor", project_name="edumentor")
async def socratic_agent(state: EduMentorState) -> EduMentorState:
    """
    Guides student with Socratic questions using the full 8-layer prompt.
    Uses edumentor-phi3.5 directly via ollama_chat (cascade router).
    Student context (theta, weak_topics, level) is injected via assemble_prompt
    so no DB tool lookups are needed.
    """
    messages = assemble_prompt(
        agent_type="socratic",
        query=state.user_query,
        student_level=state.student_level,
        theta=state.theta,
        explanation_style=state.explanation_style,
        language=state.language,
        weak_topics=state.weak_topics,
        session_goal=state.session_goal,
        session_summary=state.session_summary,
        user_doc_chunks=state.user_doc_chunks,
        rag_chunks=state.rag_chunks,
        history=state.history,
    )

    response = await ollama_chat(
        messages,
        temperature=0.7,
        max_tokens=512,
        agent_type="socratic",
    )

    logger.info("socratic_response", length=len(response))
    return state.model_copy(update={"agent_response": response, "agent_type": "socratic"})
