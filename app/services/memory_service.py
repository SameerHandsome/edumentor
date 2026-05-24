"""Memory Consolidation Service — summarizes sessions for user_memory Qdrant.

Called exclusively by Celery session_summarize task at session end.
Not part of the LangGraph graph — background infrastructure, not a conversational agent.
"""

from __future__ import annotations

import structlog
from langsmith import traceable

from app.agents.ollama_client import ollama_chat
from app.agents.state import EduMentorState
from app.prompts.assembly import assemble_prompt

logger = structlog.get_logger(__name__)


@traceable(name="memory_consolidation", project_name="edumentor")
async def memory_consolidation(state: EduMentorState) -> EduMentorState:
    """
    Summarizes the full session history into a compact memory entry.
    Uses the 7-layer prompt (history layer is the full session transcript).
    """
    if not state.history:
        return state.model_copy(update={"agent_response": "", "agent_type": "memory"})

    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in state.history)
    messages = assemble_prompt(
        agent_type="memory",
        query=f"Please summarize this tutoring session:\n\n{transcript}",
        student_level=state.student_level,
        theta=state.theta,
        explanation_style=state.explanation_style,
        language=state.language,
        weak_topics=state.weak_topics,
        session_goal=state.session_goal,
        history=[],
    )
    summary = await ollama_chat(messages, temperature=0.3, max_tokens=300, agent_type="memory")
    logger.info("memory_consolidated", session_id=state.session_id, length=len(summary))
    return state.model_copy(update={"agent_response": summary, "agent_type": "memory"})
