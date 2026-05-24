"""Intent classifier node — routes to the correct agent."""

from __future__ import annotations

import structlog
from langsmith import traceable

from app.agents.ollama_client import ollama_chat
from app.agents.state import EduMentorState
from app.prompts.assembly import assemble_prompt

logger = structlog.get_logger(__name__)

VALID_INTENTS = {"socratic", "quiz", "explain", "feedback", "meta"}


@traceable(name="intent_classifier", project_name="edumentor")
async def classify_intent(state: EduMentorState) -> EduMentorState:
    """
    Layer-7-prompt-assembled intent classification.
    Sets state.intent to one of the VALID_INTENTS.
    """
    messages = assemble_prompt(
        agent_type="intent",
        query=state.user_query,
        student_level=state.student_level,
        theta=state.theta,
        explanation_style=state.explanation_style,
        language=state.language,
        history=state.history,
    )
    raw = await ollama_chat(messages, temperature=0.1, max_tokens=10, agent_type="intent")
    intent = raw.strip().lower()
    if intent not in VALID_INTENTS:
        intent = "socratic"
    logger.info("intent_classified", intent=intent, query=state.user_query[:60])
    return state.model_copy(update={"intent": intent})
