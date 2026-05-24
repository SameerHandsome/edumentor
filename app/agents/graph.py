"""
LangGraph stateful multi-agent orchestrator.

Graph flow:
  START → retrieval → intent_classify → [socratic | quiz | explain] → END

Routing:
  intent_classify sets state.intent; conditional_edge routes to the correct agent.
  Memory consolidation is triggered explicitly from Celery tasks via
  app.services.memory_service — it is NOT a graph node.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from app.agents.explainer_agent import explainer_agent
from app.agents.intent_classifier import classify_intent
from app.agents.quiz_agent import quiz_agent
from app.agents.socratic_agent import socratic_agent
from app.agents.state import EduMentorState

logger = structlog.get_logger(__name__)

_qdrant_client: Any = None
_bm25_encoder: Any = None


def set_rag_dependencies(qdrant_client: Any, bm25_encoder: Any) -> None:
    global _qdrant_client, _bm25_encoder
    _qdrant_client = qdrant_client
    _bm25_encoder = bm25_encoder


async def _retrieval_wrapper(state: EduMentorState) -> EduMentorState:
    from app.agents.retrieval_node import retrieval_node

    return await retrieval_node(state, _qdrant_client, _bm25_encoder)


def _route_by_intent(state: EduMentorState) -> str:
    intent_map = {
        "socratic": "socratic_agent",
        "quiz": "quiz_agent",
        "explain": "explainer_agent",
        "feedback": "socratic_agent",
        "meta": "socratic_agent",
    }
    return intent_map.get(state.intent, "socratic_agent")


def build_graph() -> StateGraph:
    builder = StateGraph(EduMentorState)

    builder.add_node("retrieval", _retrieval_wrapper)
    builder.add_node("intent_classify", classify_intent)
    builder.add_node("socratic_agent", socratic_agent)
    builder.add_node("explainer_agent", explainer_agent)
    builder.add_node("quiz_agent", quiz_agent)

    builder.set_entry_point("retrieval")
    builder.add_edge("retrieval", "intent_classify")
    builder.add_conditional_edges("intent_classify", _route_by_intent)
    builder.add_edge("socratic_agent", END)
    builder.add_edge("explainer_agent", END)
    builder.add_edge("quiz_agent", END)

    return builder.compile()


_graph = None


def get_graph() -> StateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
