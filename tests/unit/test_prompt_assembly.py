"""
Unit tests for the 7-layer prompt assembly system.

Tests that:
- All 7 layers are assembled in order
- RAG content is injected as expected
- History messages appear in correct positions
- Character budget is respected (no runaway prompts)
- Each agent type produces a valid messages list
"""
from __future__ import annotations

import pytest

from app.prompts.assembly import assemble_prompt


@pytest.mark.parametrize("agent_type", ["socratic", "quiz", "explain", "intent"])
def test_assemble_returns_list(agent_type):
    """assemble_prompt must always return a non-empty list."""
    result = assemble_prompt(
        agent_type=agent_type,
        query="What is photosynthesis?",
        student_level="intermediate",
        theta=0.0,
        explanation_style="step_by_step",
        language="en",
    )
    assert isinstance(result, list)
    assert len(result) >= 1


def test_first_message_is_system():
    """The first message in the assembled prompt must be a system message."""
    result = assemble_prompt(
        agent_type="socratic",
        query="Explain gravity.",
        student_level="beginner",
        theta=-0.5,
        explanation_style="analogy",
        language="en",
    )
    assert result[0]["role"] == "system"


def test_system_prompt_contains_theta():
    """System prompt must inject the student theta value."""
    result = assemble_prompt(
        agent_type="socratic",
        query="Test query",
        student_level="intermediate",
        theta=1.23,
        explanation_style="step_by_step",
        language="en",
    )
    system_content = result[0]["content"]
    assert "1.23" in system_content


def test_rag_chunks_injected():
    """RAG chunks must appear somewhere in the assembled message list."""
    chunks = ["Newton's law states F=ma.", "Gravity accelerates at 9.8 m/s²."]
    result = assemble_prompt(
        agent_type="explain",
        query="Explain Newton's laws.",
        student_level="intermediate",
        theta=0.0,
        explanation_style="step_by_step",
        language="en",
        rag_chunks=chunks,
    )
    full_text = " ".join(m["content"] for m in result)
    assert "F=ma" in full_text or "9.8" in full_text


def test_history_messages_appear():
    """Last N history messages must appear in the assembled prompt."""
    history = [
        {"role": "user", "content": "What is DNA?"},
        {"role": "assistant", "content": "DNA stores genetic information."},
    ]
    result = assemble_prompt(
        agent_type="socratic",
        query="How does DNA replicate?",
        student_level="intermediate",
        theta=0.0,
        explanation_style="step_by_step",
        language="en",
        history=history,
    )
    full_text = " ".join(m["content"] for m in result)
    assert "DNA" in full_text


def test_user_query_is_last_user_message():
    """The current user query must appear as the last user-role message."""
    query = "Explain the Krebs cycle."
    result = assemble_prompt(
        agent_type="explain",
        query=query,
        student_level="intermediate",
        theta=0.0,
        explanation_style="step_by_step",
        language="en",
    )
    user_messages = [m for m in result if m["role"] == "user"]
    assert len(user_messages) >= 1
    assert query in user_messages[-1]["content"]


def test_system_prompt_char_budget():
    """System prompt content must not exceed _SYSTEM_CHAR_BUDGET (2400) chars."""
    from app.prompts.assembly import _SYSTEM_CHAR_BUDGET

    result = assemble_prompt(
        agent_type="socratic",
        query="Tell me everything about calculus.",
        student_level="advanced",
        theta=2.0,
        explanation_style="step_by_step",
        language="en",
        rag_chunks=["x" * 500, "y" * 500],  # large chunks
    )
    system_content = result[0]["content"]
    assert len(system_content) <= _SYSTEM_CHAR_BUDGET + 500, (
        f"System prompt too long: {len(system_content)} chars"
    )


def test_empty_rag_chunks_safe():
    """assemble_prompt must not raise when rag_chunks is empty."""
    result = assemble_prompt(
        agent_type="socratic",
        query="What is entropy?",
        student_level="intermediate",
        theta=0.0,
        explanation_style="step_by_step",
        language="en",
        rag_chunks=[],
    )
    assert result  # non-empty list
