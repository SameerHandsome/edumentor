"""
Unit tests for the Socratic Tutor agent.

Tests that:
- Response is stored in state.agent_response
- agent_type is set to 'socratic'
- assemble_prompt is called with agent_type='socratic'
- Response is non-empty
- ollama_chat is called once with expected params
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.socratic_agent import socratic_agent
from app.agents.state import EduMentorState


@pytest.fixture
def mock_ollama():
    with patch("app.agents.socratic_agent.ollama_chat") as m:
        m.return_value = "What do you think happens when you add two sides of a triangle?"
        yield m


@pytest.fixture
def mock_assemble():
    with patch("app.agents.socratic_agent.assemble_prompt") as m:
        m.return_value = [{"role": "system", "content": "You are Socratic tutor"}]
        yield m


@pytest.mark.asyncio
async def test_socratic_sets_agent_response(mock_ollama, mock_assemble, base_state):
    result = await socratic_agent(base_state)
    assert result.agent_response != ""
    assert result.agent_response == mock_ollama.return_value


@pytest.mark.asyncio
async def test_socratic_sets_agent_type(mock_ollama, mock_assemble, base_state):
    result = await socratic_agent(base_state)
    assert result.agent_type == "socratic"


@pytest.mark.asyncio
async def test_socratic_assemble_called_with_correct_agent_type(mock_ollama, mock_assemble, base_state):
    await socratic_agent(base_state)
    mock_assemble.assert_called_once()
    call_kwargs = mock_assemble.call_args[1]
    assert call_kwargs.get("agent_type") == "socratic"


@pytest.mark.asyncio
async def test_socratic_passes_rag_chunks_to_prompt(mock_ollama, mock_assemble, base_state):
    """RAG chunks must be forwarded to the prompt assembler."""
    await socratic_agent(base_state)
    call_kwargs = mock_assemble.call_args[1]
    assert call_kwargs.get("rag_chunks") == base_state.rag_chunks


@pytest.mark.asyncio
async def test_socratic_ollama_called_once(mock_ollama, mock_assemble, base_state):
    await socratic_agent(base_state)
    mock_ollama.assert_called_once()


@pytest.mark.asyncio
async def test_socratic_does_not_mutate_original_state(mock_ollama, mock_assemble, base_state):
    """model_copy must be used — original state should remain unchanged."""
    original_response = base_state.agent_response
    await socratic_agent(base_state)
    assert base_state.agent_response == original_response
