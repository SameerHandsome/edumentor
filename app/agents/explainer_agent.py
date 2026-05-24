"""Concept Explainer Agent — analogy-first, voice-optimized (<4 sentences)."""

from __future__ import annotations

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langsmith import traceable

from app.agents.ollama_client import ollama_chat
from app.agents.state import EduMentorState
from app.agents.tools.mcp_tools import get_mcp_tools
from app.agents.tools.web_search_tool import WEB_SEARCH_TOOL
from app.core.config import settings
from app.prompts.assembly import assemble_prompt

logger = structlog.get_logger(__name__)

# Max chars of web search content to inject — prevents phi-3.5 from drowning
# in raw HTML/LaTeX and echoing it back instead of answering.
_MAX_TOOL_CONTEXT_CHARS = 600


def _get_tool_llm():
    """
    qwen3.5:0.8b handles tool routing only — it is trained for tool calling.
    Low temperature and small max_tokens because we only need tool-call JSON,
    not a full response.
    """
    tools = [WEB_SEARCH_TOOL] + get_mcp_tools()
    return ChatOllama(
        model=settings.OLLAMA_FALLBACK_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=256,
    ).bind_tools(tools)


async def _execute_tool_calls(tool_calls: list) -> list[ToolMessage]:
    """
    Execute each tool call returned by qwen and return ToolMessage results.
    Runs all tools even if one fails — errors are caught per-tool.
    """
    tool_map = {t.name: t for t in [WEB_SEARCH_TOOL] + get_mcp_tools()}
    results = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]
        try:
            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                content = f"Tool '{tool_name}' not found."
            else:
                content = await tool_fn.ainvoke(tool_args)
        except Exception as exc:
            content = f"Tool '{tool_name}' failed: {exc}"
            logger.warning("explainer_tool_exec_failed", tool=tool_name, error=str(exc))
        results.append(ToolMessage(content=str(content), tool_call_id=tool_id))
    return results


def _truncate_tool_context(raw_context: str) -> str:
    """
    BUG FIX 1: Tavily returns full webpage text — LaTeX math, HTML fragments,
    long article bodies — often 2000+ chars. phi-3.5 (small model) cannot
    handle this volume: it echoes the raw content verbatim instead of answering.

    Fix: hard-truncate to _MAX_TOOL_CONTEXT_CHARS so phi-3.5 only sees
    a short clean snippet it can actually reason over.
    """
    if len(raw_context) <= _MAX_TOOL_CONTEXT_CHARS:
        return raw_context
    truncated = raw_context[:_MAX_TOOL_CONTEXT_CHARS]
    logger.warning(
        "tool_context_truncated",
        original_chars=len(raw_context),
        truncated_to=_MAX_TOOL_CONTEXT_CHARS,
    )
    return truncated + "... [truncated]"


@traceable(name="concept_explainer", project_name="edumentor")
async def explainer_agent(state: EduMentorState) -> EduMentorState:
    """
    Two-step flow:
      Step 1 — qwen3.5:0.8b decides which tools to call (web_search).
               Executes the tools, TRUNCATES results to safe length.
      Step 2 — edumentor:latest produces the final explanation with the
               truncated context injected as a user turn.

    Falls back to plain ollama_chat (no tools) if Step 1 fails.
    """
    messages = assemble_prompt(
        agent_type="explain",
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

    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
        else:
            lc_messages.append(HumanMessage(content=m["content"]))

    tool_context = ""

    # ── Step 1: qwen3.5:0.8b — tool routing and execution ──────────────────────
    try:
        tool_llm = _get_tool_llm()
        tool_result = await tool_llm.ainvoke(lc_messages)

        if tool_result.tool_calls:
            logger.info(
                "explainer_tools_called",
                tools=[tc["name"] for tc in tool_result.tool_calls],
            )
            tool_messages = await _execute_tool_calls(tool_result.tool_calls)
            raw_tool_context = "\n\n".join(
                f"[{tm.tool_call_id}]: {tm.content}" for tm in tool_messages
            )
            # BUG FIX 1: truncate before injecting — full Tavily HTML causes echo
            tool_context = _truncate_tool_context(raw_tool_context)
        else:
            logger.debug("explainer_no_tools_called")

    except Exception as exc:
        logger.warning("explainer_tool_step_failed", error=str(exc), fallback="phi35_direct")

    # ── Step 2: edumentor:latest — final explanation ─────────────────────────
    # Inject tool results as a USER turn (not system) — small models like phi-3.5
    # tend to echo system messages back verbatim, which corrupts the response.
    if tool_context:
        enriched_messages = messages + [
            {
                "role": "user",
                "content": (
                    f"Here is a brief web reference to help your explanation:\n"
                    f"{tool_context}\n\n"
                    f"Now explain to the student (use analogy first, your own words): {state.user_query}"
                ),
            }
        ]
    else:
        enriched_messages = messages

    response = await ollama_chat(
        enriched_messages,
        temperature=0.5,
        max_tokens=300,
        agent_type="explain",
    )

    # ── Sanitize response ──────────────────────────────────────────────────────
    # BUG FIX 2: response_clean MUST be assigned from `response` FIRST before
    # any checks. Previous code had the length guard checking response_clean
    # before it was assigned (NameError crash hidden by outer exception handler).
    response_clean = response.strip()

    # Guard: if response is suspiciously long AND tool_context was injected,
    # the model echoed the web content — retry without tool context.
    _max_sane_response = 700
    if len(response_clean) > _max_sane_response and tool_context:
        logger.warning(
            "explainer_response_too_long_retry_without_tools",
            chars=len(response_clean),
        )
        response_clean = (
            await ollama_chat(
                messages,
                temperature=0.5,
                max_tokens=300,
                agent_type="explain",
            )
        ).strip()

    # Detect state-object leak: model echoed the full JSON state back
    _is_state_leak = response_clean.startswith("{") and any(
        k in response_clean for k in ("user_query", "agent_type", "rag_chunks", "session_id")
    )
    if _is_state_leak:
        logger.error("explainer_state_leak_detected", preview=response_clean[:80])
        response_clean = (
            await ollama_chat(
                messages,
                temperature=0.5,
                max_tokens=300,
                agent_type="explain",
            )
        ).strip()

    # BUG FIX 3: expanded echo prefixes — original only had 4 entries and missed
    # all URL/HTML/LaTeX patterns from Tavily results being echoed back verbatim.
    # The garbage you saw ("In Calculus, the Quotient Rule is...") was Tavily's
    # article text echoed because phi-3.5 got overwhelmed by the full raw content.
    _echo_prefixes = (
        "Additional context retrieved from tools:",
        "[Reference material for your explanation",
        "Here is a brief web reference",  # our own injection prefix
        "I'm sorry, I cannot provide",
        "I cannot provide additional",
        "http://",  # raw URL echo
        "https://",  # raw URL echo
        "WEB SEARCH [",  # CRAG web result header echo
        "[Note: retrieved from web search",
        "[Note: context confidence",
        "In Calculus,",  # Tavily article intro echo
        "In calculus,",
        "\\begin{array}",  # LaTeX echo from math articles
        "\\(",  # LaTeX echo
    )
    for prefix in _echo_prefixes:
        if response_clean.startswith(prefix):
            logger.warning(
                "explainer_echo_detected",
                prefix=prefix[:40],
                fallback="no_tool_context",
            )
            response_clean = (
                await ollama_chat(
                    messages,
                    temperature=0.5,
                    max_tokens=300,
                    agent_type="explain",
                )
            ).strip()
            break

    logger.info("explainer_response", length=len(response_clean))
    return state.model_copy(update={"agent_response": response_clean, "agent_type": "explain"})
