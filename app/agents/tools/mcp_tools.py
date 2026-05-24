"""
ArXiv research tool — uses langchain_community's ArxivQueryRun/ArxivAPIWrapper
which calls the public arXiv API directly (https://export.arxiv.org/api/query).

No local MCP server process is needed.  The tool is a standard LangChain
BaseTool so it works identically with llm.bind_tools([...]) and LangGraph nodes.

Public interface (unchanged from old mcp_tools.py so nothing else needs editing):
    await init_mcp_tools()   — call once at startup in main.py lifespan
    get_mcp_tools()          — returns list[BaseTool] for bind_tools / graph nodes
    await shutdown_mcp_tools() — no-op but kept for API compatibility
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ── Module-level cache ───────────────────────────────────────────────────────
_mcp_tools: list[Any] = []


def _build_arxiv_tool() -> Any:
    """
    Build and return a LangChain ArxivQueryRun tool.

    ArxivAPIWrapper parameters:
      top_k_results=5             — return top 5 papers per query
      ARXIV_MAX_QUERY_LENGTH=300  — truncate very long queries
      load_max_docs=5             — max docs loaded into memory
      load_all_available_meta=False  — title/authors/summary only (no full PDF)
      doc_content_chars_max=2000  — cap each paper summary at 2 000 chars

    Requires: pip install arxiv langchain-community
    """
    try:
        from langchain_community.tools.arxiv.tool import ArxivQueryRun
        from langchain_community.utilities.arxiv import ArxivAPIWrapper
    except ImportError as exc:
        raise ImportError(
            "langchain-community and arxiv packages are required. "
            "Install them with: pip install langchain-community arxiv"
        ) from exc

    wrapper = ArxivAPIWrapper(
        top_k_results=5,
        ARXIV_MAX_QUERY_LENGTH=300,
        load_max_docs=5,
        load_all_available_meta=False,
        doc_content_chars_max=2000,
    )

    tool = ArxivQueryRun(
        api_wrapper=wrapper,
        description=(
            "Search arXiv.org for academic papers. "
            "Use this when a student asks about recent research, scientific papers, "
            "or wants to learn from primary sources in Physics, Mathematics, "
            "Computer Science, Biology, Statistics, or any STEM field. "
            "Input should be a clear search query or paper topic."
        ),
    )
    return tool


async def init_mcp_tools() -> None:
    """
    Build the ArXiv tool and store it in the module-level cache.
    Called once at app startup (in main.py lifespan).
    Never raises — errors are logged and gracefully skipped.
    """
    global _mcp_tools
    try:
        arxiv_tool = _build_arxiv_tool()
        _mcp_tools = [arxiv_tool]
        logger.info("arxiv_tool_initialized", tool_name=arxiv_tool.name)
    except Exception as exc:
        logger.error("arxiv_tool_init_failed", error=str(exc))
        _mcp_tools = []


async def shutdown_mcp_tools() -> None:
    """
    No persistent connections to close for the ArXiv HTTP tool.
    Kept for API compatibility with old mcp_tools.py.
    """
    _mcp_tools.clear()
    logger.info("arxiv_tool_shutdown")


def get_mcp_tools() -> list[Any]:
    """Return cached ArXiv tool(s). Empty list if init_mcp_tools() not yet called."""
    return _mcp_tools
