"""
Web search tool for agent nodes.

Uses Tavily when TAVILY_API_KEY is set; falls back to DuckDuckGo
(no API key required) so the tool always works in development.

Bind to an agent with: llm.bind_tools([WEB_SEARCH_TOOL])
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
async def web_search(query: str) -> str:
    """
    Search the web for current information relevant to a student's question.
    Use when the question requires up-to-date facts not covered in curriculum docs.
    Returns a plain-text summary of the top results.
    """
    from app.core.config import settings

    tavily_key = getattr(settings, "TAVILY_API_KEY", "")

    if tavily_key:
        return await _tavily_search(query, tavily_key)
    return await _duckduckgo_search(query)


async def _tavily_search(query: str, api_key: str) -> str:
    try:
        import os

        from langchain_community.tools.tavily_search import TavilySearchResults

        os.environ["TAVILY_API_KEY"] = api_key
        tavily = TavilySearchResults(max_results=3)
        results = await tavily.ainvoke({"query": query})
        if not results:
            return "No results found."
        return "\n\n".join(f"[{r.get('url', '')}]\n{r.get('content', '')}" for r in results)
    except Exception as exc:
        return f"Tavily search failed: {exc}. Falling back to DuckDuckGo."


async def _duckduckgo_search(query: str) -> str:
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        ddg = DuckDuckGoSearchRun()
        return ddg.run(query)
    except Exception as exc:
        return f"Web search unavailable: {exc}"


WEB_SEARCH_TOOL = web_search
