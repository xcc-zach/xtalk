"""Optional built-in asynchronous Web Search tool."""

from __future__ import annotations

from xtalk.models.agents.tools import AsyncTool, build_async_web_search_tool


def create_tools() -> list[type[AsyncTool]]:
    """Create the App's Serper-backed asynchronous Web Search tool.

    Returns
    -------
    list[type[AsyncTool]]
        Public XTalk asynchronous tool classes accepted by the Agent.
    """

    return [build_async_web_search_tool()]
