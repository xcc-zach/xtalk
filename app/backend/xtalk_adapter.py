"""Public XTalk runtime integration for the desktop sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xtalk import Xtalk
from xtalk.models.agents.tools import (
    build_async_web_search_tool,
    build_time_tool,
)

from .timer_tool import TimerTool
from .tool_registry import load_enabled_tools


def build_xtalk_runtime(
    config: dict[str, Any],
    *,
    tools_root: Path | None = None,
    web_search_enabled: bool = False,
) -> Any:
    """Build XTalk with desktop and enabled built-in Agent tools.

    Parameters
    ----------
    config : dict[str, Any]
        Fully composed generic XTalk configuration.
    tools_root : pathlib.Path | None, optional
        Application data directory containing installed developer tools.
    web_search_enabled : bool, optional
        Whether to register the asynchronous web-search tool.

    Returns
    -------
    Any
        Public XTalk runtime wrapper exposing ``mount_routes``.
    """

    builder = Xtalk.configure(config)
    if config.get("llm_agent") is not None:
        tools = (
            load_enabled_tools(tools_root)
            if tools_root is not None
            else []
        )
        tool_names = {getattr(tool, "name", None) for tool in tools}
        if TimerTool.name not in tool_names:
            tools.insert(0, TimerTool)
        if "get_time" not in tool_names:
            tools.append(build_time_tool())
        if web_search_enabled and "web_search" not in tool_names:
            tools.append(build_async_web_search_tool())
        builder.add_agent_tools(tools)
    return builder.build()


def mount_xtalk_routes(runtime: Any, app: Any) -> None:
    """Mount public XTalk HTTP and WebSocket routes on a FastAPI app.

    Parameters
    ----------
    runtime : Any
        Runtime returned by :func:`build_xtalk_runtime`.
    app : Any
        FastAPI application receiving XTalk's public routes.
    """

    runtime.mount_routes(app)
