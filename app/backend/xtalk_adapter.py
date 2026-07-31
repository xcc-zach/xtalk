"""Public XTalk runtime integration for the desktop sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xtalk import Xtalk

from .timer_tool import TimerTool
from .tool_registry import load_enabled_tools
from .tool_ui import ToolUIBroker


def build_xtalk_runtime(
    config: dict[str, Any],
    *,
    tools_root: Path | None = None,
    anonymous_user_id: str | None = None,
    tool_ui_broker: ToolUIBroker | None = None,
) -> Any:
    """Build XTalk with the sample-compatible desktop timer tool.

    Parameters
    ----------
    config : dict[str, Any]
        Fully composed generic XTalk configuration.
    tools_root : pathlib.Path | None, optional
        Application data directory containing installed developer tools.
    anonymous_user_id : str | None, optional
        Runtime-only stable identity for the built-in anonymous login.
    tool_ui_broker : ToolUIBroker | None, optional
        App-only observer for developer tools that declare a UI entrypoint.

    Returns
    -------
    Any
        Public XTalk runtime wrapper exposing ``mount_routes``.
    """

    builder = Xtalk.configure(config)
    if config.get("llm_agent") is not None:
        if tools_root is None:
            tools = []
        elif tool_ui_broker is None:
            tools = load_enabled_tools(tools_root)
        else:
            tools = load_enabled_tools(
                tools_root,
                tool_ui_broker=tool_ui_broker,
            )
        if not any(getattr(tool, "name", None) == TimerTool.name for tool in tools):
            tools.insert(0, TimerTool)
        builder.add_agent_tools(tools)
    runtime = builder.build()
    if anonymous_user_id is not None:
        # Keep the desktop-owned login identity outside user model/service config.
        runtime._anonymous_user_id = anonymous_user_id
    return runtime


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
