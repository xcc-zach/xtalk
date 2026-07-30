"""Public XTalk runtime integration for the desktop sidecar."""

from __future__ import annotations

from typing import Any

from xtalk import Xtalk

from .timer_tool import TimerTool


def build_xtalk_runtime(config: dict[str, Any]) -> Any:
    """Build XTalk with the sample-compatible desktop timer tool.

    Parameters
    ----------
    config : dict[str, Any]
        Fully composed generic XTalk configuration.

    Returns
    -------
    Any
        Public XTalk runtime wrapper exposing ``mount_routes``.
    """

    builder = Xtalk.configure(config)
    if config.get("llm_agent") is not None:
        builder.add_agent_tools([TimerTool])
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
