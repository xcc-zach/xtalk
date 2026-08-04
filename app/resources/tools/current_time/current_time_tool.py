"""Required built-in current-time tool."""

from __future__ import annotations

from typing import Any

from xtalk.models.agents.tools import build_time_tool


def create_tools() -> list[Any]:
    """Create the App's always-enabled current-time tool.

    Returns
    -------
    list[Any]
        Public XTalk tool values accepted by the configured Agent.
    """

    return [build_time_tool()]
