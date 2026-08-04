"""Desktop registration wrapper for the default XTalk Agent."""

from __future__ import annotations

from xtalk import model
from xtalk.models.agents.default import DefaultAgent


@model
class DesktopDefaultAgent(DefaultAgent):
    """Register the default Agent under the desktop model catalog."""
