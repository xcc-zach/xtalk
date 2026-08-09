"""Desktop registration wrapper carrying App tool session context."""

from __future__ import annotations

from xtalk import model
from xtalk.models.agents.default import DefaultAgent


@model
class DesktopDefaultAgent(DefaultAgent):
    """Preserve DefaultAgent scheduling while adding desktop metadata."""

    def bind_session(self, session_id: str) -> None:
        """Bind a live backend session to subsequent tool calls.

        Parameters
        ----------
        session_id : str
            XTalk service session owning this cloned Agent.
        """

        if not session_id:
            raise ValueError("desktop tool session ID must not be empty")
        if not isinstance(self.tool_engine.state, dict):
            raise TypeError("desktop tool state must be a dictionary")
        self.tool_engine.state["session_id"] = session_id

    def clone(self) -> "DesktopDefaultAgent":
        """Clone the desktop Agent while preserving App session binding support.

        Returns
        -------
        DesktopDefaultAgent
            Session-safe Agent clone using DefaultAgent's tool scheduler.
        """

        return DesktopDefaultAgent(
            model=self.model,
            backchannel_model=self.backchannel_model,
            backchannel_source_dir=self.backchannel_source_dir,
            system_prompt=self._additional_system_prompt,
            tools=self._base_tools,
            proactive=self.proactive,
        )
