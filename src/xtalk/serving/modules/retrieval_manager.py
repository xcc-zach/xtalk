import logging
from typing import Any

from ...models.agents.tools.retrievers import LOCAL_SEARCH_TOOL, WEB_SEARCH_TOOL
from ..event_bus import EventBus
from ..events import (
    RetrievalUpdated,
    ToolCallOccurred,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class RetrievalManager(Manager):
    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

    @Manager.event_handler(ToolCallOccurred, priority=80)
    async def _on_tool_called(self, event: ToolCallOccurred) -> None:
        """
        Handle tool-call events.

        We only care about tool_call_result messages from web search or local
        retrieval tools, which are rebroadcast as RetrievalUpdated events.
        """
        try:
            name = event.name or ""
            if name != "tool_call_result":
                return

            args = event.args or {}
            orig_tool = str(args.get("name", "") or "")
            content = str(args.get("content", "") or "")
            if orig_tool not in [WEB_SEARCH_TOOL, LOCAL_SEARCH_TOOL]:
                return
            if not content.strip():
                return

            await self.event_bus.publish(
                RetrievalUpdated(
                    session_id=self.session_id,
                    text=content,
                    is_final=True,
                )
            )
        except Exception as e:
            logger.error("Handle ToolCallOccurred failed: %s", e)

    async def shutdown(self) -> None:
        return
