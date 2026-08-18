# -*- coding: utf-8 -*-
"""Manager that forwards direct-audio tool calls as TTS chunks."""

import logging
from typing import Any

from ..event_bus import EventBus
from ..events import ToolCallOccurred, TTSChunkReady
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class DirectAudioManager(Manager):
    """Forward ``direct_audio`` tool calls to the outbound audio stream."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the direct-audio manager.

        Parameters
        ----------
        event_bus : EventBus
            Session-scoped event bus.
        session_id : str
            Active session identifier.
        config : dict[str, Any] | None, optional
            Session configuration shared with managers.
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

    @Manager.event_handler(ToolCallOccurred, priority=90)
    async def _handle_tool_call_occurred(self, event: ToolCallOccurred) -> None:
        """Publish direct audio payloads as generated TTS chunks.

        Parameters
        ----------
        event : ToolCallOccurred
            Tool-call notification emitted by the agent stream consumer.
        """
        if event.name != "direct_audio":
            return

        args = event.args or {}
        audio = args.get("audio", b"")
        if isinstance(audio, bytearray):
            audio = bytes(audio)
        elif isinstance(audio, memoryview):
            audio = audio.tobytes()

        if not isinstance(audio, bytes) or not audio:
            logger.warning(
                "direct_audio missing valid audio bytes - session: %s",
                self.session_id,
            )
            return

        try:
            sample_rate = int(args.get("sample_rate", 48000) or 48000)
        except (TypeError, ValueError):
            logger.warning(
                "direct_audio received invalid sample_rate %r - session: %s",
                args.get("sample_rate"),
                self.session_id,
            )
            return

        if sample_rate <= 0:
            logger.warning(
                "direct_audio received non-positive sample_rate %r - session: %s",
                sample_rate,
                self.session_id,
            )
            return

        await self.event_bus.publish(
            TTSChunkReady(
                session_id=self.session_id,
                audio_chunk=audio,
                sample_rate=sample_rate,
            )
        )

    async def shutdown(self) -> None:
        """Shut down the manager."""
        return
