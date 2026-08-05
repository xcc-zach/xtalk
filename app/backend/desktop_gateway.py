# -*- coding: utf-8 -*-
"""Desktop output gateway that projects complete, monotonic conversation text.

The generic XTalk pipeline derives the frontend-facing ``update_resp`` text
from TTS playback progress. When an assistant turn is interrupted server-side
(for example by turn-taking cleanup after playback finishes, a new text input,
or a VAD echo), the played-text stream restarts with a non-prefix value. The
client SDK interprets such a restart as a brand-new message, so the chat keeps
truncated fragment bubbles even though TTS already spoke the full response.

This module installs a desktop-only output gateway that watches the
authoritative LLM accumulated text and guarantees the stream the SDK sees is
monotonic: every turn is closed with its complete text before the next turn
starts, and playback-progress updates that would shrink or jump are mapped back
onto the LLM text instead of being forwarded verbatim.
"""

from __future__ import annotations

import asyncio
from typing import Any

from xtalk.serving.events import (
    ErrorOccurred,
    LLMAgentResponseFinish,
    LLMAgentResponseUpdate,
    ResponseFinish,
    ResponseUpdate,
    ToolCallOccurred,
)
from xtalk.serving.interfaces import EventListenerMixin
from xtalk.serving.modules.output_gateway import OutputGateway


class DesktopTextProjectionGateway(OutputGateway):
    """Output gateway keeping the frontend text stream complete and monotonic.

    Parameters
    ----------
    event_bus : EventBus
        Session event bus subscribed by the gateway.
    session_id : str
        Session identifier sent back to the frontend.
    websocket : WebSocket
        Live WebSocket connection used for outbound messages.
    config : dict[str, Any] | None, optional
        Service configuration relevant to output behavior.
    """

    def __init__(
        self,
        event_bus: Any,
        session_id: str,
        websocket: Any,
        config: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        """Initialize the projection state alongside the standard gateway."""

        super().__init__(
            event_bus,
            session_id,
            websocket,
            config=config,
        )
        self._projection_lock = asyncio.Lock()
        self._desktop_message_open = False
        self._desktop_llm_text = ""
        self._desktop_sent_text = ""
        self._desktop_final_pending = ""

    # ── LLM authoritative text tracking ────────────────────────────

    @EventListenerMixin.event_handler(LLMAgentResponseUpdate, priority=60)
    async def _desktop_track_llm_update(
        self,
        event: LLMAgentResponseUpdate,
    ) -> None:
        """Track the accumulated LLM text and close restarted turns cleanly.

        Parameters
        ----------
        event : LLMAgentResponseUpdate
            Event carrying the newest accumulated agent response text.
        """

        text = event.text or ""
        if not text:
            return
        async with self._projection_lock:
            if (
                self._desktop_message_open
                and not text.startswith(self._desktop_llm_text)
            ):
                # The agent restarted mid-turn: finalize the previous response
                # with its complete generated text before opening a new one.
                await self._desktop_close_message(self._desktop_llm_text)
            self._desktop_message_open = True
            self._desktop_llm_text = text

    @EventListenerMixin.event_handler(LLMAgentResponseFinish, priority=60)
    async def _desktop_track_llm_finish(
        self,
        event: LLMAgentResponseFinish,
    ) -> None:
        """Remember the complete LLM text for the upcoming playback finish.

        Parameters
        ----------
        event : LLMAgentResponseFinish
            Event carrying the final agent response text.
        """

        text = event.text or ""
        async with self._projection_lock:
            if text:
                self._desktop_llm_text = text
            self._desktop_final_pending = text

    @EventListenerMixin.event_handler(ErrorOccurred, priority=60)
    async def _desktop_handle_error(self, event: ErrorOccurred) -> None:
        """Close any open message with its complete text before an error.

        Parameters
        ----------
        event : ErrorOccurred
            Event signalling a session-level error.
        """

        del event
        async with self._projection_lock:
            await self._desktop_close_message(self._desktop_llm_text)

    @EventListenerMixin.event_handler(ToolCallOccurred, priority=60)
    async def _desktop_record_tool_call_offset(
        self,
        event: ToolCallOccurred,
    ) -> None:
        """Record where each UI tool call appears in the assistant text.

        The tool UI broker consumes these offsets in FIFO order and attaches
        them to its emit payloads, letting the chat render tool rows inside
        the assistant message at the exact generation point.

        Parameters
        ----------
        event : ToolCallOccurred
            Event describing one agent tool invocation.
        """

        bridge = (self.config or {}).get("_desktop_tool_call_bridge")
        if bridge is None:
            return
        async with self._projection_lock:
            bridge.record_tool_call(
                session_id=self.session_id,
                name=event.name,
                offset=len(self._desktop_llm_text),
            )

    # ── Frontend-facing stream projection ──────────────────────────

    async def _send_update_resp_signal(self, event: ResponseUpdate) -> None:
        """Forward played-text updates only when they extend the stream.

        Parameters
        ----------
        event : ResponseUpdate
            Event whose text is the latest playback-confirmed prefix.
        """

        async with self._projection_lock:
            await self._desktop_forward_played_update(event.text or "")

    async def _send_finish_resp_signal(self, event: ResponseFinish) -> None:
        """Finalize the open message with the complete generated text.

        Parameters
        ----------
        event : ResponseFinish
            Event signalling the end of one playback-confirmed response.
        """

        async with self._projection_lock:
            if not self._desktop_message_open:
                return
            final = (
                self._desktop_final_pending
                or self._desktop_llm_text
                or (event.text or "")
            )
            await self._desktop_close_message(final)

    async def _desktop_forward_played_update(self, played: str) -> None:
        """Send one playback update without ever emitting a fragment.

        Parameters
        ----------
        played : str
            Playback-confirmed text prefix produced by the TTS playback manager.
        """

        if not self._desktop_message_open or not played:
            return
        sent = self._desktop_sent_text
        if played.startswith(sent):
            if played != sent:
                self._desktop_sent_text = played
                await self._forward("update_resp", {"text": played})
            return
        if len(played) > len(sent):
            # Playback tracking restarted mid-turn; catch the display up to the
            # authoritative LLM text instead of emitting a non-prefix fragment.
            await self._desktop_catch_up_to_llm()

    async def _desktop_catch_up_to_llm(self) -> None:
        """Push the full LLM text when playback progress cannot be trusted."""

        if not self._desktop_message_open:
            return
        text = self._desktop_llm_text
        if text and len(text) > len(self._desktop_sent_text):
            self._desktop_sent_text = text
            await self._forward("update_resp", {"text": text})

    async def _desktop_close_message(self, text: str) -> None:
        """Close the open message with its complete text and reset state.

        Parameters
        ----------
        text : str
            Complete text used to finalize the open assistant message.
        """

        if not self._desktop_message_open:
            return
        sent = self._desktop_sent_text
        final = text or self._desktop_llm_text or sent
        if final and len(final) > len(sent):
            self._desktop_sent_text = final
            await self._forward("update_resp", {"text": final})
        elif not final and sent:
            final = sent
        if final:
            await self._forward("finish_resp", {"text": final})
        self._desktop_message_open = False
        self._desktop_llm_text = ""
        self._desktop_sent_text = ""
        self._desktop_final_pending = ""
