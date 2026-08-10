# -*- coding: utf-8 -*-
import base64
import json
import logging
from typing import Any

from fastapi import WebSocket

from ...models import Models
from ..event_bus import EventBus
from ..events import (
    ASRResultFinal,
    ASRResultPartial,
    CaptionUpdated,
    ErrorOccurred,
    FullAudioFrameReady,
    LatencyMetricsUpdated,
    ResponseFinish,
    ResponseUpdate,
    RetrievalUpdated,
    SpeakerRecognized,
    ToolCallOccurred,
    TTSChunkReady,
    TTSEmotionChange,
    TTSFinished,
    TTSPaused,
    TTSResumed,
    TTSStarted,
    TTSStopped,
    TTSVoiceChange,
    VADSpeechEnd,
    VADSpeechStart,
)
from ..interfaces import EventListenerMixin
from ..multi_speaker_config import speaker_history_gate_enabled

logger = logging.getLogger(__name__)


class OutputGateway(EventListenerMixin):
    """Forward backend events to the frontend WebSocket.

    Parameters
    ----------
    event_bus : EventBus
        Event bus used to subscribe to session events.
    session_id : str
        Session identifier sent back to the frontend.
    websocket : WebSocket
        Live WebSocket connection used for outbound messages.
    config : dict[str, Any] | None, optional
        Service configuration relevant to output behavior.
    models : Models | None, optional
        Session models used to place ASR previews before the speaker-history
        barrier when the focus-only gate is active.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
        config: dict[str, Any] | None = None,
        models: Models | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket
        self.config: dict[str, Any] = config or {}
        self._speaker_history_gate_enabled = (
            models is not None
            and speaker_history_gate_enabled(models, self.config)
        )
        self._full_audio_chunk_bytes = 128 * 1024

    # ── WebSocket helpers ───────────────────────────────────────────

    def _is_connected(self) -> bool:
        """Check whether the WebSocket is still in an open state."""
        return not (
            hasattr(self.websocket, "client_state")
            and self.websocket.client_state.value != 1
        )

    def _ws_state_name(self) -> str:
        """Return a human-readable name for the current WS state."""
        state = self.websocket.client_state
        return state.name if hasattr(state, "name") else str(state.value)

    async def send_signal(self, message: dict) -> None:
        """Send a JSON payload to the frontend.

        Parameters
        ----------
        message : dict
            JSON-serializable payload to send over the WebSocket.
        """
        if not self._is_connected():
            logger.warning(
                "WebSocket not connected, skip send - session: %s, state: %s",
                self.session_id,
                self._ws_state_name(),
            )
            return
        try:
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            error_msg = str(e).lower()
            if "disconnect" in error_msg or "closed" in error_msg:
                logger.warning(
                    "WebSocket disconnected, cannot send message - session: %s",
                    self.session_id,
                )
            else:
                logger.error(
                    "Failed to send WebSocket message - session: %s, error: %s",
                    self.session_id,
                    e,
                )

    async def _send_binary(self, data: bytes) -> None:
        """Send raw binary data (e.g., audio chunks)."""
        if not self._is_connected():
            logger.warning(
                "WebSocket not connected, skip audio send - session: %s, state: %s",
                self.session_id,
                self._ws_state_name(),
            )
            return
        try:
            await self.websocket.send_bytes(data)
        except Exception as e:
            error_msg = str(e).lower()
            if "disconnect" in error_msg or "closed" in error_msg:
                logger.warning(
                    "WebSocket disconnected, cannot send audio - session: %s",
                    self.session_id,
                )
            else:
                logger.error(
                    "Failed to send binary data - session: %s, error: %s",
                    self.session_id,
                    e,
                )

    # ── Message building ────────────────────────────────────────────

    @staticmethod
    def _build_message(action: str, data: Any) -> dict:
        """Build a simple JSON payload for frontend consumption."""
        return {"action": action, "data": data}

    async def _forward(self, action: str, data: Any) -> None:
        """Build a message and send it, with unified error logging."""
        try:
            await self.send_signal(self._build_message(action, data))
        except Exception as e:
            logger.error(
                "Failed to send %s signal - session: %s, error: %s",
                action,
                self.session_id,
                e,
            )

    # ── Shared helpers ──────────────────────────────────────────────

    async def _forward_asr(
        self,
        action: str,
        event: ASRResultPartial | ASRResultFinal,
    ) -> None:
        """Forward an ASR result (partial or final) to the frontend."""
        display_text = event.display_text or event.text
        await self._forward(
            action,
            {
                "text": display_text,
                "origin": event.origin,
            },
        )

    # ── Session ─────────────────────────────────────────────────────

    async def send_session_attached(self) -> None:
        """Send the attached session identifier to the frontend."""
        await self._forward("session_attached", {"session_id": self.session_id})

    # ── Event handlers ──────────────────────────────────────────────

    @EventListenerMixin.event_handler(
        ASRResultPartial,
        priority=40,
        enabled_if=lambda gateway: gateway._speaker_history_gate_enabled,
    )
    async def _send_gated_update_asr_signal(
        self,
        event: ASRResultPartial,
    ) -> None:
        """Forward a focus-gated preview before the history barrier."""

        await self._forward_asr("update_asr", event)

    @EventListenerMixin.event_handler(
        ASRResultPartial,
        priority=5,
        enabled_if=lambda gateway: not gateway._speaker_history_gate_enabled,
    )
    async def _send_update_asr_signal(self, event: ASRResultPartial) -> None:
        await self._forward_asr("update_asr", event)

    @EventListenerMixin.event_handler(ASRResultFinal, priority=5)
    async def _send_finish_asr_signal(self, event: ASRResultFinal) -> None:
        await self._forward_asr("finish_asr", event)

    @EventListenerMixin.event_handler(TTSStarted, priority=5)
    async def _on_start_tts(self, event: TTSStarted) -> None:
        await self._forward("start_tts", {"response_id": event.response_id})

    @EventListenerMixin.event_handler(TTSStopped, priority=5)
    async def _send_stop_tts_signal(self, event: TTSStopped) -> None:
        await self._forward("stop_tts", {"response_id": event.response_id})

    @EventListenerMixin.event_handler(ErrorOccurred, priority=5)
    async def _send_error_signal(self, event: ErrorOccurred) -> None:
        await self._forward("error", event.error_message)

    @EventListenerMixin.event_handler(TTSPaused, priority=5)
    async def _send_pause_tts_signal(self, event) -> None:
        await self._forward("pause_tts", "")

    @EventListenerMixin.event_handler(TTSResumed, priority=5)
    async def _send_resume_tts_signal(self, event) -> None:
        await self._forward("resume_tts", "")

    @EventListenerMixin.event_handler(ResponseUpdate, priority=5)
    async def _send_update_resp_signal(self, event: ResponseUpdate) -> None:
        await self._forward(
            "update_resp",
            {"response_id": event.response_id, "text": event.text},
        )

    @EventListenerMixin.event_handler(ResponseFinish, priority=5)
    async def _send_finish_resp_signal(self, event: ResponseFinish) -> None:
        await self._forward(
            "finish_resp",
            {"response_id": event.response_id, "text": event.text},
        )

    @EventListenerMixin.event_handler(TTSFinished, priority=5)
    async def _send_tts_finished_signal(self, event: TTSFinished) -> None:
        await self._forward("tts_finished", {"response_id": event.response_id})

    @EventListenerMixin.event_handler(SpeakerRecognized, priority=5)
    async def _send_speaker_updated_signal(self, event: SpeakerRecognized) -> None:
        await self._forward(
            "speaker_updated",
            {
                "speaker_id": event.speaker_id,
                "reason": event.reason,
            },
        )

    @EventListenerMixin.event_handler(CaptionUpdated, priority=5)
    async def _send_caption_updated(self, event: CaptionUpdated) -> None:
        await self._forward(
            "caption_updated",
            {
                "text": event.text or "",
                "is_final": bool(event.is_final),
                "reason": event.reason,
            },
        )

    @EventListenerMixin.event_handler(RetrievalUpdated, priority=5)
    async def _send_retrieval_updated(self, event: RetrievalUpdated) -> None:
        await self._forward(
            "retrieval_updated",
            {
                "text": event.text or "",
                "is_final": bool(event.is_final),
            },
        )

    @EventListenerMixin.event_handler(TTSChunkReady, priority=5)
    async def _send_tts_chunk_signal(self, event: TTSChunkReady) -> None:
        if hasattr(event, "audio_chunk") and event.audio_chunk:
            try:
                await self._send_binary(event.audio_chunk)
            except Exception as e:
                logger.error(
                    "Failed to send TTS chunk - session: %s, error: %s",
                    self.session_id,
                    e,
                )

    @EventListenerMixin.event_handler(FullAudioFrameReady, priority=5)
    async def _send_full_audio_frame_signal(self, event: FullAudioFrameReady) -> None:
        if not event.audio_chunk:
            return
        total_bytes = len(event.audio_chunk)
        for start in range(0, total_bytes, self._full_audio_chunk_bytes):
            chunk = event.audio_chunk[start : start + self._full_audio_chunk_bytes]
            await self._forward(
                "full_audio_frame",
                {
                    "audio_base64": base64.b64encode(chunk).decode("ascii"),
                    "sample_rate": int(event.sample_rate),
                    "channels": int(event.channels),
                    "format": event.format,
                },
            )

    @EventListenerMixin.event_handler(TTSVoiceChange, priority=5)
    async def _send_voice_changed_signal(self, event: TTSVoiceChange) -> None:
        await self._forward("voice_changed", {"name": event.voice_name})

    @EventListenerMixin.event_handler(VADSpeechStart, priority=5)
    async def _send_vad_start_signal(self, event: VADSpeechStart) -> None:
        if event.origin == "client":
            return
        await self._forward("vad_speech_start", {"origin": event.origin})

    @EventListenerMixin.event_handler(VADSpeechEnd, priority=5)
    async def _send_vad_end_signal(self, event: VADSpeechEnd) -> None:
        if event.origin == "client":
            return
        await self._forward("vad_speech_end", {"origin": event.origin})

    @EventListenerMixin.event_handler(TTSEmotionChange, priority=5)
    async def _send_emotion_changed_signal(self, event: TTSEmotionChange) -> None:
        await self._forward(
            "emotion_changed",
            {
                "name": event.emotion_name,
                "vector": event.emotion_vector or [],
            },
        )

    @EventListenerMixin.event_handler(LatencyMetricsUpdated, priority=5)
    async def _send_latency_metrics_signal(self, event: LatencyMetricsUpdated) -> None:
        await self._forward(
            "latency_metrics",
            {
                "network_latency_ms": int(event.network_latency_ms),
                "asr_latency_ms": int(event.asr_latency_ms),
                "llm_first_token_ms": int(event.llm_first_token_ms),
                "llm_sentence_ms": int(event.llm_sentence_ms),
                "tts_first_chunk_ms": int(event.tts_first_chunk_ms),
            },
        )

    @EventListenerMixin.event_handler(ToolCallOccurred, priority=5)
    async def _send_tool_called_signal(self, event: ToolCallOccurred) -> None:
        if event.name in {"tool_call_result", "direct_audio"}:
            return
        await self._forward(
            "tool_called",
            {
                "name": event.name,
                "args": event.args or {},
            },
        )
