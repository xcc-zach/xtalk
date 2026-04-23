# -*- coding: utf-8 -*-
import base64
import json
from fastapi import WebSocket
from ...log_utils import logger
from typing import Any

from ..event_bus import EventBus
from ..interfaces import EventListenerMixin
from ..events import (
    ASRResultPartial,
    ASRResultFinal,
    VADSpeechStart,
    VADSpeechEnd,
    TTSStarted,
    TTSStopped,
    TTSPaused,
    TTSResumed,
    TTSFinished,
    LLMAgentResponseUpdate,
    LLMAgentResponseFinish,
    ErrorOccurred,
    TTSChunkGenerated,
    TTSVoiceChange,
    TTSEmotionChange,
    ThoughtUpdated,
    CaptionUpdated,
    LatencyMetricsUpdated,
    ToolCallOccurred,
    RetrievalUpdated,
    SpeakerRecognized,
    FullAudioFrameReady,
)


class OutputGateway(EventListenerMixin):
    """Send conversation events to the frontend over WebSocket."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket
        self.config: dict[str, Any] = config or {}

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
        """Send a JSON message to the WebSocket if still connected."""
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

    async def _forward_asr(self, action: str, event) -> None:
        """Forward an ASR result (partial or final) to the frontend."""
        display_text = getattr(event, "display_text", "") or event.text
        await self._forward(
            action,
            {
                "text": display_text,
                "confidence": getattr(event, "confidence", 0.0),
                "is_final": getattr(event, "is_final", action == "finish_asr"),
                "turn_id": event.turn_id,
            },
        )

    # ── Session ─────────────────────────────────────────────────────

    async def send_session_info(self) -> None:
        """Send current session metadata to the frontend."""
        await self._forward("session_info", {"session_id": self.session_id})

    # ── Event handlers ──────────────────────────────────────────────

    @EventListenerMixin.event_handler(ASRResultPartial, priority=5)
    async def _send_update_asr_signal(self, event: ASRResultPartial) -> None:
        await self._forward_asr("update_asr", event)

    @EventListenerMixin.event_handler(ASRResultFinal, priority=5)
    async def _send_finish_asr_signal(self, event: ASRResultFinal) -> None:
        await self._forward_asr("finish_asr", event)

    @EventListenerMixin.event_handler(TTSStarted, priority=5)
    async def _on_start_tts(self, event) -> None:
        await self._forward("start_tts", "")

    @EventListenerMixin.event_handler(TTSStopped, priority=5)
    async def _send_stop_tts_signal(self, event) -> None:
        await self._forward("stop_tts", "")

    @EventListenerMixin.event_handler(ErrorOccurred, priority=5)
    async def _send_error_signal(self, event: ErrorOccurred) -> None:
        await self._forward("error", event.error_message)

    @EventListenerMixin.event_handler(TTSPaused, priority=5)
    async def _send_pause_tts_signal(self, event) -> None:
        await self._forward("pause_tts", "")

    @EventListenerMixin.event_handler(TTSResumed, priority=5)
    async def _send_resume_tts_signal(self, event) -> None:
        await self._forward("resume_tts", "")

    @EventListenerMixin.event_handler(LLMAgentResponseUpdate, priority=5)
    async def _send_update_resp_signal(self, event: LLMAgentResponseUpdate) -> None:
        await self._forward(
            "update_resp", {"text": event.text, "turn_id": event.turn_id}
        )

    @EventListenerMixin.event_handler(LLMAgentResponseFinish, priority=5)
    async def _send_finish_resp_signal(self, event: LLMAgentResponseFinish) -> None:
        await self._forward(
            "finish_resp", {"text": event.text, "turn_id": event.turn_id}
        )

    @EventListenerMixin.event_handler(TTSFinished, priority=5)
    async def _send_tts_finished_signal(self, event: TTSFinished) -> None:
        await self._forward("tts_finished", {})

    @EventListenerMixin.event_handler(SpeakerRecognized, priority=5)
    async def _send_speaker_updated_signal(self, event: SpeakerRecognized) -> None:
        await self._forward(
            "speaker_updated",
            {
                "speaker_id": getattr(event, "speaker_id", None),
                "reason": getattr(event, "reason", ""),
            },
        )

    @EventListenerMixin.event_handler(ThoughtUpdated, priority=5)
    async def _send_thought_updated(self, event: ThoughtUpdated) -> None:
        await self._forward(
            "thought_updated",
            {
                "text": getattr(event, "text", "") or "",
                "is_final": bool(getattr(event, "is_final", False)),
            },
        )

    @EventListenerMixin.event_handler(CaptionUpdated, priority=5)
    async def _send_caption_updated(self, event: CaptionUpdated) -> None:
        await self._forward(
            "caption_updated",
            {
                "text": getattr(event, "text", "") or "",
                "is_final": bool(getattr(event, "is_final", False)),
                "reason": getattr(event, "reason", ""),
            },
        )

    @EventListenerMixin.event_handler(RetrievalUpdated, priority=5)
    async def _send_retrieval_updated(self, event: RetrievalUpdated) -> None:
        await self._forward(
            "retrieval_updated",
            {
                "text": getattr(event, "text", "") or "",
                "is_final": bool(getattr(event, "is_final", False)),
            },
        )

    @EventListenerMixin.event_handler(TTSChunkGenerated, priority=5)
    async def _send_tts_chunk_signal(self, event: TTSChunkGenerated) -> None:
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
        await self._forward(
            "full_audio_frame",
            {
                "audio_base64": base64.b64encode(event.audio_chunk).decode("ascii"),
                "sample_rate": int(getattr(event, "sample_rate", 48000)),
                "channels": int(getattr(event, "channels", 2)),
                "format": getattr(event, "format", "pcm_s16le"),
            },
        )

    @EventListenerMixin.event_handler(TTSVoiceChange, priority=5)
    async def _send_voice_changed_signal(self, event: TTSVoiceChange) -> None:
        await self._forward("voice_changed", {"name": event.voice_name})

    @EventListenerMixin.event_handler(VADSpeechStart, priority=5)
    async def _send_vad_start_signal(self, event: VADSpeechStart) -> None:
        if event.origin != "server":
            return
        await self._forward("vad_speech_start", {"origin": event.origin})

    @EventListenerMixin.event_handler(VADSpeechEnd, priority=5)
    async def _send_vad_end_signal(self, event: VADSpeechEnd) -> None:
        if event.origin != "server":
            return
        await self._forward("vad_speech_end", {"origin": event.origin})

    @EventListenerMixin.event_handler(TTSEmotionChange, priority=5)
    async def _send_emotion_changed_signal(self, event: TTSEmotionChange) -> None:
        await self._forward(
            "emotion_changed",
            {
                "name": getattr(event, "emotion_name", ""),
                "vector": getattr(event, "emotion_vector", []) or [],
            },
        )

    @EventListenerMixin.event_handler(LatencyMetricsUpdated, priority=5)
    async def _send_latency_metrics_signal(self, event: LatencyMetricsUpdated) -> None:
        await self._forward(
            "latency_metrics",
            {
                "network_latency_ms": int(getattr(event, "network_latency_ms", 0)),
                "asr_latency_ms": int(getattr(event, "asr_latency_ms", 0)),
                "llm_first_token_ms": int(getattr(event, "llm_first_token_ms", 0)),
                "llm_sentence_ms": int(getattr(event, "llm_sentence_ms", 0)),
                "tts_first_chunk_ms": int(getattr(event, "tts_first_chunk_ms", 0)),
                "e2e_latency_ms": int(getattr(event, "e2e_latency_ms", 0)),
            },
        )

    @EventListenerMixin.event_handler(ToolCallOccurred, priority=5)
    async def _send_tool_called_signal(self, event: ToolCallOccurred) -> None:
        if getattr(event, "name", "") == "tool_call_result":
            return
        await self._forward(
            "tool_called",
            {
                "name": getattr(event, "name", ""),
                "args": getattr(event, "args", {}) or {},
            },
        )
