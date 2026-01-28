# -*- coding: utf-8 -*-
import json
from fastapi import WebSocket
from dataclasses import dataclass
from ...log_utils import logger
from typing import Any, Callable, Type

from ..event_bus import EventBus
from ..interfaces import EventListenerMixin, Manager
from ..events import (
    BaseEvent,
    ASRResultPartial,
    ASRResultFinal,
    VerificationResult,
    TTSStarted,
    TTSStopped,
    TTSPaused,
    TTSResumed,
    TTSFinished,
    LLMAgentResponseUpdate,
    LLMAgentResponseFinish,
    ErrorOccurred,
    TTSChunkGenerated,
    TTSPlaybackFinished,
    TTSVoiceChange,
    TTSEmotionChange,
)
from ..events import (
    TranscriptionRefined,
    ThoughtUpdated,
    CaptionUpdated,
    LatencyMetricsUpdated,
    ToolCallOccurred,
    RetrievalUpdated,
    SpeakerRecognized,
)


@dataclass
class ModulesState:
    """State snapshot for downstream modules."""

    tts_active: bool = False  # Whether TTS is currently active


class OutputGateway(EventListenerMixin):
    """Send conversation events to the frontend over WebSocket."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the signal dispatcher.

        Args:
            session_id: unique session identifier
            event_bus: shared event bus
            websocket: active WebSocket connection
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket
        # Per-session config
        self.config: dict[str, Any] = config or {}

        self.state = ModulesState()

    async def send_session_info(self) -> None:
        """Send current session metadata to the frontend."""
        try:
            await self.send_signal(
                {
                    "action": "session_info",
                    "data": {"session_id": self.session_id},
                }
            )
        except Exception as e:
            logger.error(
                "Failed to send session_info signal to frontend - session: %s, error: %s",
                self.session_id,
                e,
            )

    def _build_message(self, action: str, data: Any, event: Any | None = None) -> dict:
        """Build a simple JSON payload for frontend consumption."""
        return {"action": action, "data": data}

    async def send_signal(self, message: dict) -> None:
        """Send a JSON message to the WebSocket if still connected."""
        try:
            if (
                hasattr(self.websocket, "client_state")
                and self.websocket.client_state.value != 1
            ):
                logger.warning(
                    "WebSocket not connected, skip send - session: %s, state: %s",
                    self.session_id,
                    (
                        self.websocket.client_state.name
                        if hasattr(self.websocket.client_state, "name")
                        else self.websocket.client_state.value
                    ),
                )
                return
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            # Detect disconnect errors
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
        try:
            # Ensure the socket is still open
            if (
                hasattr(self.websocket, "client_state")
                and self.websocket.client_state.value != 1
            ):
                logger.warning(
                    "WebSocket not connected, skip audio send - session: %s, state: %s",
                    self.session_id,
                    (
                        self.websocket.client_state.name
                        if hasattr(self.websocket.client_state, "name")
                        else self.websocket.client_state.value
                    ),
                )
                return
            await self.websocket.send_bytes(data)
        except Exception as e:
            # Detect disconnect errors
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

    @EventListenerMixin.event_handler(TTSPlaybackFinished, priority=5)
    async def _on_tts_playback_finished(self, event) -> None:
        self.state.tts_active = False

    @EventListenerMixin.event_handler(ASRResultPartial, priority=5)
    async def _send_update_asr_signal(self, event: ASRResultPartial) -> None:
        """Send partial ASR update to the frontend."""
        try:
            # Prefer display_text when available
            display_text = getattr(event, "display_text", "") or event.text

            await self.send_signal(
                self._build_message(
                    "update_asr",
                    {
                        "text": display_text,
                        "confidence": getattr(event, "confidence", 0.0),
                        "is_final": getattr(event, "is_final", False),
                        "turn_id": event.turn_id,
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send update_asr signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(ASRResultFinal, priority=5)
    async def _send_finish_asr_signal(self, event: ASRResultFinal) -> None:
        """Send final ASR result to the frontend."""
        try:
            # Prefer display_text when available
            display_text = getattr(event, "display_text", "") or event.text

            await self.send_signal(
                self._build_message(
                    "finish_asr",
                    {
                        "text": display_text,
                        "confidence": getattr(event, "confidence", 0.0),
                        "is_final": getattr(event, "is_final", True),
                        "turn_id": event.turn_id,
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send finish_asr signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(VerificationResult, priority=5)
    async def _handle_verification_result(self, event: VerificationResult) -> None:
        """Process ASR verification results and send follow-up control signals."""
        is_valid = event.is_valid

        if not self.state.tts_active:
            if not is_valid:
                await self.send_signal(
                    self._build_message("invalid_asr_result", "", event)
                )

            return

    @EventListenerMixin.event_handler(TTSStarted, priority=5)
    async def _on_start_tts(self, event) -> None:
        self.state.tts_active = True
        try:

            await self.send_signal(self._build_message("start_tts", "", event))
        except Exception as e:
            logger.error(
                "Failed to send start_tts signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TTSStopped, priority=5)
    async def _send_stop_tts_signal(self, event) -> None:
        """Send stop_tts signal to the frontend."""
        try:
            await self.send_signal(self._build_message("stop_tts", "", event))

        except Exception as e:
            logger.error(
                "Failed to send stop_tts signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(ErrorOccurred, priority=5)
    async def _send_error_signal(self, event: ErrorOccurred) -> None:
        """Forward backend error events to the frontend."""
        try:
            await self.send_signal(
                self._build_message("error", event.error_message, event)
            )

        except Exception as e:
            logger.error(
                "Failed to send error signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TranscriptionRefined, priority=5)
    async def _send_refine_transcription_signal(
        self, event: TranscriptionRefined
    ) -> None:
        try:
            await self.send_signal(
                self._build_message(
                    "refine_transcription",
                    {"text": event.text},
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send refine_transcription signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TTSPaused, priority=5)
    async def _send_pause_tts_signal(self, event) -> None:
        """Send pause_tts signal to the frontend."""
        try:
            await self.send_signal(self._build_message("pause_tts", "", event))

        except Exception as e:
            logger.error(
                "Failed to send pause_tts signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TTSResumed, priority=5)
    async def _send_resume_tts_signal(self, event) -> None:
        """Send resume_tts signal to the frontend."""
        try:
            await self.send_signal(self._build_message("resume_tts", "", event))

        except Exception as e:
            logger.error(
                "Failed to send resume_tts signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(LLMAgentResponseUpdate, priority=5)
    async def _send_update_resp_signal(self, event: LLMAgentResponseUpdate) -> None:
        """Send incremental assistant response updates to the frontend."""
        try:
            await self.send_signal(
                self._build_message(
                    "update_resp",
                    {"text": event.text, "turn_id": event.turn_id},
                    event,
                )
            )

        except Exception as e:
            logger.error(
                "Failed to send update_resp signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(LLMAgentResponseFinish, priority=5)
    async def _send_finish_resp_signal(self, event: LLMAgentResponseFinish) -> None:
        """Send final assistant response update to the frontend."""
        try:
            await self.send_signal(
                self._build_message(
                    "finish_resp",
                    {"text": event.text, "turn_id": event.turn_id},
                    event,
                )
            )

        except Exception as e:
            logger.error(
                "Failed to send finish_resp signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TTSFinished, priority=5)
    async def _send_tts_finished_signal(self, event: TTSFinished):
        await self.send_signal(
            self._build_message(
                "tts_finished",
                {},
                event,
            )
        )

    @EventListenerMixin.event_handler(SpeakerRecognized, priority=5)
    async def _send_speaker_updated_signal(self, event: SpeakerRecognized) -> None:
        """Send speaker recognition status to the frontend."""
        try:
            await self.send_signal(
                self._build_message(
                    "speaker_updated",
                    {
                        "speaker_id": getattr(event, "speaker_id", None),
                        "reason": getattr(event, "reason", ""),
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send speaker_updated signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(ThoughtUpdated, priority=5)
    async def _send_thought_updated(self, event: ThoughtUpdated) -> None:
        """Send Thought updates to the frontend."""
        try:
            await self.send_signal(
                self._build_message(
                    "thought_updated",
                    {
                        "text": getattr(event, "text", "") or "",
                        "is_final": bool(getattr(event, "is_final", False)),
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send thought_updated signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(CaptionUpdated, priority=5)
    async def _send_caption_updated(self, event: CaptionUpdated) -> None:
        """Send Caption updates to the frontend."""
        try:
            await self.send_signal(
                self._build_message(
                    "caption_updated",
                    {
                        "text": getattr(event, "text", "") or "",
                        "is_final": bool(getattr(event, "is_final", False)),
                        "reason": getattr(event, "reason", ""),
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send caption_updated signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(RetrievalUpdated, priority=5)
    async def _send_retrieval_updated(self, event: RetrievalUpdated) -> None:
        """Send Retrieval updates to the frontend."""
        try:
            await self.send_signal(
                self._build_message(
                    "retrieval_updated",
                    {
                        "text": getattr(event, "text", "") or "",
                        "is_final": bool(getattr(event, "is_final", False)),
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send retrieval_updated signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TTSChunkGenerated, priority=5)
    async def _send_tts_chunk_signal(self, event: TTSChunkGenerated) -> None:
        """Send TTS audio chunks (PCM bytes) to the frontend."""
        try:
            if hasattr(event, "audio_chunk") and event.audio_chunk:
                await self._send_binary(event.audio_chunk)

        except Exception as e:
            logger.error(
                "Failed to send TTS chunk - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TTSVoiceChange, priority=5)
    async def _send_voice_changed_signal(self, event: TTSVoiceChange) -> None:
        """Notify frontend that voice reference audio has changed."""
        try:
            await self.send_signal(
                self._build_message(
                    "voice_changed",
                    {
                        "name": event.voice_name,
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send voice_changed signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(TTSEmotionChange, priority=5)
    async def _send_emotion_changed_signal(self, event: TTSEmotionChange) -> None:
        """Notify frontend that TTS emotion has changed."""
        try:
            await self.send_signal(
                self._build_message(
                    "emotion_changed",
                    {
                        "name": getattr(event, "emotion_name", ""),
                        # Optional vector for clients that consume detailed info
                        "vector": getattr(event, "emotion_vector", []) or [],
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send emotion_changed signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(LatencyMetricsUpdated, priority=5)
    async def _send_latency_metrics_signal(self, event: LatencyMetricsUpdated) -> None:
        """Send latency metrics to the frontend."""
        try:
            await self.send_signal(
                self._build_message(
                    "latency_metrics",
                    {
                        "network_latency_ms": int(
                            getattr(event, "network_latency_ms", 0)
                        ),
                        "asr_latency_ms": int(getattr(event, "asr_latency_ms", 0)),
                        "llm_first_token_ms": int(
                            getattr(event, "llm_first_token_ms", 0)
                        ),
                        "llm_sentence_ms": int(getattr(event, "llm_sentence_ms", 0)),
                        "tts_first_chunk_ms": int(
                            getattr(event, "tts_first_chunk_ms", 0)
                        ),
                        "e2e_latency_ms": int(getattr(event, "e2e_latency_ms", 0)),
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send latency_metrics signal - session: %s, error: %s",
                self.session_id,
                e,
            )

    @EventListenerMixin.event_handler(ToolCallOccurred, priority=5)
    async def _send_tool_called_signal(self, event: ToolCallOccurred) -> None:
        """Send tool-call notifications to the frontend."""
        try:
            # tool_call_result is for internal propagation only
            if getattr(event, "name", "") == "tool_call_result":
                return
            await self.send_signal(
                self._build_message(
                    "tool_called",
                    {
                        "name": getattr(event, "name", ""),
                        "args": getattr(event, "args", {}) or {},
                    },
                    event,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to send tool_called signal - session: %s, error: %s",
                self.session_id,
                e,
            )
