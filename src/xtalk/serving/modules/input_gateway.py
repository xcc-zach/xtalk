# -*- coding: utf-8 -*-
import json

from fastapi import WebSocket, WebSocketDisconnect

from ...log_utils import logger

from ..event_bus import EventBus
from ..events import (
    ErrorOccurred,
    WebSocketMessageReceived,
    AudioFrameReceived,
    VADSpeechStart,
    VADSpeechEnd,
    TTSPlaybackFinished,
    TTSVoiceChange,
    TTSEmotionChange,
    TTSSpeedChange,
    TTSChunkPlayed,
    TTSModelSwitchRequested,
    LLMModelSwitchRequested,
    ClockSyncReceived,
    SessionConfigReceived,
)
from ..interfaces import EventListenerMixin
from typing import Any, Callable


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def _require(data: dict, key: str, session_id: str, context: str) -> bool:
    """Return True when *key* is present and non-empty; log a warning otherwise."""
    if not data.get(key):
        logger.warning("%s: %s is required - session: %s", context, key, session_id)
        return False
    return True


class TextMsgHandler(EventListenerMixin):
    """Text signal handler that dispatches frontend control messages to events."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
    ):
        """
        Initialize handler.

        Args:
            event_bus: shared event bus
            session_id: unique session identifier
            websocket: active WebSocket connection
        """
        self.session_id = session_id
        self.websocket = websocket
        self.event_bus = event_bus

        # Build dispatch table: message type -> handler callable
        self._dispatch: dict[str, Callable[[dict], Any]] = {
            "ping": self._handle_ping,
            "clock_sync": self._handle_clock_sync,
            "vad_speech_start": self._handle_vad_speech_start,
            "vad_speech_end": self._handle_vad_speech_end,
            "tts_playback_finished": self._handle_tts_playback_finished,
            "tts_chunk_played": self._handle_tts_chunk_played,
            "change_voice": self._handle_change_voice,
            "change_emotion": self._handle_change_emotion,
            "change_tts_speed": self._handle_change_tts_speed,
            "change_tts_model": self._handle_change_tts_model,
            "change_llm_model": self._handle_change_llm_model,
            "session_config": self._handle_session_config,
        }

    # ==================== Private helpers ====================

    async def _publish(self, event) -> None:
        """Publish an event on the bus (convenience wrapper)."""
        await self.event_bus.publish(event)

    # ==================== Message handlers ====================

    async def _handle_ping(self, message_data: dict) -> None:
        """Handle heartbeat ping and reply with pong (includes server timestamp)."""
        pong_message = json.dumps(
            {
                "action": "pong",
                "client_timestamp": message_data.get("timestamp", 0),
                "server_recv_timestamp": int(
                    message_data.get("_server_recv_ts", 0) * 1000
                ),
            }
        )
        await self.websocket.send_text(pong_message)

    async def _handle_clock_sync(self, message_data: dict) -> None:
        """Handle clock-sync payloads so LatencyManager can update offsets."""
        await self._publish(
            ClockSyncReceived(
                session_id=self.session_id,
                client_send_ts=message_data.get("client_send_ts", 0.0),
                server_recv_ts=message_data.get("server_recv_ts", 0.0),
                client_recv_ts=message_data.get("client_recv_ts", 0.0),
            )
        )

    async def _handle_vad_speech_start(self, message_data: dict) -> None:
        """Handle VAD speech-start signal."""
        await self._publish(
            VADSpeechStart(session_id=self.session_id, origin="client")
        )

    async def _handle_vad_speech_end(self, message_data: dict) -> None:
        """Handle VAD speech-end signal after flushing higher-priority listeners."""
        await self.event_bus.publish(
            VADSpeechEnd(
                session_id=self.session_id,
                origin="client",
            ),
            wait_for_completion=True,
        )

    async def _handle_tts_playback_finished(self, message_data: dict) -> None:
        """Handle frontend TTS playback completion and transition to idle."""
        await self._publish(TTSPlaybackFinished(session_id=self.session_id))

    async def _handle_change_voice(self, message_data: dict) -> None:
        """Handle requests to change the reference voice."""
        await self._publish(
            TTSVoiceChange(
                session_id=self.session_id,
                voice_name=message_data.get("voice_name", ""),
            )
        )

    async def _handle_change_emotion(self, message_data: dict) -> None:
        """Handle requests to change speech emotion."""
        await self._publish(
            TTSEmotionChange(
                session_id=self.session_id,
                emotion_name=message_data.get("emotion_name", ""),
                emotion_vector=message_data.get("emotion_vector", []),
            )
        )

    async def _handle_change_tts_speed(self, message_data: dict) -> None:
        """Handle requests to adjust TTS playback speed."""
        speed = _clamp(float(message_data.get("speed", 1.0)), 0.5, 1.5)
        await self._publish(TTSSpeedChange(session_id=self.session_id, speed=speed))

    async def _handle_change_tts_model(self, message_data: dict) -> None:
        """Handle requests to switch TTS models (IndexTTS / IndexTTS2)."""
        model_type = message_data.get("model_type", "")
        if not _require(
            message_data, "model_type", self.session_id, "change_tts_model"
        ):
            return
        await self._publish(
            TTSModelSwitchRequested(
                session_id=self.session_id,
                model_type=model_type,
                config=message_data.get("config", {}),
            )
        )

    async def _handle_change_llm_model(self, message_data: dict) -> None:
        """Handle requests to switch LLM (ChatOpenAI) model/base URL."""
        if not _require(
            message_data, "model_name", self.session_id, "change_llm_model"
        ):
            return
        await self._publish(
            LLMModelSwitchRequested(
                session_id=self.session_id,
                model_name=message_data.get("model_name", ""),
                base_url=message_data.get("base_url", ""),
                api_key=message_data.get("api_key", ""),
                extra_body=message_data.get("extra_body"),
            )
        )

    async def _handle_tts_chunk_played(self, message_data: dict) -> None:
        """Handle frontend confirmation that a TTS chunk finished playback."""
        await self._publish(TTSChunkPlayed(session_id=self.session_id))

    async def _handle_session_config(self, message_data: dict) -> None:
        """Handle per-session configuration from client."""
        await self._publish(
            SessionConfigReceived(
                session_id=self.session_id,
                recording_path=message_data.get("recording_path"),
            )
        )

    # ==================== Incoming event handlers ====================

    @EventListenerMixin.event_handler(WebSocketMessageReceived, priority=90)
    async def _handle_websocket_message_received(
        self, event: WebSocketMessageReceived
    ) -> None:
        """Handle WebSocket message events coming from the gateway layer."""
        message = event.message

        try:
            message_data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse WebSocket text - session: %s, payload: %s",
                self.session_id,
                message,
            )
            return

        # Inject server receive timestamp for ping handler
        message_data["_server_recv_ts"] = event.timestamp

        message_type = message_data.get("action") or message_data.get("type", "unknown")

        handler = self._dispatch.get(message_type)
        if handler is not None:
            await handler(message_data)
        else:
            logger.warning(
                "Unknown text signal: %s - session: %s",
                message_type,
                self.session_id,
            )


class InputGateway(EventListenerMixin):
    """WebSocket input gateway responsible for translating inbound frames into events."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
    ):
        """
        Initialize WebSocket input gateway.

        Args:
            event_bus: shared event bus instance
            session_id: unique session identifier
            websocket: active WebSocket connection
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket

        self.text_msg_handler = TextMsgHandler(event_bus, session_id, websocket)

    async def handle_connection(self, already_accepted: bool = False) -> None:
        """
        Handle a new WebSocket connection.

        Args:
            already_accepted: skip websocket.accept() if caller already accepted.
        """
        if not already_accepted:
            try:
                await self.websocket.accept()
            except Exception as e:
                logger.error(
                    "Failed to accept connection - session: %s, error: %s",
                    self.session_id,
                    e,
                )

    async def handle_message_loop(self) -> None:
        """Main receive loop that dispatches WebSocket messages."""
        try:
            while True:
                data = await self.websocket.receive()
                if data.get("type") == "websocket.disconnect":
                    break

                await self._process_message(data)
        except WebSocketDisconnect as e:
            logger.info(
                "WebSocket disconnected - session: %s, code: %s, reason: %s",
                self.session_id,
                getattr(e, "code", "N/A"),
                getattr(e, "reason", "N/A"),
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "disconnect" in error_msg or "closed" in error_msg:
                logger.info(
                    "WebSocket already closed - session: %s, detail: %s",
                    self.session_id,
                    e,
                )
            else:
                logger.error(
                    "WebSocket message processing error - session: %s, error: %s",
                    self.session_id,
                    e,
                )

    async def _process_message(self, data: dict) -> None:
        """Process a raw WebSocket receive payload and publish events."""
        if data["type"] != "websocket.receive":
            return

        if "text" in data:
            await self._handle_text_message(data["text"])
        elif "bytes" in data:
            await self._handle_audio_message(data["bytes"])

    async def _handle_text_message(self, text_message: str) -> None:
        """Publish text messages as WebSocket events without interpreting business logic."""
        await self.event_bus.publish(
            WebSocketMessageReceived(session_id=self.session_id, message=text_message)
        )

    async def _handle_audio_message(self, audio_data: bytes) -> None:
        """Publish raw audio frames; enhancer logic runs in EnhancerManager."""
        await self.event_bus.publish(
            AudioFrameReceived(
                session_id=self.session_id,
                audio_data=audio_data,
                is_final=False,
                sample_rate=16000,
            )
        )

    @EventListenerMixin.event_handler(ErrorOccurred, priority=10)
    async def _handle_error_event(self, event: ErrorOccurred) -> None:
        """Handle backend error events."""
        logger.error(
            "Error event received - session: %s, type: %s, message: %s",
            self.session_id,
            event.error_type,
            event.error_message,
        )
