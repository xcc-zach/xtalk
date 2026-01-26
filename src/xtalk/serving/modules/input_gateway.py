# -*- coding: utf-8 -*-
import json

from fastapi import WebSocket, WebSocketDisconnect

from ...log_utils import logger

from ..event_bus import EventBus
from ..events import (
    ErrorOccurred,
    WebSocketMessageReceived,
    AudioFrameReceived,
    ConversationEnded,
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
)
from ..interfaces import EventListenerMixin
from typing import Any


class TextMsgHandler(EventListenerMixin):
    """Text signal handler that dispatches frontend control messages to events."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize handler.

        Args:
            session_id: unique session identifier
            websocket: active WebSocket connection
            event_bus: shared event bus
            service: (unused) legacy reference
            gateway: (unused) legacy reference
        """
        self.session_id = session_id
        self.websocket = websocket
        self.event_bus = event_bus
        # Per-session configuration
        self.config: dict[str, Any] = config or {}

        # Event listeners are registered via decorators

    async def _handle_ping(self, message_data: dict, server_recv_ts: float) -> None:
        """Handle heartbeat ping and reply with pong (includes server timestamp)."""
        try:
            pong_message = json.dumps(
                {
                    "action": "pong",
                    "client_timestamp": message_data.get("timestamp", 0),
                    "server_recv_timestamp": int(
                        server_recv_ts * 1000
                    ),  # server receive time (milliseconds)
                }
            )
            await self.websocket.send_text(pong_message)
        except Exception as e:
            logger.error(
                "Failed to respond to ping - session: %s, error: %s",
                self.session_id,
                e,
            )

    async def _handle_clock_sync(self, message_data: dict) -> None:
        """Handle clock-sync payloads so LatencyManager can update offsets."""
        try:
            event = ClockSyncReceived(
                session_id=self.session_id,
                client_send_ts=message_data.get("client_send_ts", 0.0),
                server_recv_ts=message_data.get("server_recv_ts", 0.0),
                client_recv_ts=message_data.get("client_recv_ts", 0.0),
            )
            await self.event_bus.publish(event)
        except Exception as e:
            logger.error(
                "Failed to process clock sync - session: %s, error: %s",
                self.session_id,
                e,
            )

    async def _handle_conversation_start(self, message_data: dict) -> None:
        """Handle frontend conversation_start signal (currently placeholder)."""
        pass

    async def _handle_conversation_end(self, message_data: dict) -> None:
        """Handle frontend conversation_end signal."""
        try:
            reason = message_data.get("reason", "")
            event = ConversationEnded(session_id=self.session_id, reason=reason)
            await self.event_bus.publish(event)
        except Exception as e:
            logger.error(
                "Failed to process conversation_end - session: %s, error: %s",
                self.session_id,
                e,
            )

    async def _handle_vad_signal(self, message_type: str, message_data: dict) -> None:
        """Handle VAD signals and publish internal events for other managers."""

        if message_type == "vad_speech_start":
            vad_event = VADSpeechStart(
                session_id=self.session_id,
                confidence=message_data.get("confidence", 0.8),
            )
            await self.event_bus.publish(vad_event)

        elif message_type == "vad_speech_end":
            # Send empty final frame so ASR knows the segment ended
            final_audio_event = AudioFrameReceived(
                session_id=self.session_id, audio_data=b"", is_final=True
            )
            await self.event_bus.publish(final_audio_event, wait_for_completion=True)
            vad_event = VADSpeechEnd(
                session_id=self.session_id,
                confidence=message_data.get("confidence", 0.8),
            )
            await self.event_bus.publish(vad_event)

    async def _handle_tts_playback_finished(self, message_data: dict) -> None:
        """Handle frontend TTS playback completion and transition to idle."""
        tts_event = TTSPlaybackFinished(session_id=self.session_id)
        await self.event_bus.publish(tts_event)

    async def _handle_change_voice(self, message_data: dict) -> None:
        """Handle requests to change the reference voice."""

        ref_audio_name = message_data.get("voice_name", "")

        event = TTSVoiceChange(
            session_id=self.session_id,
            voice_name=ref_audio_name,
        )
        await self.event_bus.publish(event)

    async def _handle_change_emotion(self, message_data: dict) -> None:
        """Handle requests to change speech emotion."""

        emotion_name = message_data.get("emotion_name", "")
        emotion_vector = message_data.get("emotion_vector", [])

        event = TTSEmotionChange(
            session_id=self.session_id,
            emotion_name=emotion_name,
            emotion_vector=emotion_vector,
        )
        await self.event_bus.publish(event)

    async def _handle_change_tts_speed(self, message_data: dict) -> None:
        """Handle requests to adjust TTS playback speed."""

        speed = message_data.get("speed", 1.0)

        # Restrict speed to a safe range
        speed = max(0.5, min(1.5, float(speed)))

        event = TTSSpeedChange(
            session_id=self.session_id,
            speed=speed,
        )
        await self.event_bus.publish(event)

    async def _handle_change_tts_model(self, message_data: dict) -> None:
        """Handle requests to switch TTS models (IndexTTS / IndexTTS2)."""

        model_type = message_data.get("model_type", "")
        config = message_data.get("config", {})

        if not model_type:
            logger.warning(
                "change_tts_model: model_type is required - session: %s",
                self.session_id,
            )
            return

        event = TTSModelSwitchRequested(
            session_id=self.session_id,
            model_type=model_type,
            config=config,
        )
        await self.event_bus.publish(event)

    async def _handle_change_llm_model(self, message_data: dict) -> None:
        """Handle requests to switch LLM (ChatOpenAI) model/base URL."""

        model_name = message_data.get("model_name", "")
        base_url = message_data.get("base_url", "")
        api_key = message_data.get("api_key", "")
        extra_body = message_data.get("extra_body")  # Optional extra_body payload

        if not model_name:
            logger.warning(
                "change_llm_model: model_name is required - session: %s",
                self.session_id,
            )
            return

        event = LLMModelSwitchRequested(
            session_id=self.session_id,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body,
        )
        await self.event_bus.publish(event)

    async def _handle_tts_chunk_played(self, message_data: dict) -> None:
        """Handle frontend confirmation that a TTS chunk finished playback."""
        event = TTSChunkPlayed(session_id=self.session_id)
        await self.event_bus.publish(event)

    # ==================== Event handler methods ====================

    @EventListenerMixin.event_handler(ErrorOccurred, priority=10)
    async def _handle_error_event(self, event: ErrorOccurred) -> None:
        """Handle backend error events."""
        logger.error(
            "Error event received - session: %s, type: %s, message: %s",
            self.session_id,
            event.error_type,
            event.error_message,
        )

    @EventListenerMixin.event_handler(WebSocketMessageReceived, priority=90)
    async def _handle_websocket_message_received(
        self, event: WebSocketMessageReceived
    ) -> None:
        """Handle WebSocket message events coming from the gateway layer."""
        try:
            message = event.message
            # Timestamp when server received the message (for clock sync)
            server_recv_ts = event.timestamp

            if isinstance(message, str):  # Handle text messages
                try:
                    message_data = json.loads(message)
                    message_type = message_data.get("action") or message_data.get(
                        "type", "unknown"
                    )

                    if message_type == "ping":
                        await self._handle_ping(message_data, server_recv_ts)
                    elif message_type == "clock_sync":
                        await self._handle_clock_sync(message_data)
                    elif message_type == "conversation_start":
                        await self._handle_conversation_start(message_data)
                    elif message_type == "conversation_end":
                        await self._handle_conversation_end(message_data)
                    elif message_type in ["vad_speech_start", "vad_speech_end"]:
                        await self._handle_vad_signal(message_type, message_data)
                    elif message_type == "tts_playback_finished":
                        await self._handle_tts_playback_finished(message_data)
                    elif message_type == "tts_chunk_played":
                        await self._handle_tts_chunk_played(message_data)
                    elif message_type == "change_voice":
                        await self._handle_change_voice(message_data)
                    elif message_type == "change_emotion":
                        await self._handle_change_emotion(message_data)
                    elif message_type == "change_tts_speed":
                        await self._handle_change_tts_speed(message_data)
                    elif message_type == "change_tts_model":
                        await self._handle_change_tts_model(message_data)
                    elif message_type == "change_llm_model":
                        await self._handle_change_llm_model(message_data)
                    else:
                        logger.warning(
                            "Unknown text signal: %s - session: %s",
                            message_type,
                            self.session_id,
                        )

                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse WebSocket text - session: %s, payload: %s",
                        self.session_id,
                        message,
                    )

            else:
                logger.warning(
                    "Unsupported WebSocket message type - session: %s", self.session_id
                )

        except Exception as e:
            logger.error(
                "Failed to handle WebSocket message - session: %s, error: %s",
                self.session_id,
                e,
            )


class InputGateway:
    """WebSocket input gateway responsible for translating inbound frames into events."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize WebSocket input gateway.

        Args:
            event_bus: shared event bus instance
            session_id: unique session identifier
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket
        # Per-session configuration
        self.config: dict[str, Any] = config or {}

        self.text_msg_handler = TextMsgHandler(
            event_bus, session_id, websocket, config=self.config
        )

    async def handle_connection(self, already_accepted: bool = False):
        """
        Handle a new WebSocket connection.

        Args:
            already_accepted: skip websocket.accept() if caller already accepted.
        """
        try:
            if not already_accepted:
                await self.websocket.accept()

        except Exception as e:
            logger.error(
                f"Failed to accept connection - session: {self.session_id}, error: {e}"
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
                f"WebSocket disconnected (WebSocketDisconnect) - session: {self.session_id}, "
                f"code: {getattr(e, 'code', 'N/A')}, reason: {getattr(e, 'reason', 'N/A')}"
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "disconnect" in error_msg or "closed" in error_msg:
                logger.info(
                    f"WebSocket already closed - session: {self.session_id}, detail: {e}"
                )
            else:
                logger.error(
                    f"WebSocket message processing error - session: {self.session_id}, error: {e}"
                )

    async def _process_message(self, data: dict) -> None:
        """Process a raw WebSocket receive payload and publish events."""
        try:
            if data["type"] == "websocket.receive":
                if "text" in data:
                    # Text message
                    await self._handle_text_message(data["text"])
                elif "bytes" in data:
                    # Binary audio payload
                    await self._handle_audio_message(data["bytes"])

        except Exception as e:
            logger.error("Message processing error (%s): %s", self.session_id, e)

    async def _handle_text_message(self, text_message: str) -> None:
        """Publish text messages as WebSocket events without interpreting business logic."""
        try:
            # Attempt to parse JSON
            message = json.loads(text_message)
            msg_type = message.get("action") or message.get("type", "unknown")

            # Always publish the original text for business logic handling
            event = WebSocketMessageReceived(
                session_id=self.session_id, message=text_message
            )
            await self.event_bus.publish(event)

        except json.JSONDecodeError:
            # Non-JSON payloads are still forwarded to the business layer
            event = WebSocketMessageReceived(
                session_id=self.session_id, message=text_message
            )
            await self.event_bus.publish(event)

    async def _handle_audio_message(self, audio_data: bytes) -> None:
        """Publish raw audio frames; enhancer logic runs in EnhancerManager."""
        event = AudioFrameReceived(
            session_id=self.session_id,
            audio_data=audio_data,
            is_final=False,
            sample_rate=16000,
        )
        await self.event_bus.publish(event)
