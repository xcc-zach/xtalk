import asyncio
import base64
import binascii
import copy
import json
import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import numpy as np
import soxr
from pydantic import TypeAdapter, ValidationError

from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    AgentContextAppendRequested,
    AgentInstructionsUpdateRequested,
    ASRResultFinal,
    ASRResultPartial,
    AudioFrameReceived,
    AudioStreamControlRequested,
    AudioStreamReady,
    ClientEvent,
    ClockSyncReceived,
    ConversationItemCreate,
    ErrorOccurred,
    InputAudioAppend,
    InputAudioControl,
    InputSpeechStart,
    LLMModelSwitchRequested,
    PlaybackAcknowledgement,
    PlaybackStopped,
    RealtimeSession,
    ResponseCancelRequested,
    SessionConfigReceived,
    SessionUpdate,
    TTSChunkPlayed,
    TTSConfigurationUpdateRequested,
    TTSEmotionChange,
    TTSModelSwitchRequested,
    TTSPlaybackFinished,
    TTSPlaybackStopped,
    TTSSpeedChange,
    TTSVoiceChange,
    TurnInputAbortRequested,
    VADSpeechEnd,
    VADSpeechStart,
    VADThresholdUpdateRequested,
    WebSocketMessageReceived,
)
from ..interfaces import Connection as WebSocket
from ..interfaces import EventListenerMixin

_CLIENT_EVENT_ADAPTER = TypeAdapter(ClientEvent)
_AUDIO_CONTROL_ADAPTER = TypeAdapter(InputAudioControl)
_SPEECH_START_ADAPTER = TypeAdapter(InputSpeechStart)
_SESSION_UPDATE_ADAPTER = TypeAdapter(SessionUpdate)
_SESSION_ADAPTER = TypeAdapter(RealtimeSession)
_AUDIO_APPEND_ADAPTER = TypeAdapter(InputAudioAppend)
_ITEM_CREATE_ADAPTER = TypeAdapter(ConversationItemCreate)
_PLAYBACK_ADAPTER = TypeAdapter(PlaybackAcknowledgement)
_PLAYBACK_STOPPED_ADAPTER = TypeAdapter(PlaybackStopped)

logger = logging.getLogger(__name__)

MAX_TEXT_INPUT_BYTES = 8 * 1024


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def _require(data: dict, key: str, session_id: str, context: str) -> bool:
    """Return True when *key* is present and non-empty; log a warning otherwise."""
    if not data.get(key):
        logger.warning("%s: %s is required - session: %s", context, key, session_id)
        return False
    return True


def _resolve_input_sample_rate(config: dict[str, Any] | None) -> int:
    """Resolve the configured input sample rate with a safe default."""
    if config is None:
        return 16000

    configured_sample_rate = config.get("input_sample_rate", 16000)
    try:
        return int(configured_sample_rate)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid input_sample_rate %r; falling back to 16000",
            configured_sample_rate,
        )
        return 16000


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
        self._text_turn_lock = asyncio.Lock()

        # Build dispatch table: message type -> handler callable
        self._dispatch: dict[str, Callable[[dict[str, Any]], Any]] = {
            "clock_sync": self._handle_clock_sync,
            "submit_text": self._handle_submit_text,
            "vad_speech_start": self._handle_vad_speech_start,
            "vad_speech_end": self._handle_vad_speech_end,
            "tts_playback_finished": self._handle_tts_playback_finished,
            "tts_playback_stopped": self._handle_tts_playback_stopped,
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

    async def _handle_submit_text(self, message_data: dict[str, Any]) -> None:
        """Publish one validated text submission as a synthetic speech turn."""
        text = message_data.get("text")
        if not isinstance(text, str):
            logger.warning(
                "submit_text requires a string - session: %s",
                self.session_id,
            )
            return

        normalized_text = text.strip()
        if not normalized_text:
            logger.warning(
                "submit_text requires non-blank text - session: %s",
                self.session_id,
            )
            return
        if len(normalized_text.encode("utf-8")) > MAX_TEXT_INPUT_BYTES:
            logger.warning(
                "submit_text exceeds %s bytes - session: %s",
                MAX_TEXT_INPUT_BYTES,
                self.session_id,
            )
            return

        async with self._text_turn_lock:
            await self.event_bus.publish(
                TurnInputAbortRequested(
                    session_id=self.session_id,
                    origin="text",
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

            try:
                await self.event_bus.publish(
                    VADSpeechStart(
                        session_id=self.session_id,
                        origin="text",
                    ),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )
                await self.event_bus.publish(
                    ASRResultPartial(
                        session_id=self.session_id,
                        item_id=message_data.get("item_id", ""),
                        text=normalized_text,
                        display_text=normalized_text,
                        speech_pause=True,
                        origin="text",
                    ),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )
            finally:
                await self.event_bus.publish(
                    VADSpeechEnd(
                        session_id=self.session_id,
                        origin="text",
                    ),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )

            await self.event_bus.publish(
                ASRResultFinal(
                    session_id=self.session_id,
                    item_id=message_data.get("item_id", ""),
                    text=normalized_text,
                    display_text=normalized_text,
                    origin="text",
                )
            )

    async def _handle_vad_speech_start(self, message_data: dict) -> None:
        """Handle VAD speech-start signal."""
        await self._publish(VADSpeechStart(session_id=self.session_id, origin="client"))

    async def _handle_vad_speech_end(self, message_data: dict) -> None:
        """Handle VAD speech-end signal after flushing higher-priority listeners."""
        await self.event_bus.publish(
            VADSpeechEnd(
                session_id=self.session_id,
                origin="client",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    async def _handle_tts_playback_finished(self, message_data: dict) -> None:
        """Handle frontend TTS playback completion and transition to idle."""
        await self.event_bus.publish(
            TTSPlaybackFinished(
                session_id=self.session_id,
                response_id=str(message_data.get("response_id", "") or ""),
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    async def _handle_tts_playback_stopped(self, message_data: dict) -> None:
        """Apply the final partial-chunk playback reported during interruption."""
        try:
            played_audio_ms = max(
                0.0,
                float(message_data.get("played_audio_ms", 0.0)),
            )
        except (TypeError, ValueError):
            played_audio_ms = 0.0
        await self.event_bus.publish(
            TTSPlaybackStopped(
                session_id=self.session_id,
                response_id=str(message_data.get("response_id", "") or ""),
                played_audio_ms=played_audio_ms,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

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
        speed = _clamp(float(message_data.get("speed", 1.0)), 0.25, 1.5)
        await self._publish(TTSSpeedChange(session_id=self.session_id, speed=speed))

    async def _handle_change_tts_model(self, message_data: dict) -> None:
        """Handle requests to switch the IndexTTS protocol version."""
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
        await self.event_bus.publish(
            TTSChunkPlayed(
                session_id=self.session_id,
                response_id=str(message_data.get("response_id", "") or ""),
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    async def _handle_session_config(self, message_data: dict) -> None:
        """Handle per-session configuration from client."""
        await self._publish(
            SessionConfigReceived(
                session_id=self.session_id,
                recording_path=message_data.get("recording_path"),
            )
        )

    # ==================== Incoming event handlers ====================

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
        if not isinstance(message_data, dict):
            logger.warning(
                "WebSocket text signal must be a JSON object - session: %s",
                self.session_id,
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
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize WebSocket input gateway.

        Args:
            event_bus: shared event bus instance
            session_id: unique session identifier
            websocket: active WebSocket connection
            config: service configuration shared with the session
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket
        self.config: dict[str, Any] = config or {}
        self.input_sample_rate = _resolve_input_sample_rate(self.config)
        self._pending_text_tasks: set[asyncio.Task[None]] = set()

        self.text_msg_handler = TextMsgHandler(event_bus, session_id, websocket)
        self._handlers = {}
        self._service = None
        self._buffered_bytes = 0
        self._resampler = soxr.ResampleStream(
            24000, self.input_sample_rate, 1, dtype="int16"
        )
        self._items = []
        self._turn_input_bytes = bytearray()
        self._manual_speech_started = False
        self._session_lock = asyncio.Lock()

    async def handle_connection(self, already_accepted: bool = False) -> None:
        """
        Handle a new WebSocket connection.

        Args:
            already_accepted: skip websocket.accept() if caller already accepted.
        """
        if not already_accepted:
            await self.websocket.accept()

    async def handle_message_loop(self) -> None:
        """Main receive loop that dispatches WebSocket messages."""
        try:
            while True:
                data = await self.websocket.receive()
                if data.get("type") == "websocket.disconnect":
                    break
                await self._process_message(data)
        finally:
            await self._wait_for_pending_text_tasks()

    async def _wait_for_pending_text_tasks(self) -> None:
        """Await all in-flight text message tasks before shutting down the loop."""
        if not self._pending_text_tasks:
            return
        pending = tuple(self._pending_text_tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def _process_message(self, data: dict) -> None:
        """Process a raw WebSocket receive payload and publish events."""
        if data["type"] != "websocket.receive":
            return

        if data.get("text") is not None:
            await self._handle_text_message(data["text"])
        elif data.get("bytes") is not None:
            pcm = data["bytes"]
            if len(pcm) > 1024 * 1024:
                await self._error("invalid_value", "Binary message exceeds one MiB")
                return
            try:
                await self._append_pcm(pcm)
            except ValueError as exc:
                await self._error("invalid_value", str(exc))

    async def _handle_text_message(self, text_message: str) -> None:
        """Validate and dispatch a Realtime message.

        Parameters
        ----------
        text_message : str
            JSON client message.
        """
        message = {}
        try:
            if len(text_message.encode()) > (65536 if self.config.get("_end_to_end_profile") else 1024 * 1024):
                raise ValueError("Message exceeds one MiB")
            message = json.loads(text_message)
            if not isinstance(message, dict):
                raise TypeError("Event must be an object")
            message = _CLIENT_EVENT_ADAPTER.validate_python(message)
            kind = message["type"]
            if kind in self._handlers:
                await self._handlers[kind](self, message)
                return
            await self._error(
                "unsupported_event", "Unsupported event type", message.get("event_id")
            )
        except ValidationError as exc:
            error = exc.errors(include_url=False)[0]
            location = ".".join(str(part) for part in error["loc"])
            if message.get("type") == "session.update" and not location.startswith(
                "session"
            ):
                if error["loc"][0] in {"voice", "emotion", "speed"}:
                    location = "audio.output." + location
                elif error["loc"][0] == "threshold":
                    location = "audio.input.turn_detection." + location
                location = "session." + location
            code = (
                "missing_required_parameter"
                if error["type"] == "missing"
                else "invalid_value"
            )
            await self._error(code, error["msg"], message.get("event_id"), location)
        except (
            ValueError,
            TypeError,
            KeyError,
            binascii.Error,
            NotImplementedError,
        ) as exc:
            await self._error(
                "invalid_value",
                str(exc),
                message.get("event_id") if isinstance(message, dict) else None,
            )

        except Exception:
            logger.exception("Realtime request failed - session: %s", self.session_id)
            await self._error(
                "server_error",
                "Request execution failed",
                message.get("event_id"),
                error_type="server_error",
            )

    async def _unsupported_live_input(self, message):
        """Reject controls outside the negotiated single-call capabilities."""
        await self._error("unsupported_capability", "Control is unavailable in single-call Live", message.get("event_id"))

    async def _handle_stream_control(self, message):
        """Validate stream identity and dispatch one Live readiness or control event."""
        profile = self.config.get("_end_to_end_profile")
        if profile is None:
            raise ValueError("Continuous streams require an EndToEnd model")
        kind = message["type"].rsplit(".", 1)[1]
        allowed = {"type", "event_id", "stream_id"}
        if kind == "progress": allowed.add("played_samples")
        if kind == "input_muted": allowed.add("muted")
        if message.keys() - allowed or message.get("stream_id") != profile["stream_id"]:
            raise ValueError("Invalid or stale audio stream control")
        if kind == "ready":
            if not self.config.get("_live_protocol_negotiated"):
                raise ValueError("Rebuild and refresh the frontend: Live protocol version 2 required")
            event = AudioStreamReady(session_id=self.session_id, stream_id=profile["stream_id"])
        else:
            if kind == "input_muted" and type(message.get("muted")) is not bool:
                raise ValueError("muted must be boolean")
            command = {"progress": "audio_stream_progress", "input_muted": "set_input_muted", "close": "close_audio_stream"}[kind]
            event = AudioStreamControlRequested(session_id=self.session_id, stream_id=profile["stream_id"], command=command,
                                                data={key: message[key] for key in ("played_samples", "muted") if key in message})
        await self.event_bus.publish(event, mode=EventDispatchMode.WAIT_UNTIL_COMPLETE)

    async def _handle_audio_append(self, message):
        """Handle the registered input_audio_buffer.append input.

        Parameters
        ----------
        message : dict
            Complete client event received by the current gateway.
        """
        request = _AUDIO_APPEND_ADAPTER.validate_python(message)
        pcm = base64.b64decode(request["audio"], validate=True)
        await self._append_pcm(pcm)

    async def _append_pcm(self, pcm: bytes) -> None:
        """Validate and append audio through the shared input pipeline.

        Parameters
        ----------
        pcm : bytes
            Mono signed little-endian PCM16 audio sampled at 24 kHz.

        Raises
        ------
        ValueError
            If the byte count is odd or the shared buffer exceeds 60 seconds.
        """
        if len(pcm) % 2 or self._buffered_bytes + len(pcm) > 24000 * 2 * 60:
            raise ValueError("Invalid PCM or audio buffer exceeds 60 seconds")
        if self.config.get("_end_to_end_profile"):
            await self.event_bus.publish(
                AudioFrameReceived(session_id=self.session_id, audio_data=pcm,
                                   sample_rate=self.input_sample_rate),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )
            return
        self._buffered_bytes += len(pcm)
        converted = self._resampler.resample_chunk(
            np.frombuffer(pcm, dtype="<i2")
        ).tobytes()
        if converted:
            if self.config.get("realtime_turn_detection", True) is None and (
                not self._manual_speech_started
            ):
                self._turn_input_bytes.extend(converted)
            else:
                await self._handle_audio_message(converted)

    async def _handle_audio_commit(self, message):
        """Handle the registered input_audio_buffer.commit input.

        Parameters
        ----------
        message : dict
            Complete client event received by the current gateway.
        """
        _AUDIO_CONTROL_ADAPTER.validate_python(message)
        if not self._buffered_bytes:
            raise ValueError("Cannot commit an empty buffer")
        if self.config.get("realtime_turn_detection", True) is None:
            await self._begin_manual_turn(message)
        self._buffered_bytes = 0
        await self.text_msg_handler._handle_vad_speech_end(message)
        self._manual_speech_started = False

    async def _handle_audio_clear(self, message):
        """Handle the registered input_audio_buffer.clear input.

        Parameters
        ----------
        message : dict
            Complete client event received by the current gateway.
        """
        _AUDIO_CONTROL_ADAPTER.validate_python(message)
        await self.event_bus.publish(
            TurnInputAbortRequested(session_id=self.session_id, origin="client"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        self._buffered_bytes = 0
        self._turn_input_bytes.clear()
        self._manual_speech_started = False
        self._resampler = soxr.ResampleStream(
            24000, self.input_sample_rate, 1, dtype="int16"
        )
        await self._emit("input_audio_buffer.cleared")

    async def _handle_speech_start(self, message):
        """Handle the registered input_audio_buffer.speech_start input.

        Parameters
        ----------
        message : dict
            Complete client event received by the current gateway.
        """
        _SPEECH_START_ADAPTER.validate_python(message)
        if self.config.get("realtime_turn_detection", True) is None:
            await self._begin_manual_turn(message)
        else:
            await self.text_msg_handler._handle_vad_speech_start(message)

    async def _handle_item_create(self, message):
        """Handle the registered conversation.item.create input.

        Parameters
        ----------
        message : dict
            Complete client event received by the current gateway.
        """
        request = _ITEM_CREATE_ADAPTER.validate_python(message)
        item = request["item"]
        item = dict(item, id=item.get("id") or "item_" + uuid4().hex)
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError("Item ID must be nonempty text")
        if any(existing["id"] == item["id"] for existing in self._items):
            raise ValueError("Duplicate item ID")
        text = "".join(p["text"] for p in item["content"])
        generate = request.get("xtalk", {}).get("generate_response", False)
        if generate and item["role"] != "user":
            raise ValueError("Only user input can trigger generation")
        if generate and (
            not text.strip() or len(text.strip().encode("utf-8")) > MAX_TEXT_INPUT_BYTES
        ):
            raise ValueError("Real-time text must be nonblank and at most 8192 bytes")
        if not generate:
            await self.event_bus.request(
                AgentContextAppendRequested(
                    session_id=self.session_id,
                    role=item["role"],
                    content=text,
                )
            )
        self._items.append(item)
        await self._emit("conversation.item.added", item=item)
        await self._emit("conversation.item.done", item=item)
        if generate:
            self._schedule_text_submission(text, item_id=item["id"])

    async def _handle_response_cancel(self, message):
        """Handle the registered response.cancel input.

        Parameters
        ----------
        message : dict
            Complete client event received by the current gateway.
        """
        await self.event_bus.request(
            ResponseCancelRequested(
                session_id=self.session_id,
                **{
                    key: value
                    for key, value in message.items()
                    if key not in {"type", "event_id"}
                },
            )
        )

    async def _handle_playback_ack(self, message):
        """Handle the registered response.output_audio.playback.chunk_played, response.output_audio.playback.finished, response.output_audio.playback.stopped input.

        Parameters
        ----------
        message : dict
            Complete client event received by the current gateway.
        """
        kind = message["type"]
        message = (
            _PLAYBACK_STOPPED_ADAPTER
            if kind.endswith(".stopped")
            else _PLAYBACK_ADAPTER
        ).validate_python(message)
        event_type = {
            "chunk_played": TTSChunkPlayed,
            "finished": TTSPlaybackFinished,
            "stopped": TTSPlaybackStopped,
        }[kind.rsplit(".", 1)[1]]
        fields = {"response_id": message["response_id"]}
        if kind.endswith(".stopped"):
            fields["played_audio_ms"] = message["played_audio_ms"]
        await self.event_bus.publish(
            event_type(session_id=self.session_id, **fields),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        self._service.output_gateway.settle_audio_playback(
            message["response_id"], complete=not kind.endswith(".chunk_played")
        )

    async def _begin_manual_turn(self, message):
        """Start manual ASR once and replay the uncommitted input buffer."""
        if self._manual_speech_started:
            return
        self._manual_speech_started = True
        await self.event_bus.publish(
            VADSpeechStart(session_id=self.session_id, origin="client"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        audio = bytes(self._turn_input_bytes)
        self._turn_input_bytes.clear()
        if audio:
            await self.event_bus.publish(
                AudioFrameReceived(
                    session_id=self.session_id,
                    audio_data=audio,
                    sample_rate=self.input_sample_rate,
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

    def _schedule_text_submission(self, text: str, *, item_id: str = "") -> None:
        """Schedule a serialized real-time user text turn."""
        task = asyncio.create_task(
            self.text_msg_handler._handle_submit_text(
                {"text": text, "item_id": item_id}
            )
        )
        self._pending_text_tasks.add(task)
        task.add_done_callback(self._pending_text_tasks.discard)

    async def _emit(self, kind, **data):
        """Send a protocol event with a fresh server event identifier."""
        await self.websocket.send_json(
            {"type": kind, "event_id": "evt_" + uuid4().hex, **data}
        )

    async def _error(
        self,
        code,
        message,
        event_id=None,
        param=None,
        *,
        error_type="invalid_request_error",
    ):
        """Report a client error correlated with its optional event identifier."""
        await self._emit(
            "error",
            error={
                "type": error_type,
                "code": code,
                "message": message,
                "param": param,
                "event_id": event_id if isinstance(event_id, str) else None,
            },
        )

    async def _update_session(self, message: dict[str, Any]) -> None:
        """Merge, validate, and coordinate a partial session configuration update."""
        request = _SESSION_UPDATE_ADAPTER.validate_python(message)
        async with self._session_lock, self._service.output_gateway.audio_encoding_lock:
            await self._apply_session_update(request["session"])

    async def _apply_session_update(self, update: dict[str, Any]) -> None:
        """Validate all settings before dispatching single-owner manager requests."""
        current = self._service.output_gateway.session_snapshot()
        candidate = copy.deepcopy(current)

        def merge(target, patch):
            """Merge a partial configuration without modifying live state."""
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value

        if self.config.get("_end_to_end_profile"):
            if update.get("xtalk", {}).get("end_to_end_protocol") != 2:
                raise ValueError("Rebuild and refresh the frontend: Live protocol version 2 required")
            merge(candidate, update)
            candidate = _SESSION_ADAPTER.validate_python(candidate)
            comparable = copy.deepcopy(candidate)
            comparable["xtalk"] = current["xtalk"]
            if comparable != current or candidate["xtalk"].get("end_to_end") != current["xtalk"]["end_to_end"]:
                raise ValueError("Live media, instructions, voice and capabilities are fixed for this call")
            if self.config.get("_live_protocol_negotiated") and candidate["xtalk"]["output_audio_encoding"] != current["xtalk"]["output_audio_encoding"]:
                raise ValueError("Cannot change audio encoding during a Live call")
            self._service.output_gateway.set_session_configuration(candidate)
            self.config["_live_protocol_negotiated"] = True
            await self._emit("session.updated", session=candidate)
            return
        for direction in ("input", "output"):
            if update.get("audio", {}).get(direction, {}).get("format", {}).get("rate", 24000) != 24000:
                raise ValueError("Cascade Realtime audio requires 24000 Hz")
        # Re-enabling VAD restores the documented defaults before merging.
        detection_patch = update.get("audio", {}).get("input", {}).get("turn_detection")
        if candidate["audio"]["input"]["turn_detection"] is None and isinstance(
            detection_patch, dict
        ):
            candidate["audio"]["input"]["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": True,
                "interrupt_response": True,
            }
        merge(candidate, update)
        candidate = _SESSION_ADAPTER.validate_python(candidate)
        if (
            candidate["xtalk"]["output_audio_encoding"]
            != current["xtalk"]["output_audio_encoding"]
            and self._service.output_gateway.has_unsettled_output()
        ):
            raise ValueError("Cannot change audio encoding during response or playback")
        transcription = candidate["audio"]["input"]["transcription"]
        if transcription is not None and transcription["model"] != current["model"]:
            raise ValueError(
                "Transcription supports only the configured model alias or null"
            )
        operations = []
        detection = candidate["audio"]["input"]["turn_detection"]
        previous_detection = current["audio"]["input"]["turn_detection"]
        if detection is not None and (
            previous_detection is None
            or detection["threshold"] != previous_detection["threshold"]
        ):
            operations.append(
                (
                    VADThresholdUpdateRequested(
                        session_id=self.session_id, threshold=detection["threshold"]
                    ),
                    ("audio", "input", "turn_detection"),
                    detection,
                )
            )
        if candidate["instructions"] != current["instructions"]:
            operations.append(
                (
                    AgentInstructionsUpdateRequested(
                        session_id=self.session_id,
                        instructions=candidate["instructions"],
                    ),
                    ("instructions",),
                    candidate["instructions"],
                )
            )
        output = candidate["audio"]["output"]
        output_patch = update.get("audio", {}).get("output", {})
        for setting in ("voice", "emotion", "speed"):
            if setting not in output_patch:
                continue
            value = output[setting]
            event = TTSConfigurationUpdateRequested(
                session_id=self.session_id, **{setting: value}
            )
            if value != current["audio"]["output"][setting]:
                operations.append((event, ("audio", "output", setting), value))
        for event, _, _ in operations:
            event.check_only = True
            await self.event_bus.request(event)
        for event, path, value in operations:
            event.check_only = False
            await self.event_bus.request(event)
            target = current
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            self._store_session(current)
        self._store_session(candidate)
        await self._emit("session.updated", session=candidate)

    def _store_session(self, session: RealtimeSession) -> None:
        """Commit settings confirmed by managers and synchronize runtime options."""
        self._service.output_gateway.set_session_configuration(session)
        self.config["output_modalities"] = session["output_modalities"]
        self.config["realtime_turn_detection"] = session["audio"]["input"][
            "turn_detection"
        ]

    async def _handle_audio_message(self, audio_data: bytes) -> None:
        """Publish raw audio frames; enhancer logic runs in EnhancerManager."""
        await self.event_bus.publish(
            AudioFrameReceived(
                session_id=self.session_id,
                audio_data=audio_data,
                sample_rate=self.input_sample_rate,
            )
        )

    @EventListenerMixin.event_handler(ASRResultFinal, priority=-100)
    async def _reset_committed_buffer(self, event: ASRResultFinal) -> None:
        """Release the committed input byte budget after an audio turn."""
        if event.origin != "text":
            self._buffered_bytes = 0

    @EventListenerMixin.event_handler(ErrorOccurred, priority=10)
    async def _handle_error_event(self, event: ErrorOccurred) -> None:
        """Handle backend error events."""
        logger.error(
            "Error event received - session: %s, type: %s, message: %s",
            self.session_id,
            event.error_type,
            event.error_message,
        )
