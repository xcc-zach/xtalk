import asyncio
import base64
import copy
import json
import logging
from inspect import isawaitable
from typing import Any
from uuid import uuid4

import numpy as np
import soxr

from ...models import Models
from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    ASRFailed,
    ASRResultFinal,
    ASRResultPartial,
    AudioStreamControlRequested,
    BinaryAudioHeader,
    CaptionUpdated,
    EndToEndAudioChunkReady,
    EndToEndSignal,
    EndToEndTranscriptDelta,
    ErrorOccurred,
    FullAudioFrameReady,
    LatencyMetricsUpdated,
    RealtimeSession,
    ResponseCancelRequested,
    ResponseFinish,
    ResponseUpdate,
    RetrievalUpdated,
    SpeakerRecognized,
    ToolCallOccurred,
    TTSChunkReady,
    TTSEmotionChange,
    TTSFinished,
    TTSStarted,
    TTSStopped,
    TTSVoiceChange,
    TurnASREndRequested,
    TurnLLMAgentStopRequested,
    VADSpeechEnd,
    VADSpeechStart,
)
from ..interfaces import Connection as WebSocket
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
        """Initialize session-scoped transport and protocol state."""
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket
        self.config: dict[str, Any] = config or {}
        self._speaker_history_gate_enabled = (
            models is not None and speaker_history_gate_enabled(models, self.config)
        )
        self.audio_encoding_lock = asyncio.Lock()
        self._full_audio_chunk_bytes = 128 * 1024
        self._active_response = None
        self._input_items = set()
        self._committed_input_items = set()
        self._responses = {}
        self._failed_responses = set()
        self._cancelling_responses: set[str] = set()
        self._output_subscriptions = {}
        self._session = {
            "id": session_id,
            "object": "realtime.session",
            "type": "realtime",
            "model": self.config.get("realtime_api", {}).get("model", "xtalk"),
            "instructions": "",
            "xtalk": {"output_audio_encoding": "base64"},
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": self.config.get("realtime_api", {}).get(
                            "model", "xtalk"
                        )
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "default",
                    "speed": 1.0,
                    "emotion": {"name": "", "vector": []},
                },
            },
        }

        self._stream_transport_failed = False
        profile = self.config.get("_end_to_end_profile")
        if profile:
            self._session["xtalk"]["end_to_end"] = profile
            self._session["audio"]["input"]["format"]["rate"] = profile["media"]["sample_rate"]
            self._session["audio"]["output"]["format"]["rate"] = profile["media"]["sample_rate"]
            self._session["audio"]["input"]["turn_detection"] = None
            self._session["audio"]["output"]["voice"] = profile["metadata"].get("voice", "default")

    async def _report_stream_failure(self, stream_id: str) -> None:
        """Notify the call owner once when a Realtime write fails."""
        if self._stream_transport_failed: return
        self._stream_transport_failed = True
        await self.event_bus.publish(AudioStreamControlRequested(
            session_id=self.session_id, stream_id=stream_id, command="_transport_failure"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE)

    async def _stream_signal(self, event: EndToEndSignal) -> None:
        """Send a normalized Live control through the ordered Realtime transport."""
        actions = {"audio_stream_start": "xtalk.audio_stream.started", "audio_stream_end": "xtalk.audio_stream.ended",
                   "end_to_end_error": "xtalk.audio_stream.error"}
        try:
            await self.send_signal({"type": actions[event.action], **event.data})
        except Exception:
            await self._report_stream_failure(event.data["stream_id"])
            raise

    async def _stream_audio(self, event: EndToEndAudioChunkReady) -> None:
        """Deliver continuous PCM using the negotiated JSON or binary envelope."""
        try:
            if self._session["xtalk"]["output_audio_encoding"] == "base64":
                await self.send_signal({"type": "xtalk.audio_stream.delta", "stream_id": event.stream_id,
                                        "delta": base64.b64encode(event.audio_data).decode("ascii")})
            else:
                header = json.dumps({"version": 1, "type": "xtalk.audio_stream.delta", "event_id": "evt_" + uuid4().hex,
                                     "stream_id": event.stream_id}, separators=(",", ":")).encode()
                await self.websocket.send_bytes(len(header).to_bytes(4, "big") + header + event.audio_data)
        except Exception:
            await self._report_stream_failure(event.stream_id)
            raise

    async def _stream_transcript(self, event: EndToEndTranscriptDelta) -> None:
        """Preserve independent transcript fragments in the Realtime extension."""
        try:
            await self.send_signal({"type": "xtalk.transcript.delta", **{
                key: value for key, value in vars(event).items() if key not in {"session_id", "timestamp"}}})
        except Exception:
            await self._report_stream_failure(event.stream_id)
            raise

    # ── WebSocket helpers ───────────────────────────────────────────

    def session_snapshot(self) -> RealtimeSession:
        """Return an independent snapshot of the effective session configuration."""
        return copy.deepcopy(self._session)

    def set_session_configuration(self, session: RealtimeSession) -> None:
        """Store a configuration confirmed by the input gateway."""
        self._session = copy.deepcopy(session)

    def has_unsettled_output(self) -> bool:
        """Return whether generation or delivered audio prevents encoding changes."""
        return any(
            not state["done"] or state.get("pending_audio", 0) > 0
            for state in self._responses.values()
        )

    def settle_audio_playback(self, response_id: str, *, complete: bool) -> None:
        """Release delivered audio blocks after client playback settlement.

        Parameters
        ----------
        response_id : str
            Response acknowledged by the client.
        complete : bool
            Whether all remaining blocks have finished or stopped.
        """
        state = self._responses.get(response_id)
        if state is not None:
            state["pending_audio"] = (
                0 if complete else max(0, state.get("pending_audio", 0) - 1)
            )

    async def _send_audio(self, kind: str, pcm: bytes, metadata: dict) -> None:
        """Encode one PCM block using the negotiated mutually exclusive transport."""
        if self._session["xtalk"]["output_audio_encoding"] == "base64":
            await self.send_signal({
                "type": kind, **metadata, "delta": base64.b64encode(pcm).decode("ascii")
            })
            return
        header: BinaryAudioHeader = {
            "version": 1,
            "type": "xtalk.output_audio.delta" if kind == "response.output_audio.delta"
            else "xtalk.audio.mix.delta",
            "event_id": "evt_" + uuid4().hex,
        }
        encoded = json.dumps({**header, **metadata}, separators=(",", ":")).encode("utf-8")
        if not 1 <= len(encoded) <= 65536:
            raise ValueError("Binary audio header exceeds protocol limit")
        await self.websocket.send_bytes(len(encoded).to_bytes(4, "big") + encoded + pcm)

    async def send_signal(self, message: dict) -> None:
        """Send a complete external event through the ordered connection queue.

        Parameters
        ----------
        message : dict
            Event with a type and optional event ID.
        """
        if "event_id" in message and (
            not isinstance(message["event_id"], str) or not message["event_id"]
        ):
            raise ValueError("Outbound event_id must be nonempty text")
        if not isinstance(message.get("type"), str):
            raise TypeError("Outbound event requires type")
        await self.websocket.send_json({"event_id": "evt_" + uuid4().hex, **message})

    async def _forward(self, action, data):
        """Convert a legacy internal signal into its Realtime event envelope."""
        names = {
            "update_resp": "response.output_audio.playback.updated",
            "finish_resp": "response.output_audio.playback.done",
            "start_tts": "response.content_part.added",
            "pause_tts": "response.output_audio.playback.pause",
            "resume_tts": "response.output_audio.playback.resume",
            "stop_tts": "response.output_audio.playback.stop",
            "speaker_updated": "conversation.speaker.updated",
            "caption_updated": "conversation.caption.updated",
            "retrieval_updated": "conversation.retrieval.updated",
            "latency_metrics": "session.latency.updated",
            "tool_called": "response.tool.called",
            "full_audio_frame": "session.audio.mix.delta",
        }
        if action == "voice_changed" or action == "emotion_changed":
            return
        if action == "error":
            await self.send_signal(
                {
                    "type": "error",
                    "error": {
                        "type": "server_error",
                        "code": "generation_error",
                        "message": str(data),
                    },
                }
            )
            return
        payload = dict(data) if isinstance(data, dict) else {}
        if action == "full_audio_frame":
            payload["delta"] = payload.pop("audio_base64")
        if action in {"vad_speech_start", "vad_speech_end"}:
            kind = (
                "input_audio_buffer.speech_started"
                if action.endswith("start")
                else "input_audio_buffer.speech_stopped"
            )
        else:
            kind = names.get(action, action)
        await self.send_signal({"type": kind, **payload})

    async def _send_session_created(self):
        """Announce a ready temporary Realtime session."""
        try:
            await self.send_signal({"type": "session.created", "session": self._session})
        except Exception:
            profile = self.config.get("_end_to_end_profile")
            if profile is None: raise
            await self._report_stream_failure(profile["stream_id"])

    @staticmethod
    def _map_playback(event):
        """Build a playback text snapshot from the existing settlement event."""
        return {"response_id": event.response_id, "text": event.text}

    @staticmethod
    def _map_speaker(event):
        """Build the original speaker identity notification."""
        return {"speaker_id": event.speaker_id, "reason": event.reason}

    @staticmethod
    def _map_caption(event):
        """Build a complete caption snapshot."""
        return {
            "text": event.text or "",
            "is_final": bool(event.is_final),
            "reason": event.reason,
        }

    @staticmethod
    def _map_retrieval(event):
        """Build a complete retrieval snapshot."""
        return {"text": event.text or "", "is_final": bool(event.is_final)}

    @staticmethod
    def _map_latency(event):
        """Preserve the five original latency metrics in milliseconds."""
        return {
            "network_latency_ms": int(event.network_latency_ms),
            "asr_latency_ms": int(event.asr_latency_ms),
            "llm_first_token_ms": int(event.llm_first_token_ms),
            "llm_sentence_ms": int(event.llm_sentence_ms),
            "tts_first_chunk_ms": int(event.tts_first_chunk_ms),
        }

    @staticmethod
    def _map_tool(event):
        """Expose tool calls while retaining internal result filtering."""
        if event.name in {"tool_call_result", "direct_audio"}:
            return None
        return {"name": event.name, "args": event.args or {}}

    def _register_custom_outputs(self, registrations):
        """Install output mappings, replacing existing subscriptions for the same key."""
        for (source, kind), handler in registrations.items():
            previous = self._output_subscriptions.get((source, kind))
            if previous is not None:
                self.event_bus.unsubscribe(source, previous)

            async def forward(event, _handler=handler, _kind=kind):
                """Build and send one registered output mapping payload."""
                payload = _handler(event)
                if isawaitable(payload):
                    payload = await payload
                if payload is not None:
                    if set(payload) & {"type", "event_id"}:
                        raise ValueError("Output payload cannot override type/event_id")
                    await self.send_signal({"type": _kind, **payload})

            self.event_bus.subscribe(source, forward)
            self._output_subscriptions[(source, kind)] = forward

    @EventListenerMixin.event_handler(ResponseCancelRequested)
    async def _cancel_response(self, event: ResponseCancelRequested) -> None:
        """Resolve a response and coordinate stop before emitting its terminal state."""
        response_id = event.response_id or self._active_response
        state = self._responses.get(response_id)
        if state is None or state["done"]:
            raise ValueError("No active response")
        self._cancelling_responses.add(response_id)
        try:
            await self.event_bus.request(
                TurnLLMAgentStopRequested(
                    session_id=self.session_id, response_id=response_id, reason="client_cancel"
                )
            )
            await self._finish_response(response_id, "cancelled")
        finally:
            self._cancelling_responses.discard(response_id)

    async def _ensure_response(self, response_id):
        """Announce a response and its content once before emitting deltas."""
        if response_id not in self._responses:
            self._responses[response_id] = {
                "text": "",
                "transcript": "",
                "done": False,
                "text_done": True,
                "has_text_source": False,
                "audio_done": False,
                "modalities": list(self._session["output_modalities"]),
            }
            self._active_response = response_id
            item = {
                "id": "item_" + response_id,
                "type": "message",
                "role": "assistant",
                "content": [],
            }
            await self.send_signal(
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "status": "in_progress",
                        "output": [],
                    },
                }
            )
            await self.send_signal(
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": item,
                }
            )
            await self.send_signal(
                {
                    "type": "response.content_part.added",
                    **self._identity(response_id),
                    "part": {
                        "type": "audio"
                        if self._session["output_modalities"] == ["audio"]
                        else "text"
                    },
                }
            )
        return self._responses[response_id]

    @staticmethod
    def _identity(response_id):
        """Build stable output content identifiers for one response."""
        return {
            "response_id": response_id,
            "item_id": "item_" + response_id,
            "output_index": 0,
            "content_index": 0,
        }

    async def _on_generated_text(self, event):
        """Forward a generated text suffix independently from played text."""
        state = await self._ensure_response(event.response_id)
        if state["done"] or (state["has_text_source"] and state["text_done"]):
            return
        state["has_text_source"] = True
        state["text_done"] = False
        delta = event.delta
        state["text"] = event.text
        if state["modalities"] == ["text"]:
            await self.send_signal(
                {
                    "type": "response.output_text.delta",
                    **self._identity(event.response_id),
                    "delta": delta,
                }
            )

    async def _on_generated_done(self, event):
        """Finish the generated text stream for text-only responses."""
        state = await self._ensure_response(event.response_id)
        if state["done"] or (state["has_text_source"] and state["text_done"]):
            return
        state["has_text_source"] = True
        state["text_done"] = True
        state["text"] = event.text
        if state["modalities"] == ["text"]:
            await self.send_signal(
                {
                    "type": "response.output_text.done",
                    **self._identity(event.response_id),
                    "text": event.text,
                }
            )
        if state["modalities"] == ["text"] or state["audio_done"]:
            await self._finish_response(
                event.response_id,
                "failed"
                if event.response_id in self._failed_responses
                else "completed",
            )

    async def _finish_response(self, response_id, status="completed"):
        """Close the generation lifecycle exactly once for a response."""
        state = await self._ensure_response(response_id)
        if state["done"]:
            return
        state["done"] = True
        if response_id in self._cancelling_responses:
            status = "cancelled"
            self._cancelling_responses.discard(response_id)
        part = (
            {"type": "audio", "transcript": state["transcript"]}
            if state["modalities"] == ["audio"]
            else {"type": "text", "text": state["text"]}
        )
        await self.send_signal(
            {
                "type": "response.content_part.done",
                **self._identity(response_id),
                "part": part,
            }
        )
        item = {
            "id": "item_" + response_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [part],
        }
        await self.send_signal(
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": item,
            }
        )
        await self.send_signal(
            {
                "type": "response.done",
                "response": {"id": response_id, "status": status, "output": [item]},
            }
        )

    async def _on_sentence(self, event):
        """Publish transcript text only after successful full segment delivery."""
        if event.succeeded:
            await self._transcript(event.response_id, event.text)

    async def _transcript(self, response_id, text):
        """Append synthesized transcript text to its response stream."""
        state = await self._ensure_response(response_id)
        if state["done"]:
            return
        state["transcript"] += text
        await self.send_signal(
            {
                "type": "response.output_audio_transcript.delta",
                **self._identity(response_id),
                "delta": text,
            }
        )

    # ── Shared helpers ──────────────────────────────────────────────

    async def _commit_input_item(self, item_id: str) -> None:
        """Acknowledge an acoustic commit before waiting for transcription.

        Parameters
        ----------
        item_id : str
            Stable input identifier assigned by the existing turn manager.
        """
        if item_id in self._committed_input_items:
            return
        self._committed_input_items.add(item_id)
        await self.send_signal(
            {"type": "input_audio_buffer.committed", "item_id": item_id}
        )
        if item_id not in self._input_items:
            self._input_items.add(item_id)
            await self.send_signal(
                {
                    "type": "conversation.item.added",
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_audio"}],
                    },
                }
            )

    async def _on_input_commit(self, event: TurnASREndRequested) -> None:
        """Expose the hard turn boundary before ASR finalization begins.

        Parameters
        ----------
        event : TurnASREndRequested
            Existing manager-owned acoustic turn boundary.
        """
        await self._commit_input_item(event.item_id)

    async def _forward_asr(
        self,
        action: str,
        event: ASRResultPartial | ASRResultFinal,
    ) -> None:
        """Forward an ASR result (partial or final) to the frontend."""
        if event.origin == "text":
            return
        item_id = event.item_id
        if item_id not in self._input_items:
            self._input_items.add(item_id)
            await self.send_signal(
                {
                    "type": "conversation.item.added",
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_audio"}],
                    },
                }
            )
        transcription_enabled = (
            self._session["audio"]["input"]["transcription"] is not None
        )
        if action == "update_asr":
            if not transcription_enabled:
                return
            await self.send_signal(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": item_id,
                    "content_index": 0,
                    "delta": event.delta,
                    "xtalk": {"mode": event.update_mode, "revision": event.revision},
                }
            )
        else:
            await self._commit_input_item(item_id)
            await self.send_signal(
                {
                    "type": "conversation.item.done",
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "transcript": (event.display_text or event.text)
                                if transcription_enabled
                                else None,
                            }
                        ],
                    },
                }
            )
            if not transcription_enabled:
                return
            await self.send_signal(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": event.display_text or event.text,
                }
            )

    # ── Session ─────────────────────────────────────────────────────

    # ── Event handlers ──────────────────────────────────────────────

    async def _send_gated_update_asr_signal(
        self,
        event: ASRResultPartial,
    ) -> None:
        """Forward a focus-gated preview before the history barrier."""

        await self._forward_asr("update_asr", event)

    async def _send_update_asr_signal(self, event: ASRResultPartial) -> None:
        """Forward the current ASR preview revision."""
        await self._forward_asr("update_asr", event)

    async def _send_asr_failure(self, event: ASRFailed) -> None:
        """Expose a failed transcription without terminating its connection."""
        if self._session["audio"]["input"]["transcription"] is not None:
            await self.send_signal(
                {
                    "type": "conversation.item.input_audio_transcription.failed",
                    "item_id": event.item_id,
                    "content_index": 0,
                    "error": {
                        "type": "server_error",
                        "code": "asr_error",
                        "message": event.message,
                    },
                }
            )

    async def _send_finish_asr_signal(self, event: ASRResultFinal) -> None:
        """Forward final transcription and complete its input item."""
        await self._forward_asr("finish_asr", event)

    async def _on_start_tts(self, event: TTSStarted) -> None:
        """Ensure output identity exists when synthesis starts."""
        await self._ensure_response(event.response_id)

    async def _send_stop_tts_signal(self, event: TTSStopped) -> None:
        """Notify playback to stop and finalize generation as cancelled."""
        await self._forward("stop_tts", {"response_id": event.response_id})
        await self._finish_response(event.response_id, "cancelled")

    async def _send_error_signal(self, event: ErrorOccurred) -> None:
        """Expose a runtime failure through a structured server error."""
        if event.response_id:
            self._failed_responses.add(event.response_id)
        await self._forward("error", event.error_message)

    async def _send_pause_tts_signal(self, event) -> None:
        """Forward the existing playback pause notification."""
        await self._forward("pause_tts", {"response_id": event.response_id})

    async def _send_resume_tts_signal(self, event) -> None:
        """Forward the existing playback resume notification."""
        await self._forward("resume_tts", {"response_id": event.response_id})

    async def _send_update_resp_signal(self, event: ResponseUpdate) -> None:
        """Forward a complete playback-confirmed text snapshot."""
        await self._forward(
            "update_resp",
            {"response_id": event.response_id, "text": event.text},
        )

    async def _send_finish_resp_signal(self, event: ResponseFinish) -> None:
        """Forward the final playback-confirmed text snapshot."""
        await self._forward(
            "finish_resp",
            {"response_id": event.response_id, "text": event.text},
        )

    async def _send_tts_finished_signal(self, event: TTSFinished) -> None:
        """Finish audio and transcript generation once synthesis delivery ends."""
        state = await self._ensure_response(event.response_id)
        if not state["done"] and not state["audio_done"]:
            state["audio_done"] = True
            await self.send_signal(
                {
                    "type": "response.output_audio.done",
                    **self._identity(event.response_id),
                }
            )
            await self.send_signal(
                {
                    "type": "response.output_audio_transcript.done",
                    **self._identity(event.response_id),
                    "transcript": state["transcript"],
                }
            )
        if not state["done"] and state["audio_done"] and state["text_done"]:
            await self._finish_response(
                event.response_id,
                "failed"
                if event.response_id in self._failed_responses
                else "completed",
            )

    async def _send_speaker_updated_signal(self, event: SpeakerRecognized) -> None:
        """Forward the original speaker identity and reason."""
        await self._forward(
            "speaker_updated",
            {
                "speaker_id": event.speaker_id,
                "reason": event.reason,
            },
        )

    async def _send_caption_updated(self, event: CaptionUpdated) -> None:
        """Forward a complete caption snapshot."""
        await self._forward(
            "caption_updated",
            {
                "text": event.text or "",
                "is_final": bool(event.is_final),
                "reason": event.reason,
            },
        )

    async def _send_retrieval_updated(self, event: RetrievalUpdated) -> None:
        """Forward a complete retrieval result snapshot."""
        await self._forward(
            "retrieval_updated",
            {
                "text": event.text or "",
                "is_final": bool(event.is_final),
            },
        )

    async def _send_tts_chunk_signal(self, event: TTSChunkReady) -> None:
        """Deliver one original audio block in the negotiated output encoding."""
        async with self.audio_encoding_lock:
            state = await self._ensure_response(event.response_id)
            if state["done"] or state["modalities"] != ["audio"]:
                return
            pcm = np.frombuffer(event.audio_chunk, dtype="<i2")
            if event.sample_rate != 24000:
                pcm = soxr.resample(pcm, event.sample_rate, 24000).astype("<i2")
            state["pending_audio"] = state.get("pending_audio", 0) + 1
            await self._send_audio(
                "response.output_audio.delta", pcm.tobytes(), self._identity(event.response_id)
            )

    async def _send_full_audio_frame_signal(self, event: FullAudioFrameReady) -> None:
        """Forward mixed PCM audio in bounded messages without duplicate payloads."""
        async with self.audio_encoding_lock:
            if not event.audio_chunk:
                return
            if self._session["xtalk"]["output_audio_encoding"] == "binary_pcm" and (
                event.format != "pcm_s16le" or event.channels < 1
                or len(event.audio_chunk) % (2 * event.channels)
            ):
                raise ValueError("Binary mixed audio requires aligned PCM16")
            block_bytes = max(2 * event.channels, self._full_audio_chunk_bytes // (2 * event.channels) * (2 * event.channels))
            for start in range(0, len(event.audio_chunk), block_bytes):
                chunk = event.audio_chunk[start : start + block_bytes]
                await self._send_audio("session.audio.mix.delta", chunk, {
                    "sample_rate": int(event.sample_rate),
                    "channels": int(event.channels), "format": event.format,
                })

    async def _send_voice_changed_signal(self, event: TTSVoiceChange) -> None:
        """Preserve internal voice change handling without a legacy network event."""
        await self._forward("voice_changed", {"name": event.voice_name})

    async def _send_vad_start_signal(self, event: VADSpeechStart) -> None:
        """Forward existing speech-start notifications with their origin."""
        if event.origin == "client":
            return
        await self._forward("vad_speech_start", {"origin": event.origin})

    async def _send_vad_end_signal(self, event: VADSpeechEnd) -> None:
        """Forward existing speech-end notifications with their origin."""
        if event.origin == "client":
            return
        await self._forward("vad_speech_end", {"origin": event.origin})

    async def _send_emotion_changed_signal(self, event: TTSEmotionChange) -> None:
        """Preserve internal emotion handling without a legacy network event."""
        await self._forward(
            "emotion_changed",
            {
                "name": event.emotion_name,
                "vector": event.emotion_vector or [],
            },
        )

    async def _send_latency_metrics_signal(self, event: LatencyMetricsUpdated) -> None:
        """Forward the original five latency metrics."""
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

    async def _send_tool_called_signal(self, event: ToolCallOccurred) -> None:
        """Forward public tool calls while filtering internal result messages."""
        if event.name in {"tool_call_result", "direct_audio"}:
            return
        await self._forward(
            "tool_called",
            {
                "name": event.name,
                "args": event.args or {},
            },
        )
