# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass, field, make_dataclass
from typing import AsyncIterator, ClassVar, Dict, Any, Type
from ..models.agents.interfaces import AgentOutput


@dataclass
class Event:
    """Base dataclass for all Xtalk events.

    Parameters
    ----------
    session_id : str
        Session identifier associated with the event.

    Attributes
    ----------
    timestamp : float
        Unix timestamp recorded when the event instance is created.
    session_id : str
        Session identifier associated with the event.
    TYPE : str
        Stable event type string used by the event bus.
    """

    timestamp: float = field(init=False)
    session_id: str
    TYPE: ClassVar[str] = "base"

    def __post_init__(self):
        self.timestamp = time.time()

    @property
    def event_type(self) -> str:
        return self.TYPE


def create_event_class(
    *, name: str, fields: dict[str, Any] | None = None, type_name: str | None = None
) -> Type[Event]:
    """Create an ``Event`` subclass dynamically.

    Parameters
    ----------
    name : str
        Dataclass name for the generated event type.
    fields : dict[str, Any] | None, optional
        Mapping of field names to default values. Value types are inferred from
        the defaults.
    type_name : str | None, optional
        Event bus type string. Defaults to ``name.lower()`` when omitted.

    Returns
    -------
    Type[Event]
        Generated dataclass type inheriting from ``Event``.

    Examples
    --------
    >>> CustomEvent = create_event_class(
    ...     name="CustomEvent",
    ...     fields={"text": ""},
    ... )
    """
    fields = fields or {}
    dataclass_fields = []
    for key, default in fields.items():
        dataclass_fields.append((key, type(default), field(default=default)))
    type_name = type_name or name.lower()
    return make_dataclass(
        name, dataclass_fields, bases=(Event,), namespace={"TYPE": type_name}
    )


@dataclass
class WebSocketMessageReceived(Event):
    TYPE: ClassVar[str] = "websocket.message_received"
    message: str = ""


@dataclass
class AudioFrameReceived(Event):
    TYPE: ClassVar[str] = "audio.frame_received"
    audio_data: bytes
    sample_rate: int = 16000


@dataclass
class EnhancedAudioFrameReceived(Event):
    """Enhanced audio frame for downstream VAD, speaker, and turn detection."""

    TYPE: ClassVar[str] = "audio.enhanced_frame_received"
    audio_data: bytes
    sample_rate: int = 16000


@dataclass
class VADSpeechStart(Event):
    TYPE: ClassVar[str] = "vad.speech_start"
    origin: str = "client"


@dataclass
class VADSpeechEnd(Event):
    TYPE: ClassVar[str] = "vad.speech_end"
    origin: str = "client"


@dataclass
class ASRResultPartial(Event):
    """Incremental user transcript produced by audio ASR or text input."""

    TYPE: ClassVar[str] = "asr.result_partial"
    text: str = ""
    display_text: str = ""  # Cleaned text for frontend display
    speech_pause: bool = False
    origin: str = "asr"


@dataclass
class ASRResultFinal(Event):
    """Final user transcript that is ready for response generation."""

    TYPE: ClassVar[str] = "asr.result_final"
    text: str = ""
    display_text: str = ""  # Cleaned text for frontend display
    origin: str = "asr"


@dataclass
class LLMFirstChunk(Event):
    """Event for first LLM chunk/tool call (measure first token latency)."""

    TYPE: ClassVar[str] = "llm.first_chunk"


@dataclass
class LLMFirstSentence(Event):
    """Event for first synthesizable sentence (measure sentence latency)."""

    TYPE: ClassVar[str] = "llm.sentence_ready"


@dataclass
class TTSStarted(Event):
    TYPE: ClassVar[str] = "tts.started"
    response_id: str = ""


@dataclass
class TTSStopped(Event):
    TYPE: ClassVar[str] = "tts.stopped"
    response_id: str = ""


@dataclass
class TTSPaused(Event):
    TYPE: ClassVar[str] = "tts.paused"


@dataclass
class TTSResumed(Event):
    TYPE: ClassVar[str] = "tts.resumed"


@dataclass
class TTSFinished(Event):
    TYPE: ClassVar[str] = "tts.finished"
    response_id: str = ""


@dataclass
class LLMAgentResponseUpdate(Event):
    TYPE: ClassVar[str] = "llm_agent.response_update"
    response_id: str = ""
    text: str = ""


@dataclass
class LLMAgentResponseFinish(Event):
    """Final text emitted by the agent for one response.

    Attributes
    ----------
    response_id : str
        Stable identifier shared by generation, TTS, playback, and display.
    text : str
        Final response text.
    """

    TYPE: ClassVar[str] = "llm_agent.response_finish"
    response_id: str = ""
    text: str = ""


@dataclass
class ResponseUpdate(Event):
    """Text prefix whose corresponding TTS playback progress has been confirmed.

    Attributes
    ----------
    response_id : str
        Response whose cumulative playback-confirmed text is being updated.
    text : str
        Text prefix that has been played to the user.
    """

    TYPE: ClassVar[str] = "response.update"
    response_id: str = ""
    text: str = ""


@dataclass
class ResponseFinish(Event):
    """Final playback-confirmed text for a completed or interrupted TTS turn.

    Attributes
    ----------
    response_id : str
        Response whose playback settlement is complete.
    text : str
        Final response prefix that was actually played to the user.
    """

    TYPE: ClassVar[str] = "response.finish"
    response_id: str = ""
    text: str = ""


@dataclass
class TTSTextSynthesisStarted(Event):
    """Mark the start of one FIFO-ordered TTS text segment.

    Attributes
    ----------
    text : str
        Complete text of the segment about to be synthesized.
    """

    TYPE: ClassVar[str] = "tts.text_synthesis_started"
    response_id: str = ""
    text: str = ""


@dataclass
class TTSStreamingTextAccepted(Event):
    """Report text accepted by the active streaming TTS session.

    Attributes
    ----------
    text : str
        Incremental text successfully accepted by ``StreamingTextTTS``.
    prepared_audio_ms : float
        Duration of speed-processed PCM already prepared before the text was
        forwarded. The value is a lower-bound timeline anchor for this text.
    """

    TYPE: ClassVar[str] = "tts.streaming_text_accepted"
    response_id: str = ""
    text: str = ""
    prepared_audio_ms: float = 0.0


@dataclass
class TTSTextSynthesized(Event):
    """Text marker emitted after one synthesized text segment is fully produced.

    Attributes
    ----------
    text : str
        Text segment that was synthesized.
    audio_duration : float
        Duration of the final emitted PCM audio in milliseconds.
    audio_chunk : bytes
        Optional complete PCM 16-bit mono audio accepted for compatibility.
    sample_rate : int
        Sample rate of ``audio_chunk``. Zero when no audio is attached.
    """

    TYPE: ClassVar[str] = "tts.text_synthesized"
    response_id: str = ""
    text: str = ""
    audio_duration: float = 0.0
    audio_chunk: bytes = b""
    sample_rate: int = 0


@dataclass
class TTSTextDeliveryFinished(Event):
    """Mark completion of FIFO audio delivery for one TTS text segment.

    Attributes
    ----------
    text : str
        Text associated with the delivered audio segment.
    succeeded : bool
        Whether synthesis produced a complete deliverable segment.
    """

    TYPE: ClassVar[str] = "tts.text_delivery_finished"
    response_id: str = ""
    text: str = ""
    succeeded: bool = True


@dataclass
class TTSVoiceChange(Event):
    TYPE: ClassVar[str] = "tts.reference_audio_changed"
    voice_name: str = ""


@dataclass
class TTSEmotionChange(Event):
    TYPE: ClassVar[str] = "tts.emotion_changed"
    emotion_name: str = ""
    emotion_vector: list = None

    def __post_init__(self):
        super().__post_init__()
        if self.emotion_vector is None:
            self.emotion_vector = []


@dataclass
class TTSSpeedChange(Event):
    TYPE: ClassVar[str] = "tts.speed_changed"
    speed: float = 1.0  # 0.5 ~ 1.5


@dataclass
class TTSChunkReady(Event):
    """Indicates one TTS audio chunk is ready for sending. Not emitted when the chunk is generated."""
    TYPE: ClassVar[str] = "tts.chunk_ready"
    response_id: str = ""
    audio_chunk: bytes = b""
    sample_rate: int = 48000


@dataclass
class TTSChunkPlayed(Event):
    """Frontend confirmed playback completion for a TTS audio chunk.

    InputGateway publishes this after receiving tts_chunk_played so downstream
    listeners can observe frontend playback completion in FIFO order.
    """

    TYPE: ClassVar[str] = "tts.chunk_played_confirm"
    response_id: str = ""


@dataclass
class TTSPlaybackStopped(Event):
    """Frontend confirmed how much active audio played before an early stop.

    ``played_audio_ms`` contains only playback time not already represented by
    preceding ``TTSChunkPlayed`` events.
    """

    TYPE: ClassVar[str] = "tts.playback_stopped"
    response_id: str = ""
    played_audio_ms: float = 0.0


@dataclass
class TTSPlaybackFinished(Event):
    TYPE: ClassVar[str] = "tts.playback_finished"
    response_id: str = ""


@dataclass
class FullAudioFrameReady(Event):
    TYPE: ClassVar[str] = "audio.full_frame_ready"
    audio_chunk: bytes = b""
    sample_rate: int = 48000
    channels: int = 2
    format: str = "pcm_s16le"


@dataclass
class ErrorOccurred(Event):
    TYPE: ClassVar[str] = "error.occurred"
    error_type: str = ""
    error_message: str = ""


@dataclass
class CaptionUpdated(Event):
    TYPE: ClassVar[str] = "caption.updated"
    text: str = ""
    is_final: bool = False
    reason: str = ""  # optional: refresh|final|error


@dataclass
class ToolCallOccurred(Event):
    """LLM/Agent tool invocation notification."""

    TYPE: ClassVar[str] = "agent.tool_called"
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalUpdated(Event):
    TYPE: ClassVar[str] = "retrieval.updated"
    text: str = ""
    is_final: bool = False


@dataclass
class EmbeddingStatusUpdated(Event):
    """Embedding lifecycle update consumed by the LLM agent context manager."""

    TYPE: ClassVar[str] = "embedding.status_updated"
    status: str = ""
    text: str | None = None
    vector_store_instance: Any = None


@dataclass
class LLMAgentLoop(Event):
    """Request one agent-context accept loop iteration for the session."""

    TYPE: ClassVar[str] = "llm.agent_loop"


@dataclass
class TextForEmbeddingReady(Event):
    TYPE: ClassVar[str] = "embeddings.text_ready"
    text: str = ""


# ==================== Metrics / Latency Events ====================


@dataclass
class LatencyMetricsUpdated(Event):
    """Fine-grained backend latency metrics (milliseconds)."""

    TYPE: ClassVar[str] = "metrics.latency_updated"
    network_latency_ms: int = 0  # VAD start → backend receives first frame
    asr_latency_ms: int = 0  # First frame → ASR final result
    llm_first_token_ms: int = 0  # ASR done → LLM first token
    llm_sentence_ms: int = 0  # ASR done → first synthesizable sentence
    tts_first_chunk_ms: int = 0  # LLM sentence done → first TTS chunk


# ==================== Mediator (Turn Taking) Events ====================


@dataclass
class TurnTTSStartRequested(Event):
    TYPE: ClassVar[str] = "turn.tts_start_requested"
    response_id: str = ""


@dataclass
class TurnTTSDeliveryStartRequested(Event):
    """Allow one prepared TTS response to begin client delivery.

    ``response_id`` selects the prepared response whose text and audio may now
    cross the client-delivery boundary.
    """

    TYPE: ClassVar[str] = "turn.tts_delivery_start_requested"
    response_id: str = ""


@dataclass
class TurnTTSPauseRequested(Event):
    TYPE: ClassVar[str] = "turn.tts_pause_requested"


@dataclass
class TurnTTSResumeRequested(Event):
    TYPE: ClassVar[str] = "turn.tts_resume_requested"


@dataclass
class TurnTTSStopRequested(Event):
    TYPE: ClassVar[str] = "turn.tts_stop_requested"
    response_id: str | None = None
    reason: str = ""  # e.g., verification_valid|playback_finished


@dataclass
class TurnTTSFlushRequested(Event):
    TYPE: ClassVar[str] = "turn.tts_flush_requested"
    response_id: str = ""


@dataclass
class ConsumeLLMAgentGenerationRequested(Event):
    """Request consumption of one LLM-agent output stream.

    Attributes
    ----------
    stream : AsyncIterator[AgentOutput]
        Agent output stream to consume.
    persistent : bool
        Whether turn-stop events must preserve the stream.
    """

    TYPE: ClassVar[str] = "llm_agent.consume_generation_requested"
    stream: AsyncIterator[AgentOutput]
    persistent: bool = False


@dataclass
class TurnLLMAgentResumeRequested(Event):
    TYPE: ClassVar[str] = "turn.llm_agent_resume_requested"


@dataclass
class TurnLLMAgentPauseRequested(Event):
    TYPE: ClassVar[str] = "turn.llm_agent_pause_requested"


@dataclass
class TurnLLMAgentStopRequested(Event):
    TYPE: ClassVar[str] = "turn.llm_agent_stop_requested"
    reason: str = ""  # e.g., vad_start|verification_valid


@dataclass
class TurnInputAbortRequested(Event):
    """Request cancellation of an unfinished input turn.

    Attributes
    ----------
    origin : str
        Origin of the replacement turn requesting cancellation.
    """

    TYPE: ClassVar[str] = "turn.input_abort_requested"
    origin: str = ""


@dataclass
class TurnASRStartRequested(Event):
    TYPE: ClassVar[str] = "turn.asr_start_requested"


@dataclass
class TurnASREndRequested(Event):
    """
    Indicates hard turn end. ASR model state is reset. Turn moves to next.
    """

    TYPE: ClassVar[str] = "turn.asr_end_requested"


@dataclass
class TurnASRPauseRequested(Event):
    """
    Used when user indicates a wait, or pauses in the speech. Triggers recognition once. ASR model state is preserved; turn unchanged.
    """

    TYPE: ClassVar[str] = "turn.asr_pause_requested"


@dataclass
class TurnTTSTextAppendRequested(Event):
    """Request to append text into ongoing TTS stream (sim-trans)."""

    TYPE: ClassVar[str] = "turn.tts_text_append_requested"
    response_id: str = ""
    text: str = ""


@dataclass
class TTSResponseClosed(Event):
    """Signal that one response has completed playback settlement and cleanup.

    ``response_id`` releases only the matching coordinator delivery slot.
    """

    TYPE: ClassVar[str] = "tts.response_closed"
    response_id: str = ""


# ==================== Speaker Notification (Frontend) ====================


@dataclass
class SpeakerRecognized(Event):
    """Speaker-recognition result for frontend display."""

    TYPE: ClassVar[str] = "speaker.recognized"
    speaker_id: str | None = None
    reason: str = ""


# ==================== Dynamic Model Switching Events ====================


@dataclass
class TTSModelSwitchRequested(Event):
    """Request to switch the IndexTTS protocol version."""

    TYPE: ClassVar[str] = "tts.model_switch_requested"
    model_type: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMModelSwitchRequested(Event):
    """Request to switch LLM configuration (ChatOpenAI model/base_url)."""

    TYPE: ClassVar[str] = "llm.model_switch_requested"
    model_name: str = ""
    base_url: str = ""
    api_key: str = ""
    extra_body: dict | None = None


@dataclass
class ClockSyncReceived(Event):
    """Clock-sync event for offset calculation."""

    TYPE: ClassVar[str] = "clock.sync_received"
    client_send_ts: float = 0.0
    server_recv_ts: float = 0.0
    client_recv_ts: float = 0.0


@dataclass
class SessionConfigReceived(Event):
    """Client sent per-session configuration (e.g., recording path)."""

    TYPE: ClassVar[str] = "session.config_received"
    recording_path: str | None = None


# ==================== Turn Detection Events ====================


@dataclass
class TurnDetectorStopSpeaking(Event):
    """Turn detector determined ai should stop speaking."""

    TYPE: ClassVar[str] = "turn_detector.stop_speaking"


@dataclass
class TurnDetectorStartGeneration(Event):
    """Turn detector determined ai should start generation."""

    TYPE: ClassVar[str] = "turn_detector.start_generation"
