# -*- coding: utf-8 -*-
import time
import types
from dataclasses import dataclass, field, make_dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    ClassVar,
    Dict,
    Literal,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import ConfigDict, Field, field_validator, with_config
from pydantic.dataclasses import dataclass as validated_dataclass
from typing_extensions import TypedDict, is_typeddict

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
    """Speech onset with optional server-detection audio lookback in milliseconds."""

    audio_lookback_ms: float | None = None
    TYPE: ClassVar[str] = "vad.speech_start"
    origin: str = "client"


@dataclass
class VADSpeechEnd(Event):
    TYPE: ClassVar[str] = "vad.speech_end"
    origin: str = "client"


class ASRGateState(str, Enum):
    """Mark whether a speaker-aware gate accepted an ASR event."""

    UNCHECKED = "unchecked"
    ACCEPTED = "accepted"


@dataclass
class ASRResultPartial(Event):
    """Incremental user transcript produced by audio ASR or text input."""

    TYPE: ClassVar[str] = "asr.result_partial"
    text: str = ""
    display_text: str = ""  # Cleaned text for frontend display
    item_id: str = ""
    delta: str = ""
    update_mode: str = "append"
    revision: int = 0
    speech_pause: bool = False
    origin: str = "asr"
    turn_id: int = 0
    segment_id: int = 0
    gate_state: ASRGateState = ASRGateState.UNCHECKED


@dataclass
class ASRFailed(Event):
    """Report a transcription failure associated with one input item.

    Attributes
    ----------
    item_id : str
        Input item whose transcription failed.
    message : str
        Provider failure message.
    """

    TYPE: ClassVar[str] = "asr.failed"
    item_id: str = ""
    message: str = ""


@dataclass
class ASRResultFinal(Event):
    """Final user transcript that is ready for response generation."""

    TYPE: ClassVar[str] = "asr.result_final"
    item_id: str = ""
    text: str = ""
    display_text: str = ""  # Cleaned text for frontend display
    origin: str = "asr"
    turn_id: int = 0
    segment_id: int = 0
    gate_state: ASRGateState = ASRGateState.UNCHECKED


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
    response_id: str = ""


@dataclass
class TTSResumed(Event):
    TYPE: ClassVar[str] = "tts.resumed"
    response_id: str = ""


@dataclass
class TTSFinished(Event):
    TYPE: ClassVar[str] = "tts.finished"
    response_id: str = ""


@dataclass
class LLMAgentResponseUpdate(Event):
    TYPE: ClassVar[str] = "llm_agent.response_update"
    response_id: str = ""
    text: str = ""
    delta: str = ""


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
    response_id: str = ""
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
    response_id: str | None = None


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
    """Start ASR with the turn-owned item ID and retained pre-onset audio.

    Attributes
    ----------
    item_id : str
        Stable input item identifier allocated by the turn-taking manager.
    """

    audio_lookback_ms: float | None = None
    TYPE: ClassVar[str] = "turn.asr_start_requested"
    item_id: str = ""
    turn_id: int = 0
    segment_id: int = 0


@dataclass
class TurnASREndRequested(Event):
    """End a hard turn and reset ASR state before the next turn.

    Attributes
    ----------
    item_id : str
        Input item identifier retained from the corresponding ASR start.
    """

    TYPE: ClassVar[str] = "turn.asr_end_requested"
    item_id: str = ""
    turn_id: int = 0
    segment_id: int = 0


@dataclass
class TurnASRPauseRequested(Event):
    """
    Used when user indicates a wait, or pauses in the speech. Triggers recognition once. ASR model state is preserved; turn unchanged.
    """

    TYPE: ClassVar[str] = "turn.asr_pause_requested"
    turn_id: int = 0
    segment_id: int = 0


# ==================== Multi-speaker Diarization Events ====================


@dataclass
class SpeakerDiarizationPartial(Event):
    """Replaceable diarization result for the current VAD-segment snapshot."""

    TYPE: ClassVar[str] = "speaker_diarization.partial"
    turn_id: int = 0
    segment_id: int = 0
    revision: int = 0
    source_start_sample: int = 0
    source_end_sample: int = 0
    sample_rate: int = 16000
    raw_text: str = ""
    diarization_text: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class SpeakerInterruptionDecision(Event):
    """Resolve a paused barge-in after identifying its active speaker."""

    TYPE: ClassVar[str] = "speaker_diarization.interruption_decision"
    turn_id: int = 0
    segment_id: int = 0
    speaker_id: str | None = None
    should_interrupt: bool = False
    reason: str = ""


@dataclass
class SpeakerDiarizationSegmentFinal(Event):
    """Terminal diarization result for one VAD segment."""

    TYPE: ClassVar[str] = "speaker_diarization.segment_final"
    turn_id: int = 0
    segment_id: int = 0
    source_start_sample: int = 0
    source_end_sample: int = 0
    sample_rate: int = 16000
    raw_text: str = ""
    diarization_text: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str = ""
    latency_ms: float = 0.0


@dataclass
class SpeakerDiarizationTurnFinal(Event):
    """Terminal diarization timeline after every segment in a hard turn ends."""

    TYPE: ClassVar[str] = "speaker_diarization.turn_final"
    turn_id: int = 0
    segment_ids: list[int] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    diarization_text: str = ""
    active_speaker_id: str | None = None
    degraded: bool = False
    degraded_reason: str = ""


@dataclass
class MultiSpeakerTurnReady(Event):
    """ASR transcript joined with its turn-level speaker timeline."""

    TYPE: ClassVar[str] = "multi_speaker.turn_ready"
    turn_id: int = 0
    asr_text: str = ""
    diarization_text: str = ""
    diarization_segments: list[dict[str, Any]] = field(default_factory=list)
    active_speaker_id: str | None = None
    should_respond: bool = True
    degraded: bool = False
    degraded_reason: str = ""


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


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Milliseconds = Annotated[int, Field(ge=0, le=10000)]
SpeechSpeed = Annotated[float, Field(ge=0.25, le=1.5, allow_inf_nan=False)]


@with_config(ConfigDict(strict=True, extra="forbid"))
class _ClientOptions(TypedDict, total=False):
    """Optional client correlation fields."""

    event_id: Annotated[str, Field(min_length=1, max_length=512)]


@with_config(ConfigDict(strict=True, extra="allow"))
class ClientEvent(_ClientOptions):
    """Describe a client message envelope and optional client event ID."""

    type: Annotated[str, Field(min_length=1)]


@with_config(ConfigDict(strict=True, extra="forbid"))
class InputAudioControl(ClientEvent):
    """Commit or clear audio without additional payload fields."""


@with_config(ConfigDict(strict=True, extra="forbid"))
class InputSpeechStart(ClientEvent, total=False):
    """Report client speech onset with an optional finite timestamp."""

    timestamp: Annotated[float, Field(allow_inf_nan=False)]


@with_config(ConfigDict(strict=True, extra="forbid"))
class InputAudioAppend(ClientEvent):
    """Describe a Base64 PCM audio append message."""

    audio: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class TranscriptionMode(TypedDict):
    """Describe whether an ASR revision appends or replaces text."""

    mode: Literal["append", "replace"]
    revision: int


@with_config(ConfigDict(strict=True, extra="forbid"))
class ServerEvent(TypedDict):
    """Describe the mandatory server message envelope."""

    type: str
    event_id: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class TextPart(TypedDict):
    """Text content of an injected conversation message."""

    type: Literal["input_text", "text"]
    text: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class _ItemOptions(TypedDict, total=False):
    """Optional client-assigned conversation item identifier."""

    id: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class MessageItem(_ItemOptions):
    """Required fields of a textual conversation item."""

    type: Literal["message"]
    role: Literal["user", "assistant", "system"]
    content: Annotated[list[TextPart], Field(min_length=1)]


@with_config(ConfigDict(strict=True, extra="forbid"))
class TextGenerationExtension(TypedDict, total=False):
    """Opt a user item into Xtalk's existing automatic text turn flow."""

    generate_response: bool


@with_config(ConfigDict(strict=True, extra="forbid"))
class _GenerationOptions(TypedDict, total=False):
    """Optional Xtalk text-generation request extension."""

    xtalk: TextGenerationExtension


@with_config(ConfigDict(strict=True, extra="forbid"))
class ConversationItemCreate(ClientEvent, _GenerationOptions):
    """Inject context, optionally requesting the Xtalk automatic text turn."""

    item: MessageItem


@with_config(ConfigDict(strict=True, extra="forbid"))
class PlaybackAcknowledgement(ClientEvent):
    """Associate a block completion or playback settlement with its response."""

    response_id: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class PlaybackStopped(PlaybackAcknowledgement):
    """Report incremental played duration of the unconfirmed audio block."""

    played_audio_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]


@with_config(ConfigDict(strict=True, extra="forbid"))
class ContentIdentity(TypedDict):
    """Stable identifiers for generated output content."""

    response_id: str
    item_id: str
    output_index: int
    content_index: int


@with_config(ConfigDict(strict=True, extra="forbid"))
class OutputDelta(ServerEvent, ContentIdentity):
    """A generated text, audio, or synthesized transcript delta."""

    delta: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class TranscriptionDelta(ServerEvent):
    """A revisable ASR preview for one input item."""

    item_id: str
    content_index: int
    delta: str
    xtalk: TranscriptionMode


INPUT_EVENT_TYPES = (
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "input_audio_buffer.speech_start",
    "conversation.item.create",
    "response.cancel",
    "response.output_audio.playback.chunk_played",
    "response.output_audio.playback.finished",
    "response.output_audio.playback.stopped",
)


@with_config(ConfigDict(strict=True, extra="forbid"))
class PCMFormat(TypedDict):
    """The supported mono signed little-endian PCM16 wire format."""

    type: Literal["audio/pcm"]
    rate: Literal[24000]


@with_config(ConfigDict(strict=True, extra="forbid"))
class ServerVADConfig(TypedDict):
    """Effective server VAD options, subject to model capabilities."""

    type: Literal["server_vad"]
    threshold: Probability
    prefix_padding_ms: Milliseconds
    silence_duration_ms: Milliseconds
    create_response: bool
    interrupt_response: bool


@with_config(ConfigDict(strict=True, extra="forbid"))
class TranscriptionConfig(TypedDict):
    """Enable transcription through the configured model combination."""

    model: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class EmotionConfig(TypedDict):
    """Xtalk synthesis emotion; a nonempty name takes precedence over vector."""

    name: str
    vector: list[Annotated[float, Field(allow_inf_nan=False)]]


@with_config(ConfigDict(strict=True, extra="forbid"))
class InputAudioConfig(TypedDict):
    """Effective audio input configuration."""

    format: PCMFormat
    transcription: TranscriptionConfig | None
    turn_detection: ServerVADConfig | None


@with_config(ConfigDict(strict=True, extra="forbid"))
class OutputAudioConfig(TypedDict):
    """Effective synthesis configuration including Xtalk dynamic controls."""

    format: PCMFormat
    voice: Annotated[str, Field(min_length=1)]
    speed: SpeechSpeed
    emotion: EmotionConfig


@with_config(ConfigDict(strict=True, extra="forbid"))
class SessionAudioConfig(TypedDict):
    """Input and output audio settings for one temporary session."""

    input: InputAudioConfig
    output: OutputAudioConfig


@with_config(ConfigDict(strict=True, extra="forbid"))
class SessionExtensions(TypedDict):
    """Configure Xtalk-specific transport behavior for one session."""

    output_audio_encoding: Literal["base64", "binary_pcm"]


class BinaryAudioHeader(TypedDict):
    """Identify a versioned PCM envelope before its raw binary payload."""

    version: Literal[1]
    type: Literal["xtalk.output_audio.delta", "xtalk.audio.mix.delta"]
    event_id: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class RealtimeSession(TypedDict):
    """Complete effective configuration returned at creation and update."""

    id: str
    object: Literal["realtime.session"]
    type: Literal["realtime"]
    model: str
    instructions: str
    output_modalities: Annotated[
        list[Literal["audio", "text"]], Field(min_length=1, max_length=1)
    ]
    audio: SessionAudioConfig
    xtalk: SessionExtensions


def _partial_configuration(shape):
    """Derive optional configuration fields from their authoritative types."""

    def optional_type(annotation):
        if is_typeddict(annotation):
            return _partial_configuration(annotation)
        if get_origin(annotation) in (Union, types.UnionType):
            return Union[tuple(optional_type(arg) for arg in get_args(annotation))]
        return annotation

    fields = {
        key: optional_type(annotation)
        for key, annotation in get_type_hints(shape, include_extras=True).items()
        if key not in {"id", "object", "model"} or shape is not RealtimeSession
    }
    partial = TypedDict("Partial" + shape.__name__, fields, total=False)
    return with_config(ConfigDict(strict=True, extra="forbid"))(partial)


SessionPatch = _partial_configuration(RealtimeSession)


@with_config(ConfigDict(strict=True, extra="forbid"))
class SessionUpdate(ClientEvent):
    """A partial update derived from the effective session configuration."""

    session: SessionPatch


@with_config(ConfigDict(strict=True, extra="forbid"))
class SessionEvent(ServerEvent):
    """Payload shared by session.created and session.updated."""

    session: RealtimeSession


@with_config(ConfigDict(strict=True, extra="forbid"))
class TranscriptionCompleted(ServerEvent):
    """Final complete ASR text associated with one input content part."""

    item_id: str
    content_index: int
    transcript: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class _ErrorCorrelation(TypedDict, total=False):
    """Optional fields identifying a rejected client request."""

    param: str | None
    event_id: str | None


@with_config(ConfigDict(strict=True, extra="forbid"))
class RealtimeError(_ErrorCorrelation):
    """Structured protocol, transcription, or generation failure."""

    type: str
    code: str
    message: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class ErrorEvent(ServerEvent):
    """A request or runtime error independent of generation completion."""

    error: RealtimeError


@with_config(ConfigDict(strict=True, extra="forbid"))
class TranscriptionFailed(ServerEvent):
    """A failed transcription tied to its original input item."""

    item_id: str
    content_index: int
    error: RealtimeError


@with_config(ConfigDict(strict=True, extra="forbid"))
class ResponseResource(TypedDict):
    """Generation lifecycle resource, separate from playback settlement."""

    id: str
    status: Literal["in_progress", "completed", "cancelled", "failed", "incomplete"]
    output: list[dict[str, Any]]


@with_config(ConfigDict(strict=True, extra="forbid"))
class ResponseEvent(ServerEvent):
    """Response resource announced at generation start or termination."""

    response: ResponseResource


@with_config(ConfigDict(strict=True, extra="forbid"))
class OutputItemEvent(ServerEvent):
    """Add or complete one output message item."""

    response_id: str
    output_index: int
    item: dict[str, Any]


@with_config(ConfigDict(strict=True, extra="forbid"))
class ContentPartEvent(ServerEvent, ContentIdentity):
    """Add or complete a generated text or audio content part."""

    part: dict[str, Any]


@with_config(ConfigDict(strict=True, extra="forbid"))
class OutputTextDone(ServerEvent, ContentIdentity):
    """Complete generated text for a text-only response."""

    text: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class OutputTranscriptDone(ServerEvent, ContentIdentity):
    """Complete synthesized text after successful audio delivery."""

    transcript: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class InputItemCommitted(ServerEvent):
    """Identify an input audio item committed by a client or server turn."""

    item_id: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class SpeechBoundary(ServerEvent):
    """Existing speech boundary notification retaining its Xtalk origin."""

    origin: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class PlaybackSnapshot(ServerEvent):
    """Full playback-confirmed text at an update or final settlement."""

    response_id: str
    text: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class PlaybackControl(ServerEvent):
    """Pause, resume, or stop playback of the identified response."""

    response_id: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class CaptionSnapshot(ServerEvent):
    """Full caption snapshot retaining the existing reason and final flag."""

    text: str
    is_final: bool
    reason: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class RetrievalSnapshot(ServerEvent):
    """Complete retrieval result; successive messages replace displayed text."""

    text: str
    is_final: bool


@with_config(ConfigDict(strict=True, extra="forbid"))
class SpeakerSnapshot(ServerEvent):
    """Existing speaker identity notification without added analysis fields."""

    speaker_id: str
    reason: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class ThoughtSnapshot(ServerEvent):
    """Frontend-only thought receiver shape; no core producer is provided."""

    text: str


@with_config(ConfigDict(strict=True, extra="forbid"))
class ToolCalled(ServerEvent):
    """Existing public tool name and argument payload."""

    name: str
    args: dict[str, Any]


@with_config(ConfigDict(strict=True, extra="forbid"))
class LatencySnapshot(ServerEvent):
    """The five existing latency metrics, expressed in milliseconds."""

    network_latency_ms: int
    asr_latency_ms: int
    llm_first_token_ms: int
    llm_sentence_ms: int
    tts_first_chunk_ms: int


@with_config(ConfigDict(strict=True, extra="forbid"))
class QueueSnapshot(ServerEvent):
    """Current session admission queue position and concurrency counts."""

    position: int
    limit: int
    active: int


@with_config(ConfigDict(strict=True, extra="forbid"))
class MixedAudioDelta(ServerEvent):
    """Original mixed audio stream and its actual format metadata."""

    delta: str
    sample_rate: int
    channels: int
    format: str


@validated_dataclass(config=ConfigDict(strict=True, extra="forbid"))
class VADThresholdUpdateRequested(Event):
    """Validate or apply a session VAD threshold update."""

    threshold: Probability
    check_only: bool = False
    TYPE: ClassVar[str] = "vad.threshold_update_requested"


@validated_dataclass(config=ConfigDict(strict=True, extra="forbid"))
class TTSConfigurationUpdateRequested(Event):
    """Validate or apply synthesis settings; omitted fields remain unchanged."""

    voice: Annotated[str, Field(min_length=1)] | None = None
    emotion: EmotionConfig | None = None
    speed: SpeechSpeed | None = None
    check_only: bool = False
    TYPE: ClassVar[str] = "tts.configuration_update_requested"

    @field_validator("emotion")
    @classmethod
    def _require_emotion(cls, emotion: EmotionConfig | None) -> EmotionConfig | None:
        """Reject an explicit emotion update without a name or vector."""
        if emotion is not None and not (emotion["name"] or emotion["vector"]):
            raise ValueError("Emotion must contain name or vector")
        return emotion


@validated_dataclass(config=ConfigDict(strict=True, extra="forbid"))
class AgentInstructionsUpdateRequested(Event):
    """Validate or apply replacement agent instructions."""

    instructions: str
    check_only: bool = False
    TYPE: ClassVar[str] = "agent.instructions_update_requested"


@validated_dataclass(config=ConfigDict(strict=True, extra="forbid"))
class AgentContextAppendRequested(Event):
    """Append a text message to the configured agent context."""

    role: Literal["user", "assistant", "system"]
    content: str
    TYPE: ClassVar[str] = "agent.context_append_requested"


@validated_dataclass(config=ConfigDict(strict=True, extra="forbid"))
class ResponseCancelRequested(Event):
    """Cancel a response and settle its protocol lifecycle."""

    response_id: str | None = None
    TYPE: ClassVar[str] = "response.cancel_requested"
