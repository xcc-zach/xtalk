# -*- coding: utf-8 -*-
import time
import uuid
from dataclasses import dataclass, field, asdict, make_dataclass
from typing import ClassVar, Dict, Any, Type


@dataclass
class BaseEvent:
    """Base dataclass for all xtalk events."""

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
) -> Type[BaseEvent]:
    """Dynamically create a BaseEvent subclass with the given field defaults."""
    fields = fields or {}
    dataclass_fields = []
    for key, default in fields.items():
        dataclass_fields.append((key, type(default), field(default=default)))
    type_name = type_name or name.lower()
    return make_dataclass(
        name, dataclass_fields, bases=(BaseEvent,), namespace={"TYPE": type_name}
    )


@dataclass
class WebSocketMessageReceived(BaseEvent):
    TYPE: ClassVar[str] = "websocket.message_received"
    message: str = ""


@dataclass
class AudioFrameReceived(BaseEvent):
    TYPE: ClassVar[str] = "audio.frame_received"
    audio_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    is_final: bool = False
    audio_format: str = "pcm_s16le"


@dataclass
class EnhancedAudioFrameReceived(BaseEvent):
    """Enhanced audio frame for downstream ASR/VAD."""

    TYPE: ClassVar[str] = "audio.enhanced_frame_received"
    audio_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    is_final: bool = False
    audio_format: str = "pcm_s16le"


@dataclass
class VADSpeechStart(BaseEvent):
    TYPE: ClassVar[str] = "vad.speech_start"
    confidence: float = 0.0
    speech_probability: float = 0.0


@dataclass
class VADSpeechEnd(BaseEvent):
    TYPE: ClassVar[str] = "vad.speech_end"
    confidence: float = 0.0
    speech_probability: float = 0.0


@dataclass
class ASRResultPartial(BaseEvent):
    TYPE: ClassVar[str] = "asr.result_partial"
    text: str = ""
    confidence: float = 0.0
    is_final: bool = False
    display_text: str = ""  # Cleaned text for frontend display
    turn_id: int = 0


@dataclass
class ASRResultFinal(BaseEvent):
    TYPE: ClassVar[str] = "asr.result_final"
    text: str = ""
    confidence: float = 0.0
    is_final: bool = True
    display_text: str = ""  # Cleaned text for frontend display
    semantic_tag: str = "<complete>"
    turn_id: int = 0


@dataclass
class LLMFirstChunk(BaseEvent):
    """Event for first LLM chunk/tool call (measure first token latency)."""

    TYPE: ClassVar[str] = "llm.first_chunk"


@dataclass
class LLMFirstSentence(BaseEvent):
    """Event for first synthesizable sentence (measure sentence latency)."""

    TYPE: ClassVar[str] = "llm.sentence_ready"


@dataclass
class TTSStarted(BaseEvent):
    TYPE: ClassVar[str] = "tts.started"


@dataclass
class TTSStopped(BaseEvent):
    TYPE: ClassVar[str] = "tts.stopped"


@dataclass
class TTSPaused(BaseEvent):
    TYPE: ClassVar[str] = "tts.paused"


@dataclass
class TTSResumed(BaseEvent):
    TYPE: ClassVar[str] = "tts.resumed"


@dataclass
class TTSFinished(BaseEvent):
    TYPE: ClassVar[str] = "tts.finished"


@dataclass
class LLMAgentResponseUpdate(BaseEvent):
    TYPE: ClassVar[str] = "llm_agent.response_update"
    text: str = ""
    turn_id: int = 0


@dataclass
class LLMAgentResponseFinish(BaseEvent):
    TYPE: ClassVar[str] = "llm_agent.response_finish"
    text: str = ""
    turn_id: int = 0


@dataclass
class TTSVoiceChange(BaseEvent):
    TYPE: ClassVar[str] = "tts.reference_audio_changed"
    voice_name: str = ""


@dataclass
class TTSEmotionChange(BaseEvent):
    TYPE: ClassVar[str] = "tts.emotion_changed"
    emotion_name: str = ""
    emotion_vector: list = None

    def __post_init__(self):
        if self.emotion_vector is None:
            self.emotion_vector = []


@dataclass
class TTSSpeedChange(BaseEvent):
    TYPE: ClassVar[str] = "tts.speed_changed"
    speed: float = 1.0  # 0.5 ~ 1.5


@dataclass
class TTSChunkGenerated(BaseEvent):
    TYPE: ClassVar[str] = "tts.chunk_generated"
    audio_chunk: bytes = b""


@dataclass
class TTSChunkPlayed(BaseEvent):
    """Frontend confirmed playback completion for a TTS audio chunk.

    InputGateway publishes this after receiving tts_chunk_played.
    RecordingManager subscribes and writes the chunk into right-channel buffer.
    Chunks are processed in FIFO order (no index needed).
    """

    TYPE: ClassVar[str] = "tts.chunk_played_confirm"


@dataclass
class TTSPlaybackFinished(BaseEvent):
    TYPE: ClassVar[str] = "tts.playback_finished"


@dataclass
class VerificationResult(BaseEvent):
    TYPE: ClassVar[str] = "verification.result"
    is_valid: bool = False
    text: str = ""
    confidence: float = 0.0
    reason: str = ""
    text_length: int = 0
    chunk_count: int = 0


@dataclass
class ErrorOccurred(BaseEvent):
    TYPE: ClassVar[str] = "error.occurred"
    error_type: str = ""
    error_message: str = ""
    component: str = ""


@dataclass
class CaptionUpdated(BaseEvent):
    TYPE: ClassVar[str] = "caption.updated"
    text: str = ""
    is_final: bool = False
    reason: str = ""  # optional: refresh|final|error


@dataclass
class ThoughtUpdated(BaseEvent):
    TYPE: ClassVar[str] = "thought.updated"
    text: str = ""
    is_final: bool = False


@dataclass
class ToolCallOccurred(BaseEvent):
    """LLM/Agent tool invocation notification."""

    TYPE: ClassVar[str] = "agent.tool_called"
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalUpdated(BaseEvent):
    TYPE: ClassVar[str] = "retrieval.updated"
    text: str = ""
    is_final: bool = False


@dataclass
class TextForEmbeddingReady(BaseEvent):
    TYPE: ClassVar[str] = "embeddings.text_ready"
    text: str = ""


# ==================== Metrics / Latency Events ====================


@dataclass
class LatencyMetricsUpdated(BaseEvent):
    """Fine-grained backend latency metrics (milliseconds)."""

    TYPE: ClassVar[str] = "metrics.latency_updated"
    network_latency_ms: int = 0  # VAD start → backend receives first frame
    asr_latency_ms: int = 0  # First frame → ASR final result
    llm_first_token_ms: int = 0  # ASR done → LLM first token
    llm_sentence_ms: int = 0  # ASR done → first synthesizable sentence
    tts_first_chunk_ms: int = 0  # LLM sentence done → first TTS chunk


# ==================== Mediator (Turn Taking) Events ====================


@dataclass
class TurnTTSStartRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_start_requested"


@dataclass
class TurnTTSPauseRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_pause_requested"


@dataclass
class TurnTTSResumeRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_resume_requested"


@dataclass
class TurnTTSStopRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_stop_requested"
    reason: str = ""  # e.g., verification_valid|playback_finished


@dataclass
class TurnTTSFlushRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_flush_requested"


@dataclass
class TurnLLMAgentStartRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.llm_agent_start_requested"
    text: str = ""


@dataclass
class TurnLLMAgentResumeRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.llm_agent_resume_requested"
    text: str = ""


@dataclass
class TurnLLMAgentPauseRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.llm_agent_pause_requested"


@dataclass
class TurnLLMAgentStopRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.llm_agent_stop_requested"
    reason: str = ""  # e.g., vad_start|verification_valid


@dataclass
class TurnASRResetRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.asr_reset_requested"


@dataclass
class TurnASRStartRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.asr_start_requested"


@dataclass
class TurnASREndRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.asr_end_requested"


@dataclass
class TurnASRFlushRequested(BaseEvent):
    """Request ASR to flush current stable segment in sim-trans mode."""

    TYPE: ClassVar[str] = "turn.asr_flush_requested"
    reason: str = ""  # e.g., vad_end


# ==================== Sim-Trans (Simultaneous Generation) Extensions ====================


@dataclass
class ASRStableSegmentReady(BaseEvent):
    """ASR stable segment ready (simultaneous translation)."""

    TYPE: ClassVar[str] = "asr.stable_segment_ready"
    text: str = ""
    stability: float = 1.0
    start_ts: float = 0.0
    end_ts: float = 0.0


@dataclass
class TurnTTSTextAppendRequested(BaseEvent):
    """Request to append text into ongoing TTS stream (sim-trans)."""

    TYPE: ClassVar[str] = "turn.tts_text_append_requested"
    text: str = ""
    reason: str = "asr_partial"


@dataclass
class TranscriptionRefined(BaseEvent):
    """Backend refines or corrects transcription for simultaneous mode."""

    TYPE: ClassVar[str] = "asr.transcription_refined"
    text: str = ""


# ==================== Speaker Notification (Frontend) ====================


@dataclass
class SpeakerRecognized(BaseEvent):
    """Speaker-recognition result for frontend display."""

    TYPE: ClassVar[str] = "speaker.recognized"
    speaker_id: str | None = None
    reason: str = ""


# ==================== Dynamic Model Switching Events ====================


@dataclass
class TTSModelSwitchRequested(BaseEvent):
    """Request to switch TTS model (IndexTTS / IndexTTS2)."""

    TYPE: ClassVar[str] = "tts.model_switch_requested"
    model_type: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMModelSwitchRequested(BaseEvent):
    """Request to switch LLM configuration (ChatOpenAI model/base_url)."""

    TYPE: ClassVar[str] = "llm.model_switch_requested"
    model_name: str = ""
    base_url: str = ""
    api_key: str = ""
    extra_body: dict | None = None


@dataclass
class ClockSyncReceived(BaseEvent):
    """Clock-sync event for offset calculation."""

    TYPE: ClassVar[str] = "clock.sync_received"
    client_send_ts: float = 0.0
    server_recv_ts: float = 0.0
    client_recv_ts: float = 0.0


@dataclass
class SessionConfigReceived(BaseEvent):
    """Client sent per-session configuration (e.g., recording path)."""

    TYPE: ClassVar[str] = "session.config_received"
    recording_path: str | None = None
