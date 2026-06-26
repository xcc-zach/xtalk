from abc import abstractmethod, ABC
from typing import Optional, TypedDict
from typing import Dict, Any

from ..models.agents.interfaces import Agent
from ..models.asr.interfaces import ASR
from ..models.captioner.interfaces import Captioner
from ..models.embeddings.interfaces import Embeddings
from ..models.punt_restorer.interfaces import PuntRestorer
from ..models.rewriters.interfaces import Rewriter
from ..models.speaker_encoder.interfaces import SpeakerEncoder
from ..models.speech_enhancer.interfaces import SpeechEnhancer
from ..models.speech_speed_controller.interfaces import SpeechSpeedController
from ..models.tts.interfaces import TTS
from ..models.turn_detector.interfaces import TurnDetector
from ..models.vad.interfaces import VAD


class PipelineOutputBase(TypedDict):
    """
    TypedDict for the output of the pipeline.

    audio: PCM 16bit mono, 48000Hz bytes
    """

    audio: bytes
    text: str
    asr_text: Optional[str]


class PipelineOutput(PipelineOutputBase, total=False):
    """
    Optional fields will be extended by concrete pipelines; dropped unused
    `caption`/`meta` keys.

    - tts_ref_audio: voice switch instruction (translated to events by TTSManager)
    - tts_emotion: emotion switch instruction (translated to events by TTSManager)
    - tool_call: tool invocation info for UI hints
    - tts_speed: speed multiplier for TTS audio
    """

    # tts_ref_audio: Optional instruction for TTS voice change.
    # Keys mirror event fields: 'ref_audio_name' and 'ref_audio_path'.
    tts_ref_audio: Dict[str, str]
    # tts_emotion: Optional instruction for TTS emotion change.
    # Keys: 'emotion' (emotion name) and 'vector' (emotion vector).
    tts_emotion: Dict[str, Any]
    # tool_call: Optional tool call info for UI hints.
    # Keys: 'name' (str) and 'args' (dict).
    tool_call: Dict[str, Any]
    tts_speed: float


class Pipeline(ABC):
    """Define the model accessors expected by Xtalk services.

    Notes
    -----
    Concrete pipelines expose the models and helpers consumed by service
    managers. Sample applications commonly subclass ``DefaultPipeline`` to add
    extra components while retaining this interface.
    """

    @abstractmethod
    def clone(self) -> "Pipeline":
        """Clone the pipeline for a new session.

        Returns
        -------
        Pipeline
            Session-safe pipeline instance.
        """
        pass

    def get_asr_model(self) -> ASR | None:
        """Return the configured ASR model, if available.

        Returns
        -------
        ASR | None
            ASR implementation or ``None`` when the pipeline omits speech
            recognition.
        """
        return None

    def get_tts_model(self) -> TTS | None:
        """Return the configured TTS model, if available.

        Returns
        -------
        TTS | None
            TTS implementation or ``None``.
        """
        return None

    def get_agent(self) -> Agent | None:
        """Return the configured agent, if available.

        Returns
        -------
        Agent | None
            Agent implementation or ``None``.
        """
        return None

    def get_captioner_model(self) -> Captioner | None:
        """Return the configured captioner, if available.

        Returns
        -------
        Captioner | None
            Captioner implementation or ``None``.
        """
        return None

    def get_punt_restorer_model(self) -> PuntRestorer | None:
        """Return the configured punctuation restorer, if available.

        Returns
        -------
        PuntRestorer | None
            Punctuation restoration model or ``None``.
        """
        return None

    def get_caption_rewriter_model(self) -> Rewriter | None:
        """Return the caption rewriter used by caption-related managers.

        Returns
        -------
        Rewriter | None
            Caption rewriter or ``None``.
        """
        return None

    def get_vad_model(self) -> VAD:
        """Return the configured voice activity detector, if available.

        Returns
        -------
        VAD | None
            VAD implementation or ``None``.
        """
        return None

    def get_enhancer_model(self) -> SpeechEnhancer | None:
        """Return the configured speech enhancer, if available.

        Returns
        -------
        SpeechEnhancer | None
            Speech enhancement model or ``None``.
        """
        return None

    def get_speaker_encoder(self) -> SpeakerEncoder | None:
        """Return the configured speaker encoder, if available.

        Returns
        -------
        SpeakerEncoder | None
            Speaker encoder or ``None``.
        """
        return None

    def get_speed_controller(self) -> SpeechSpeedController | None:
        """Return the optional TTS speed controller.

        Returns
        -------
        SpeechSpeedController | None
            Speed controller or ``None``.
        """
        return None

    def get_embeddings_model(self) -> Embeddings | None:
        """Return the embeddings model used for retrieval features.

        Returns
        -------
        Embeddings | None
            Embeddings model or ``None``.
        """
        return None

    def get_turn_detector_model(self) -> TurnDetector | None:
        """Return the configured turn detector, if available.

        Returns
        -------
        TurnDetector | None
            Turn detector or ``None``.
        """
        return None
