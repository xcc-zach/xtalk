import asyncio
from abc import abstractmethod, ABC
from typing import Optional, TypedDict, Iterable, AsyncIterator
from typing import Dict, Any
import numpy as np
from langchain_core.embeddings import Embeddings
from ..speech.interfaces import (
    ASR,
    TTS,
    Captioner,
    PuntRestorer,
    VAD,
    SpeechEnhancer,
    SpeakerEncoder,
    SpeechSpeedController,
    TurnDetector,
)
from ..rewriter.interfaces import Rewriter
from ..llm_agent.interfaces import Agent
from .context import PipelineContext


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
    """
    Abstract base class for the pipeline.
    """

    # Context dict shared across modules for runtime state (thought/caption/etc.)
    _context: "PipelineContext"

    @property
    def context(self) -> "PipelineContext":
        """Get pipeline context, creating default values when absent.

        - Lazily initializes with default keys (thought/caption/etc.)
        - Returns dict reference; callers should treat as read-only unless
          updating via the setter with merged fields.
        """
        if not hasattr(self, "_context"):
            # Initialize default context (fields default to None)
            self._context = {
                "thought": None,
                "caption": None,
                "speaker_id": None,
                "text_to_embed": None,
                "vector_store_instance": None,
            }
        return self._context

    @context.setter
    def context(self, value: "PipelineContext") -> None:
        """Overwrite pipeline context."""
        self._context = value

    @abstractmethod
    def clone(self) -> "Pipeline":
        """
        Clone the pipeline.
        """
        pass

    def get_asr_model(self) -> ASR | None:
        """
        Get the ASR model. If the pipeline does not have an ASR model, return None.
        """
        return None

    def get_tts_model(self) -> TTS | None:
        """
        Get the TTS model. If the pipeline does not have a TTS model, return None.
        """
        return None

    def get_agent(self) -> Agent | None:
        """
        Get the LLM agent. If the pipeline does not have a LLM agent, return None.
        """
        return None

    def get_captioner_model(self) -> Captioner | None:
        """
        Get the Captioner model. If the pipeline does not have a Captioner model, return None.
        """
        return None

    def get_punt_restorer_model(self) -> PuntRestorer | None:
        """
        Get the PuntRestorer model. If the pipeline does not have a PuntRestorer model, return None.
        """
        return None

    def get_caption_rewriter_model(self) -> Rewriter | None:
        """
        Get the Caption Rewriter model. If the pipeline does not have a Caption Rewriter model
        , return None.
        """
        return None

    def get_thought_rewriter_model(self) -> Rewriter | None:
        """Get the Thought Rewriter model (for ThoughtManager)."""
        return None

    def get_vad_model(self) -> VAD:
        """
        Get the VAD model. If the pipeline does not have a VAD model, return None.
        """
        return None

    def get_enhancer_model(self) -> SpeechEnhancer | None:
        """
        Get the SpeechEnhancer model. If the pipeline does not have a SpeechEnhancer model, return None.
        """
        return None

    def get_speaker_encoder(self) -> SpeakerEncoder | None:
        """
        Get the Speaker Encoder model. If the pipeline does not have a Speaker Encoder model, return None.
        """
        return None

    def get_speed_controller(self) -> SpeechSpeedController | None:
        """Return the optional TTS speed controller if available."""
        return None

    def get_embeddings_model(self) -> Embeddings | None:
        return None

    def get_turn_detector_model(self) -> TurnDetector | None:
        return None
