from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from ..models.agents.interfaces import Agent
from ..models.asr.interfaces import ASR
from ..models.captioner.interfaces import Captioner
from ..models.punt_restorer.interfaces import PuntRestorer
from ..models.rewriters.interfaces import Rewriter
from ..models.rewriters.simple import DefaultCaptionRewriter
from ..models.speaker_encoder.interfaces import SpeakerEncoder
from ..models.speech_enhancer.interfaces import SpeechEnhancer
from ..models.speech_speed_controller.interfaces import SpeechSpeedController
from ..models.tts.interfaces import TTS
from ..models.turn_detector.interfaces import TurnDetector
from ..models.vad.interfaces import VAD
from .interfaces import Pipeline


def _maybe_clone(obj: Any) -> Any:
    """Unified cloning strategy: call .clone() when available; otherwise share."""
    if obj is None:
        return None
    clone_fn = getattr(obj, "clone", None)
    if callable(clone_fn):
        return clone_fn()
    return obj


@dataclass(init=False)
class DefaultPipeline(Pipeline):
    """Store the standard Xtalk model bundle for a session.

    Parameters
    ----------
    asr : ASR
        Speech recognition model.
    llm_agent : Agent
        Agent used to generate text responses and tool calls.
    tts : TTS
        Text-to-speech model.
    default_response : str, optional
        Fallback text returned when no better response is available.
    use_streaming_tts : bool, optional
        Whether service managers should prefer streaming TTS when supported.
    captioner : Captioner | None, optional
        Optional audio captioning model.
    punt_restorer_model : PuntRestorer | None, optional
        Optional punctuation restoration model.
    caption_rewriter : Rewriter | BaseChatModel | None, optional
        Optional caption rewriter or chat model to wrap as a rewriter.
    vad : VAD | None, optional
        Optional voice activity detector.
    speech_enhancer : SpeechEnhancer | None, optional
        Optional speech enhancement model.
    speaker_encoder : SpeakerEncoder | None, optional
        Optional speaker embedding model.
    speech_speed_controller : SpeechSpeedController | None, optional
        Optional post-processing speed controller for synthesized audio.
    embeddings : Embeddings | None, optional
        Optional embeddings backend for retrieval features.
    turn_detector : TurnDetector | None, optional
        Optional turn detector that coordinates interruption and generation.

    Notes
    -----
    The dataclass field metadata controls how instances are cloned for each
    session. Subclasses can add new fields as long as they expose an
    ``init_key`` metadata entry.
    """

    # ---- core models ----
    asr_model: ASR = field(metadata={"init_key": "asr", "clone": True})
    llm_agent: Agent = field(metadata={"init_key": "llm_agent", "clone": True})
    tts_model: TTS = field(metadata={"init_key": "tts", "clone": True})

    # ---- configs ----
    default_response: str = field(
        default="Sorry, I didn't catch that. Could you please say it again?",
        metadata={"init_key": "default_response", "clone": False},
    )
    use_streaming_tts: bool = field(
        default=True, metadata={"init_key": "use_streaming_tts", "clone": False}
    )

    # ---- optional modules ----
    captioner_model: Optional[Captioner] = field(
        default=None, metadata={"init_key": "captioner", "clone": False}
    )
    punt_restorer_model: Optional[PuntRestorer] = field(
        default=None, metadata={"init_key": "punt_restorer_model", "clone": False}
    )

    # These will eventually be normalized into Optional[Rewriter]
    caption_rewriter: Optional[Rewriter] = field(
        default=None, metadata={"init_key": "caption_rewriter", "clone": False}
    )
    vad_model: Optional[VAD] = field(
        default=None, metadata={"init_key": "vad", "clone": True}
    )
    enhancer_model: Optional[SpeechEnhancer] = field(
        default=None, metadata={"init_key": "speech_enhancer", "clone": True}
    )
    speaker_encoder: Optional[SpeakerEncoder] = field(
        default=None, metadata={"init_key": "speaker_encoder", "clone": False}
    )
    _speed_controller: Optional[SpeechSpeedController] = field(
        default=None, metadata={"init_key": "speech_speed_controller", "clone": False}
    )
    embeddings_model: Optional[Embeddings] = field(
        default=None, metadata={"init_key": "embeddings", "clone": False}
    )
    turn_detector_model: Optional[TurnDetector] = field(
        default=None, metadata={"init_key": "turn_detector", "clone": True}
    )

    def __init__(
        self,
        asr: ASR,
        llm_agent: Agent,
        tts: TTS,
        default_response: str = "Sorry, I didn't catch that. Could you please say it again?",
        use_streaming_tts: bool = True,
        captioner: Optional[Captioner] = None,
        punt_restorer_model: Optional[PuntRestorer] = None,
        caption_rewriter: Optional[Rewriter | BaseChatModel] = None,
        vad: Optional[VAD] = None,
        speech_enhancer: Optional[SpeechEnhancer] = None,
        speaker_encoder: Optional[SpeakerEncoder] = None,
        speech_speed_controller: Optional[SpeechSpeedController] = None,
        embeddings: Optional[Embeddings] = None,
        turn_detector: Optional[TurnDetector] = None,
    ):
        """Initialize the default pipeline.

        Parameters
        ----------
        asr : ASR
            Speech recognition model.
        llm_agent : Agent
            Agent used for response generation.
        tts : TTS
            Text-to-speech model.
        default_response : str, optional
            Fallback response text.
        use_streaming_tts : bool, optional
            Whether to prefer streaming TTS paths.
        captioner : Captioner | None, optional
            Optional audio captioning model.
        punt_restorer_model : PuntRestorer | None, optional
            Optional punctuation restoration model.
        caption_rewriter : Rewriter | BaseChatModel | None, optional
            Optional caption rewriter or chat model.
        vad : VAD | None, optional
            Optional voice activity detector.
        speech_enhancer : SpeechEnhancer | None, optional
            Optional speech enhancer.
        speaker_encoder : SpeakerEncoder | None, optional
            Optional speaker encoder.
        speech_speed_controller : SpeechSpeedController | None, optional
            Optional speed controller for TTS output.
        embeddings : Embeddings | None, optional
            Optional embeddings backend.
        turn_detector : TurnDetector | None, optional
            Optional turn detector.
        """
        # Assign directly to dataclass fields
        self.asr_model = asr
        self.llm_agent = llm_agent
        self.tts_model = tts
        self.default_response = default_response
        self.use_streaming_tts = use_streaming_tts
        self.captioner_model = captioner
        self.punt_restorer_model = punt_restorer_model
        self.vad_model = vad
        self.enhancer_model = speech_enhancer
        self.speaker_encoder = speaker_encoder
        self._speed_controller = speech_speed_controller
        self.embeddings_model = embeddings
        self.turn_detector_model = turn_detector

        # Normalize rewriters: BaseChatModel -> Default*Rewriter; Rewriter -> use as-is
        if isinstance(caption_rewriter, BaseChatModel):
            self.caption_rewriter = DefaultCaptionRewriter(model=caption_rewriter)
        else:
            self.caption_rewriter = caption_rewriter

    # --------------------------
    # clone (declarative, extensible, and inheritance-friendly)
    # --------------------------
    def clone(self):
        """Clone the pipeline according to field metadata.

        Returns
        -------
        DefaultPipeline
            New pipeline instance of the same concrete type.

        Notes
        -----
        Fields marked with ``clone=True`` call their own ``clone()`` method when
        available. Remaining fields are shared by reference.
        """
        kwargs: dict[str, Any] = {}

        for f in fields(self):
            init_key = f.metadata.get("init_key")
            if not init_key:
                continue

            value = getattr(self, f.name)
            if f.metadata.get("clone", False):
                value = _maybe_clone(value)

            kwargs[init_key] = value

        return type(self)(**kwargs)

    # --------------------------
    # getters
    # --------------------------
    def get_asr_model(self) -> ASR | None:
        return self.asr_model

    def get_tts_model(self) -> TTS | None:
        return self.tts_model

    def get_agent(self) -> Agent | None:
        return self.llm_agent

    def get_captioner_model(self):
        return self.captioner_model

    def get_punt_restorer_model(self):
        return self.punt_restorer_model

    def get_caption_rewriter_model(self):
        return self.caption_rewriter

    def get_vad_model(self):
        return self.vad_model

    def get_enhancer_model(self):
        return self.enhancer_model

    def get_speaker_encoder(self):
        return self.speaker_encoder

    def get_speed_controller(self) -> SpeechSpeedController | None:
        return self._speed_controller

    def get_embeddings_model(self) -> Embeddings | None:
        return self.embeddings_model

    def get_turn_detector_model(self) -> TurnDetector | None:
        return self.turn_detector_model

    # --------------------------
    # runtime switchers (retain original logic)
    # --------------------------
    def set_tts_model(self, model_type: str, config: dict) -> None:
        """Switch the active TTS model at runtime.

        Parameters
        ----------
        model_type : str
            Supported model family name, currently ``"IndexTTS"`` or
            ``"IndexTTS2"``.
        config : dict
            Runtime configuration passed to the new TTS model constructor.

        Raises
        ------
        ValueError
            Raised if ``model_type`` is not supported.
        """
        from ..models.tts.index_tts import IndexTTS
        from ..models.tts.index_tts2 import IndexTTS2

        current_ref_paths = []
        if self.tts_model and hasattr(self.tts_model, "audio_paths"):
            current_ref_paths = self.tts_model.audio_paths or []
        elif self.tts_model and hasattr(self.tts_model, "_base_audio_paths"):
            current_ref_paths = self.tts_model._base_audio_paths or []

        host = config.get("host", "localhost")
        port = config.get("port")
        ref_audio_paths = config.get("ref_audio_paths") or current_ref_paths
        sample_rate = config.get("sample_rate", 48000)
        timeout = config.get("timeout", 30.0)

        if port is None:
            port = 11996 if model_type == "IndexTTS" else 6006

        if model_type == "IndexTTS":
            self.tts_model = IndexTTS(
                ref_audio_paths=ref_audio_paths,
                host=host,
                port=port,
                sample_rate=sample_rate,
                timeout=timeout,
            )
        elif model_type == "IndexTTS2":
            self.tts_model = IndexTTS2(
                ref_audio_paths=ref_audio_paths,
                host=host,
                port=port,
                sample_rate=sample_rate,
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported TTS model type: {model_type}")

    def set_llm_model(
        self,
        model: str,
        base_url: str = "",
        api_key: str = "",
        extra_body: dict | None = None,
    ) -> None:
        """Replace the current agent LLM with a ``ChatOpenAI`` instance.

        Parameters
        ----------
        model : str
            Target model name passed to ``ChatOpenAI``.
        base_url : str, optional
            Override for the OpenAI-compatible API base URL.
        api_key : str, optional
            API key used for the replacement model. Falls back to
            ``OPENAI_API_KEY`` when omitted.
        extra_body : dict | None, optional
            Additional request payload fields forwarded to ``ChatOpenAI``.
        """
        from langchain_openai import ChatOpenAI
        import os

        kwargs: dict[str, Any] = {"model": model}

        if api_key:
            kwargs["api_key"] = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]

        if base_url:
            kwargs["base_url"] = base_url
        elif os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        if extra_body:
            kwargs["extra_body"] = extra_body

        new_model = ChatOpenAI(**kwargs)

        if self.llm_agent and hasattr(self.llm_agent, "model"):
            self.llm_agent.model = new_model

            if hasattr(self.llm_agent, "_model_with_tools"):
                tools = list(getattr(self.llm_agent, "tools_map", {}).values())
                if tools:
                    try:
                        self.llm_agent._model_with_tools = new_model.bind_tools(tools)
                    except Exception:
                        self.llm_agent._model_with_tools = new_model
                else:
                    self.llm_agent._model_with_tools = new_model
