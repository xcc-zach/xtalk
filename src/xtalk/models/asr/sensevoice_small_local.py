from __future__ import annotations

"""
SenseVoiceSmall local ASR implementation built on FunASR.

Notes:
- SenseVoiceSmall does not support incremental chunk streaming. To fit the
  `recognize_stream` interface, this implementation buffers incoming audio
  and returns an empty string when `is_final=False`, then performs one-shot
  recognition over the accumulated audio when `is_final=True`.

Input: PCM 16-bit mono 16 kHz raw bytes.
"""

from typing import Any, Optional, Dict

import numpy as np

from .interfaces import ASR
from ...log_utils import logger
from ..audio_utils import MockStreamRecognizer
from ..registry import model

try:  # FunASR dependency
    from funasr import AutoModel  # type: ignore
    from funasr.utils.postprocess_utils import (  # type: ignore
        rich_transcription_postprocess,
    )
except Exception as e:  # pragma: no cover - early failure when dependency missing
    AutoModel = None  # type: ignore
    rich_transcription_postprocess = None  # type: ignore
    _import_error = e
else:
    _import_error = None


@model
class SenseVoiceSmallLocal(ASR):
    """SenseVoiceSmall ASR wrapper."""

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        model: str = "FunAudioLLM/SenseVoiceSmall",
        device: str = "cpu",
        hub: str = "hf",
        disable_update: bool = True,
        # VAD parameters (for long-form segmentation/merge)
        vad_model: Optional[str] = "fsmn-vad",
        vad_kwargs: Optional[Dict[str, Any]] = None,
        merge_vad: bool = True,
        merge_length_s: int = 15,
        # Recognition and post-processing parameters
        language: str = "auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn: bool = True,
        batch_size_s: int = 60,
        ban_emo_unk: bool = False,
        mock_window_size: int = 5,
        # Internal parameters
        _shared_model: Optional[Any] = None,  # shared model instance for clone()
        # Additional passthrough kwargs
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize the SenseVoiceSmall model.

        Refer to the FunASR docs for parameter descriptions; common values use
        the provided defaults.
        """

        if AutoModel is None:  # pragma: no cover - dependency missing
            raise ImportError(
                f"funasr is required for SenseVoiceSmall: {_import_error}"
            )

        self.model_name = model
        self.device = device
        self.hub = hub
        self.disable_update = bool(disable_update)

        # VAD / merge
        self.vad_model = vad_model
        self.vad_kwargs = vad_kwargs or {"max_single_segment_time": 30000}
        self.merge_vad = bool(merge_vad)
        self.merge_length_s = int(merge_length_s)

        # Inference parameters
        self.language = language
        self.use_itn = bool(use_itn)
        self.batch_size_s = int(batch_size_s)
        self.ban_emo_unk = bool(ban_emo_unk)
        self.extra_kwargs: Dict[str, Any] = dict(kwargs)

        # Initialize FunASR model (reuse shared instance if provided)
        if _shared_model is not None:
            self.model = _shared_model
        else:
            model_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "device": self.device,
                "hub": self.hub,
                "disable_update": self.disable_update,
            }
            if self.vad_model:
                model_kwargs["vad_model"] = self.vad_model
                model_kwargs["vad_kwargs"] = dict(self.vad_kwargs)
            if self.extra_kwargs:
                model_kwargs.update(self.extra_kwargs)

            self.model = AutoModel(**model_kwargs)  # type: ignore[call-arg]
        # Use the async recognize function to drive MockStreamRecognizer for future async support
        self._mock_recognizer = MockStreamRecognizer(
            self.async_recognize, window_size=mock_window_size
        )

    # ------------- Public ASR interface -------------
    def recognize(self, audio: bytes) -> str:
        """Recognize the entire audio clip at once."""
        if not audio:
            return ""
        try:
            speech = self._pcm_to_float(audio)
            result = self.model.generate(
                input=speech,
                cache={},
                language=self.language,
                use_itn=self.use_itn,
                batch_size_s=self.batch_size_s,
                merge_vad=self.merge_vad,
                merge_length_s=self.merge_length_s,
                ban_emo_unk=self.ban_emo_unk,
            )
            text = self._parse_text(result)
            return text
        except Exception as e:
            raise RuntimeError(f"SenseVoiceSmall recognize failed: {e}")

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """
        Pseudo streaming: each call runs one-shot recognition on buffered audio.
        """
        del chat_history
        if not audio:
            return self._mock_recognizer.recognized_text
        return self._mock_recognizer.recognize(audio, is_final=is_final)

    def stream_chunk_bytes_hint(self) -> int | None:
        """
        Hint the upstream logic about how often to trigger streaming recognition.
        """
        return 19200  # trigger every 19200 bytes

    def reset(self) -> None:
        """Reset internal streaming state."""
        self._mock_recognizer.reset()

    def clone(self) -> "SenseVoiceSmall":
        """Create a clone that shares the model while keeping independent state."""
        return SenseVoiceSmallLocal(
            model=self.model_name,
            device=self.device,
            hub=self.hub,
            disable_update=self.disable_update,
            vad_model=self.vad_model,
            vad_kwargs=self.vad_kwargs,
            merge_vad=self.merge_vad,
            merge_length_s=self.merge_length_s,
            language=self.language,
            use_itn=self.use_itn,
            batch_size_s=self.batch_size_s,
            ban_emo_unk=self.ban_emo_unk,
            _shared_model=self.model,  # shared model instance
            **self.extra_kwargs,
        )

    # ------------- Internal helpers -------------
    def _pcm_to_float(self, pcm: bytes) -> np.ndarray:
        """Convert PCM int16 bytes to float32 samples in [-1, 1]."""
        if not pcm:
            return np.zeros((0,), dtype=np.float32)
        x = np.frombuffer(pcm, dtype=np.int16)
        return x.astype(np.float32) / 32768.0

    def _parse_text(self, result: Any) -> str:
        """Parse FunASR output and apply rich transcription post-processing."""
        text = ""
        try:
            if isinstance(result, list) and result:
                item0 = result[0]
                if isinstance(item0, dict) and "text" in item0:
                    text = str(item0.get("text", ""))
                else:
                    text = str(item0)
            elif isinstance(result, dict) and "text" in result:
                text = str(result.get("text", ""))
            else:
                text = str(result)
        except Exception as e:  # Fallback guard for malformed data
            logger.warning("SenseVoiceSmall parse result failed: %s", e)
            text = ""

        # Rich-text post-processing if available
        if rich_transcription_postprocess is not None and text:
            try:
                text = rich_transcription_postprocess(text)
            except Exception as e:
                logger.debug("postprocess failed, fallback to raw text: %s", e)
        return text
