from __future__ import annotations

from typing import Optional
import os
import threading

import numpy as np
import yaml
import sherpa_onnx

from ..interfaces import ASR


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    """
    Convert PCM 16-bit bytes into a float32 array (-1~1).
    """
    if not pcm:
        return np.zeros((0,), dtype=np.float32)
    x = np.frombuffer(pcm, dtype=np.int16)
    return x.astype(np.float32) / 32768.0


class ZipformerLocal(ASR):
    """
    Local Zipformer ASR implementation built on sherpa_onnx.

    Input: PCM 16-bit mono 16 kHz raw bytes.

    Design constraints:
    - All ZipformerLocal instances (including clones) share one
      sherpa_onnx.OnlineRecognizer instance.
    - Class-level mutexes ensure only one thread uses the recognizer at a time
      to avoid segfaults caused by concurrent decode_streams/get_result calls.
    - Each instance owns its own OnlineStream and text buffer state.
    """

    # Class-level shared OnlineRecognizer and locks
    _init_lock = threading.Lock()  # synchronize shared recognizer init
    _recognizer_lock = threading.Lock()  # serialize recognizer usage
    _shared_recognizer: Optional[sherpa_onnx.OnlineRecognizer] = None  # type: ignore
    _shared_model_dir: Optional[str] = None

    def __init__(
        self,
        model_dir: str | None = None,
        tail_padding_sec: float = 0.66,
    ):
        """
        Initialize the Zipformer ASR model.

        Args:
            model_dir: Directory containing model files.
            tail_padding_sec: Tail padding seconds to flush the internal cache.
        """
        self.model_dir = model_dir
        self.tail_padding_sec = tail_padding_sec

        # All instances share the same lock to serialize recognizer usage
        self._lock = ZipformerLocal._recognizer_lock

        # Parse config and lazily load the shared OnlineRecognizer
        self._init_from_model_dir(model_dir)

        # Streaming state (per instance)
        self._stream: Optional[sherpa_onnx.OnlineStream] = None  # type: ignore
        self._total_result: str = ""

    # ------------------------------------------------------------------
    # Internal helper: access the shared recognizer
    # ------------------------------------------------------------------
    @property
    def _recognizer(self) -> sherpa_onnx.OnlineRecognizer:  # type: ignore
        r = ZipformerLocal._shared_recognizer
        if r is None:
            raise RuntimeError("ZipformerLocal: shared OnlineRecognizer is not ready")
        return r

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_from_model_dir(self, model_dir: str | None) -> None:
        """Load config from the model directory and lazily init the shared recognizer."""

        # Download from ModelScope if model_dir is missing
        if not model_dir or not os.path.isdir(model_dir):
            try:
                from modelscope import snapshot_download
            except ImportError as e:
                raise RuntimeError(
                    "model_dir does not exist and modelscope is unavailable for download"
                ) from e

            model_dir = snapshot_download("yhdai666/xtalk_zipformer_onnx")

        self.model_dir = model_dir
        cfg_path = os.path.join(model_dir, "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["asr"]

        model_cfg = cfg["model"]
        dec_cfg = cfg["decoding"]
        lm_cfg = cfg.get("lm", {})
        lodr_cfg = cfg.get("lodr", {})
        hot_cfg = cfg.get("hotwords", {})
        lang_cfg = cfg.get("lang", {})
        pen_cfg = cfg.get("penalty", {})

        # Instance-level parameters (used for chunk/hint calculations)
        self.TARGET_SAMPLE_RATE = dec_cfg.get("sample_rate", 16000)
        self.chunk_secs = dec_cfg.get("chunk_secs", 1.0)
        self.tail_padding_sec = pen_cfg.get("tail_padding_sec", self.tail_padding_sec)

        # Lazily create or reuse the shared recognizer
        with ZipformerLocal._init_lock:
            if ZipformerLocal._shared_recognizer is None:
                # Create the shared OnlineRecognizer once
                ZipformerLocal._shared_recognizer = (
                    sherpa_onnx.OnlineRecognizer.from_transducer(
                        tokens=os.path.join(model_dir, model_cfg["tokens"]),
                        encoder=os.path.join(model_dir, model_cfg["encoder"]),
                        decoder=os.path.join(model_dir, model_cfg["decoder"]),
                        joiner=os.path.join(model_dir, model_cfg["joiner"]),
                        num_threads=model_cfg.get("num_threads", 1),
                        provider=model_cfg.get("provider", "cpu"),
                        sample_rate=self.TARGET_SAMPLE_RATE,
                        feature_dim=80,
                        decoding_method=dec_cfg["method"],
                        max_active_paths=dec_cfg["max_active_paths"],
                        lm=lm_cfg.get("lm_path", ""),
                        lm_scale=lm_cfg.get("lm_scale", 0.1),
                        lodr_fst=lodr_cfg.get("fst", ""),
                        lodr_scale=lodr_cfg.get("scale", -0.1),
                        hotwords_file=hot_cfg.get("file", ""),
                        hotwords_score=hot_cfg.get("score", 1.5),
                        modeling_unit=lang_cfg.get("modeling_unit", ""),
                        bpe_vocab=os.path.join(
                            model_dir,
                            lang_cfg.get("bpe_vocab", ""),
                        ),
                        blank_penalty=pen_cfg.get("blank_penalty", 0.0),
                    )
                )
                ZipformerLocal._shared_model_dir = model_dir
            else:
                # Ensure new instances reuse the same model_dir as the shared recognizer
                if (
                    ZipformerLocal._shared_model_dir is not None
                    and ZipformerLocal._shared_model_dir != model_dir
                ):
                    raise RuntimeError(
                        f"ZipformerLocal already initialized with model_dir="
                        f"{ZipformerLocal._shared_model_dir!r}; cannot reuse "
                        f"a different model_dir={model_dir!r}."
                    )

    # ------------------------------------------------------------------
    # Public ASR interface
    # ------------------------------------------------------------------
    def recognize(self, audio: bytes) -> str:
        """Recognize the entire audio buffer at once."""
        if not audio:
            return ""

        try:
            speech = _pcm_to_float(audio)

            with self._lock:
                # Create a temporary stream from the shared recognizer
                stream = self._recognizer.create_stream()

                # Feed the main audio
                stream.accept_waveform(self.TARGET_SAMPLE_RATE, speech)

                # Tail padding flushes the model state
                if self.tail_padding_sec > 0:
                    pad = np.zeros(
                        int(self.tail_padding_sec * self.TARGET_SAMPLE_RATE),
                        dtype=np.float32,
                    )
                    stream.accept_waveform(self.TARGET_SAMPLE_RATE, pad)

                stream.input_finished()

                # Serial decode + get_result
                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_streams([stream])
                final_text = self._recognizer.get_result(stream)

            return final_text
        except Exception as e:
            raise RuntimeError(f"ZipformerLocal recognize failed: {e}")

    def recognize_stream(self, audio: bytes, *, is_final: bool = False) -> str:
        """
        Streaming recognition interface (incremental input).

        Args:
            audio: Incremental PCM 16-bit mono 16 kHz bytes.
            is_final: When True, add tail padding, mark input finished, and flush.
        Returns:
            Accumulated recognized text (may be empty).

        Note: is_final=True does not automatically reset the instance; call reset()
        externally before starting a new session.
        """
        try:
            speech = _pcm_to_float(audio)

            with self._lock:
                # Initialize a per-instance stream when needed
                if self._stream is None:
                    self._stream = self._recognizer.create_stream()
                    self._total_result = ""

                # Feed audio
                if speech.size > 0:
                    self._stream.accept_waveform(self.TARGET_SAMPLE_RATE, speech)

                # When final, add padding and finish input
                if is_final:
                    if self.tail_padding_sec > 0:
                        pad = np.zeros(
                            int(self.tail_padding_sec * self.TARGET_SAMPLE_RATE),
                            dtype=np.float32,
                        )
                        self._stream.accept_waveform(self.TARGET_SAMPLE_RATE, pad)
                    self._stream.input_finished()

                # Serial decode + get_result
                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_streams([self._stream])
                result = self._recognizer.get_result(self._stream)

                if is_final:
                    # Flush as much as possible when final
                    while self._recognizer.is_ready(self._stream):
                        self._recognizer.decode_streams([self._stream])
                    result = self._recognizer.get_result(self._stream)

                self._total_result = result
                return self._total_result

        except Exception:
            # Reset local state on failure to avoid stuck streams
            self.reset()
            raise

    # ------------------------------------------------------------------
    # Helper interfaces
    # ------------------------------------------------------------------
    def stream_chunk_bytes_hint(self) -> int | None:
        """
        Return the suggested streaming trigger size in bytes.

        PCM 16-bit mono 16 kHz:
        2 bytes/sample * 16000 samples/sec = 32000 bytes/sec
        """
        if getattr(self, "chunk_secs", None) and self.chunk_secs > 0:
            return int(self.TARGET_SAMPLE_RATE * 2 * self.chunk_secs)
        return None

    def reset(self) -> None:
        """Reset streaming state; call before starting a new session."""
        with self._lock:
            self._stream = None
            self._total_result = ""

    def clone(self) -> "ZipformerLocal":
        """
        Create a copy of the ASR instance.

        Requirements:
        1. Share the same OnlineRecognizer (class-level _shared_recognizer).
        2. Keep independent runtime state (OnlineStream, cached text, etc.).
        3. Allow concurrent sessions with the class-level mutex ensuring only one
           thread uses the shared recognizer at a time.
        """
        # __init__ guarantees shared recognizer reuse (lazy load happens once)
        return ZipformerLocal(
            model_dir=self.model_dir,
            tail_padding_sec=self.tail_padding_sec,
        )
