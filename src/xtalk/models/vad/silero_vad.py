"""Silero VAD backed by a shared ONNX Runtime session.

Each process creates a single ONNX session that is shared by all
``SileroVAD`` instances. Every instance keeps its own streaming model state,
context window, and pending audio buffer so concurrent sessions do not leak
state into each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import tempfile
import threading
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort

from .interfaces import VAD
from ..registry import model


SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64

SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/"
    "silero_vad/data/silero_vad.onnx"
)
_CACHE_SUBDIR = "xtalk/models"
_CACHE_FILENAME = "silero_vad.onnx"

_MODEL_FILE_LOCK = threading.Lock()
_SESSION_LOCK = threading.Lock()
_SHARED_SESSION: ort.InferenceSession | None = None
_SHARED_MODEL_PATH: str | None = None


def _user_cache_dir() -> Path:
    """Return the per-user cache directory for xtalk assets."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / _CACHE_SUBDIR


def _ensure_cached_model() -> Path:
    """Download the Silero VAD ONNX model into the user cache if needed."""
    cache_dir = _user_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / _CACHE_FILENAME
    if model_path.exists():
        return model_path

    with _MODEL_FILE_LOCK:
        if model_path.exists():
            return model_path

        temp_path: Path | None = None
        try:
            with urlopen(SILERO_VAD_ONNX_URL) as response:
                with tempfile.NamedTemporaryFile(
                    dir=cache_dir,
                    prefix="silero_vad.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    temp_file.write(response.read())

            assert temp_path is not None
            temp_path.replace(model_path)
            return model_path
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise


def _resolve_model_path(model_path: str | None) -> Path:
    """Resolve the Silero VAD ONNX model path."""
    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {resolved}")
        return resolved

    return _ensure_cached_model().resolve()


def _create_session(model_path: Path) -> ort.InferenceSession:
    """Create the shared ONNX Runtime session."""
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _get_shared_session(model_path: str | None) -> tuple[ort.InferenceSession, str]:
    """Return the process-wide ONNX session for Silero VAD."""
    global _SHARED_MODEL_PATH, _SHARED_SESSION

    resolved_path = str(_resolve_model_path(model_path))
    with _SESSION_LOCK:
        if (
            _SHARED_SESSION is None
            or _SHARED_MODEL_PATH is None
            or _SHARED_MODEL_PATH != resolved_path
        ):
            _SHARED_SESSION = _create_session(Path(resolved_path))
            _SHARED_MODEL_PATH = resolved_path
        assert _SHARED_SESSION is not None
        return _SHARED_SESSION, resolved_path


@dataclass
class _SileroStreamState:
    """Per-instance streaming state for the Silero VAD model."""

    model_state: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 1, 128), dtype=np.float32)
    )
    context: np.ndarray = field(
        default_factory=lambda: np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
    )
    pending_audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )

    def reset(self) -> None:
        """Reset per-stream model memory and buffered audio."""
        self.model_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self.pending_audio = np.zeros((0,), dtype=np.float32)


@model
class SileroVAD(VAD):
    """Silero VAD implementation with shared model session and per-instance state."""

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: str | None = None,
        frame_samples: int = FRAME_SAMPLES,
        context_samples: int = CONTEXT_SAMPLES,
    ) -> None:
        """Initialize the VAD instance.

        Parameters
        ----------
        threshold : float, optional
            Speech probability threshold used to convert the model output into a
            boolean decision.
        model_path : str | None, optional
            Optional path to the Silero VAD ONNX model. When omitted, several
            common repository-local paths are tried.
        frame_samples : int, optional
            Expected frame size in samples for model inference. Current Silero
            ONNX models typically expect 512 samples at 16 kHz.
        context_samples : int, optional
            Number of trailing samples from the previous frame to prepend as
            model context on the next inference call.
        """
        self.threshold = float(threshold)
        self.frame_samples = int(frame_samples)
        self.context_samples = int(context_samples)
        self.session, self.model_path = _get_shared_session(model_path)
        self._inference_lock = threading.Lock()
        self._state = _SileroStreamState()
        self._state.context = np.zeros(
            (1, self.context_samples),
            dtype=np.float32,
        )

    def clone(self) -> "SileroVAD":
        """Clone the VAD while reusing the shared ONNX session."""
        return SileroVAD(
            threshold=self.threshold,
            model_path=self.model_path,
            frame_samples=self.frame_samples,
            context_samples=self.context_samples,
        )

    def reset(self) -> None:
        """Reset this instance's streaming state."""
        self._state.reset()
        if self.context_samples != CONTEXT_SAMPLES:
            self._state.context = np.zeros(
                (1, self.context_samples),
                dtype=np.float32,
            )

    def is_speech(self, frame: bytes) -> bool:
        """Determine whether the latest complete frame contains speech.

        Parameters
        ----------
        frame : bytes
            PCM 16-bit mono audio bytes at 16 kHz. Input may contain any number
            of samples; the instance buffers partial data internally and only
            runs inference on complete ``frame_samples`` windows.

        Returns
        -------
        bool
            ``True`` when the most recent complete frame has speech
            probability greater than or equal to ``threshold``. If not enough
            audio is available to form a complete frame, returns ``False``.
        """
        if not frame:
            return False

        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return False

        self._state.pending_audio = np.concatenate([self._state.pending_audio, audio])
        last_prob: float | None = None

        while self._state.pending_audio.shape[0] >= self.frame_samples:
            chunk = self._state.pending_audio[: self.frame_samples]
            self._state.pending_audio = self._state.pending_audio[self.frame_samples :]
            last_prob = self._infer_one_frame(chunk)

        if last_prob is None:
            return False
        return last_prob >= self.threshold

    def _infer_one_frame(self, frame: np.ndarray) -> float:
        """Run ONNX inference for one complete frame."""
        if frame.shape[0] != self.frame_samples:
            raise ValueError(f"frame must have {self.frame_samples} samples")

        x = frame.reshape(1, -1).astype(np.float32, copy=False)
        x = np.concatenate([self._state.context, x], axis=1)
        ort_inputs = {
            "input": x,
            "state": self._state.model_state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }

        with self._inference_lock:
            out, new_state = self.session.run(None, ort_inputs)

        self._state.model_state = new_state
        self._state.context = x[:, -self.context_samples :].astype(
            np.float32,
            copy=False,
        )
        return float(np.asarray(out).squeeze())
