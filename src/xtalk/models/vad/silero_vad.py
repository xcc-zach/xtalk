"""Silero VAD backed by a shared ONNX Runtime session.

Each process creates a single ONNX session that is shared by all
``SileroVAD`` instances. Every instance keeps its own streaming model state,
context window, and pending audio buffer so concurrent sessions do not leak
state into each other.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import socket
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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
_REMOTE_TIMEOUT_SECONDS = 10.0
_REMOTE_VAD_QUERY = urlencode(
    {
        "sample_rate": SAMPLE_RATE,
        "encoding": "pcm_s16le",
        "channels": 1,
        "min_speech_duration_ms": 0,
        "min_silence_duration_ms": 0,
        "speech_pad_ms": 0,
        "return_seconds": "true",
    }
)

_MODEL_FILE_LOCK = threading.Lock()
_SESSION_LOCK = threading.Lock()
_SHARED_SESSION: Any | None = None
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
    if ort is None:
        raise RuntimeError(
            "onnxruntime is required for local Silero VAD inference; "
            "install xtalk[silero-vad] or pass base_url to use a remote service"
        )

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


def _resolve_remote_vad_url(base_url: str) -> tuple[str, str]:
    """Resolve the remote Silero VAD base URL and inference URL."""
    normalized_base_url = base_url.strip().rstrip("/")
    if not normalized_base_url:
        raise ValueError("base_url must not be empty")
    if normalized_base_url.endswith("/v1/vad"):
        endpoint_url = normalized_base_url
    else:
        endpoint_url = f"{normalized_base_url}/v1/vad"
    return normalized_base_url, f"{endpoint_url}?{_REMOTE_VAD_QUERY}"


def _remote_response_has_speech(payload: dict[str, Any]) -> bool:
    """Return whether a remote Silero VAD response contains speech segments."""
    segments = payload["segments"]
    if not isinstance(segments, list):
        raise ValueError("remote VAD response field 'segments' must be a list")
    return bool(segments)


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
        base_url: str | None = None,
    ) -> None:
        """Initialize the VAD instance.

        Parameters
        ----------
        base_url : str | None, optional
            Optional base URL of an external Silero VAD HTTP service. When set,
            this instance sends frames to ``/v1/vad`` and ignores all local
            inference parameters.
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
        self.base_url: str | None = None
        self._remote_vad_url: str | None = None
        if base_url is not None:
            self.base_url, self._remote_vad_url = _resolve_remote_vad_url(base_url)
            return

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
        if self.base_url is not None:
            return SileroVAD(base_url=self.base_url)

        return SileroVAD(
            threshold=self.threshold,
            model_path=self.model_path,
            frame_samples=self.frame_samples,
            context_samples=self.context_samples,
        )

    def reset(self) -> None:
        """Reset this instance's streaming state."""
        if self.base_url is not None:
            return

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

        if self.base_url is not None:
            return self._is_speech_remote(frame)

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

    def _is_speech_remote(self, frame: bytes) -> bool:
        """Send a PCM frame to the remote Silero VAD service."""
        if self._remote_vad_url is None:
            raise RuntimeError("remote VAD URL is not configured")

        request = Request(
            self._remote_vad_url,
            data=frame,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urlopen(request, timeout=_REMOTE_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return _remote_response_has_speech(payload)



def _is_tcp_port_open(host: str, port: int, timeout: float) -> bool:
    """Return whether a TCP listener accepts connections at host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _build_test_pcm(sample_rate: int, duration_seconds: float) -> bytes:
    """Build a small PCM s16le test tone for the remote VAD smoke test."""
    sample_count = int(sample_rate * duration_seconds)
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    samples = 0.2 * np.sin(2.0 * np.pi * 440.0 * time_axis)
    return (samples * 32767.0).astype(np.int16).tobytes()


def _run_remote_smoke_test() -> int:
    """Run a simple client request against a running Silero VAD service."""
    parser = argparse.ArgumentParser(description="Smoke test the Silero VAD HTTP client.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--duration-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if not _is_tcp_port_open(args.host, args.port, timeout=2.0):
        print(
            "No service is listening on "
            f"{args.host}:{args.port}. Please start silerovad first.",
            file=sys.stderr,
        )
        return 2

    base_url = f"http://{args.host}:{args.port}"
    vad = SileroVAD(base_url=base_url)
    pcm_bytes = _build_test_pcm(SAMPLE_RATE, args.duration_seconds)
    try:
        is_speech = vad.is_speech(pcm_bytes)
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"Silero VAD client request failed for {base_url}: {exc}", file=sys.stderr)
        return 1

    print(
        "Silero VAD client request succeeded: "
        f"input_bytes={len(pcm_bytes)}, is_speech={is_speech}, base_url={base_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_remote_smoke_test())

