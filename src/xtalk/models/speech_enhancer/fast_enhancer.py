"""FastEnhancer speech enhancement module.

Implements streaming enhancement based on the FastEnhancer-S ONNX model:
- Input: 16 kHz PCM s16le audio frames
- Output: enhanced 16 kHz PCM s16le audio frames
- Maintains ONNX cache state for streaming processing
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Optional

import numpy as np
import requests

try:
    import onnxruntime
except ImportError:  # pragma: no cover - remote mode does not need ONNX Runtime.
    onnxruntime = None  # type: ignore[assignment]

from .interfaces import SpeechEnhancer
from ..registry import model


@model(aliases=["FastEnhancerS", "speech_enhancer"])
class FastEnhancer(SpeechEnhancer):
    """Streaming or remote speech enhancer using FastEnhancer.

    Notes
    -----
    When ``base_url`` is provided, all other initialization parameters are
    ignored and audio is enhanced by the FastEnhancer HTTP service at
    ``POST /v1/enhance/pcm``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_fft: int = 512,
        hop_size: int = 256,
        _shared_session: Optional[object] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the enhancer.

        Parameters
        ----------
        model_path : str | None, optional
            Local ONNX model path for in-process enhancement.
        n_fft : int, optional
            FFT size used for delay compensation in local mode.
        hop_size : int, optional
            Number of samples processed per local ONNX step.
        _shared_session : object | None, optional
            Existing ONNX Runtime session reused by local clones.
        base_url : str | None, optional
            Base URL of a running FastEnhancer HTTP service. When set,
            ``model_path``, ``n_fft``, ``hop_size``, and ``_shared_session`` are
            ignored and no local ONNX model is loaded.
        """
        self.base_url = base_url.rstrip("/") if base_url else None
        self.sample_rate = 16000
        if self.base_url is not None:
            self.model_path = None
            return

        self.n_fft = n_fft
        self.hop_size = hop_size
        self.model_path = model_path

        # Reuse shared session if provided
        if _shared_session is not None:
            self.session = _shared_session
        else:
            # Otherwise create a new session
            self._init_session(model_path)

        # Cache state per instance
        self._init_cache()

        # Input/output buffers per instance
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)

        # Flag for first frame (requires special padding)
        self.is_first_frame = True

        # Track total samples for tail padding/alignment
        self._total_input_samples = 0
        self._total_output_samples = 0

    def _init_session(self, model_path: Optional[str]) -> None:
        """Initialize ONNX Runtime session (only during first creation)."""
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for local FastEnhancer mode")

        # Resolve default model paths relative to this file
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base_dir, "fastenhancer_s.onnx"),
                os.path.join(base_dir, "model", "fastenhancer_s.onnx"),
                os.path.normpath(
                    os.path.join(
                        base_dir,
                        "..",
                        "..",
                        "..",
                        "..",
                        "frontend",
                        "src",
                        "fastenhancer_s.onnx",
                    )
                ),
            ]
            model_path = next(
                (p for p in candidates if os.path.exists(p)), candidates[0]
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Speech enhancer model not found: {model_path}")

        self.model_path = model_path

        # Create ONNX Runtime session
        sess_options = onnxruntime.SessionOptions()
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def _init_cache(self) -> None:
        """Initialize per-instance cache state."""
        self.cache_inputs = {
            x.name: np.zeros(x.shape, dtype=np.float32)
            for x in self.session.get_inputs()
            if x.name.startswith("cache_in_")
        }

    def reset(self) -> None:
        """Reset enhancer state."""
        if self.base_url is not None:
            return

        self._init_cache()
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)
        self.is_first_frame = True
        # Reset sample counters for alignment
        self._total_input_samples = 0
        self._total_output_samples = 0

    def clone(self) -> "FastEnhancer":
        """Clone enhancer sharing local weights or remote service settings.

        Returns
        -------
        FastEnhancer
            Clone with isolated runtime state.
        """
        if self.base_url is not None:
            return FastEnhancer(base_url=self.base_url)

        return FastEnhancer(
            model_path=self.model_path,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            _shared_session=self.session,  # Shared ONNX session
        )

    def enhance(self, pcm_bytes: bytes) -> bytes:
        """Enhance audio frames in streaming mode.

        Parameters
        ----------
        pcm_bytes : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        bytes
            Enhanced PCM 16-bit mono audio bytes at 16 kHz.
        """
        if not pcm_bytes:
            return b""
        if self.base_url is not None:
            return self._enhance_remote(pcm_bytes)

        # Convert to float32 [-1, 1]
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0
        pcm_float = np.clip(pcm_float, a_min=-1.0, a_max=1.0)

        input_len = len(pcm_int16)
        self._total_input_samples += input_len

        # Append to input buffer
        self.input_buffer = np.concatenate([self.input_buffer, pcm_float])

        # Process frames by hop size
        while len(self.input_buffer) >= self.hop_size:
            frame_in = self.input_buffer[: self.hop_size].reshape(1, -1)
            self.input_buffer = self.input_buffer[self.hop_size :]

            # ONNX inference
            self.cache_inputs["wav_in"] = frame_in
            outputs = self.session.run(None, self.cache_inputs)

            # Update cache state
            for j in range(len(outputs) - 1):
                self.cache_inputs[f"cache_in_{j}"] = outputs[j + 1]

            # Append output frame
            frame_out = outputs[0][0]
            self.output_buffer = np.concatenate([self.output_buffer, frame_out])

        # Drop first (n_fft - hop_size) samples for delay compensation
        if self.is_first_frame and len(self.output_buffer) >= (
            self.n_fft - self.hop_size
        ):
            self.output_buffer = self.output_buffer[self.n_fft - self.hop_size :]
            self.is_first_frame = False

        # Extract same-length audio from output buffer
        if len(self.output_buffer) < input_len:
            # Zero-pad when there is not enough output to maintain length
            output_samples = np.concatenate(
                [
                    self.output_buffer,
                    np.zeros(input_len - len(self.output_buffer), dtype=np.float32),
                ]
            )
            self.output_buffer = np.array([], dtype=np.float32)
        else:
            output_samples = self.output_buffer[:input_len]
            self.output_buffer = self.output_buffer[input_len:]

        self._total_output_samples += len(output_samples)

        # Convert back to s16le
        output_samples = np.clip(output_samples, a_min=-1.0, a_max=1.0)
        output_int16 = (output_samples * 32768.0).astype(np.int16)
        return output_int16.tobytes()

    def flush(self) -> bytes:
        """Flush remaining buffers at the end of the stream.

        Returns
        -------
        bytes
            Remaining enhanced PCM audio bytes.
        """
        if self.base_url is not None:
            return b""

        # Pad silence to process leftover input (similar to official pad-right)
        padding_needed = self.n_fft
        padding = np.zeros(padding_needed, dtype=np.float32)
        self.input_buffer = np.concatenate([self.input_buffer, padding])

        # Process any remaining frames
        while len(self.input_buffer) >= self.hop_size:
            frame_in = self.input_buffer[: self.hop_size].reshape(1, -1)
            self.input_buffer = self.input_buffer[self.hop_size :]

            self.cache_inputs["wav_in"] = frame_in
            outputs = self.session.run(None, self.cache_inputs)

            for j in range(len(outputs) - 1):
                self.cache_inputs[f"cache_in_{j}"] = outputs[j + 1]

            frame_out = outputs[0][0]
            self.output_buffer = np.concatenate([self.output_buffer, frame_out])

        # Apply first-frame compensation if it didn't trigger before
        if self.is_first_frame and len(self.output_buffer) >= (
            self.n_fft - self.hop_size
        ):
            self.output_buffer = self.output_buffer[self.n_fft - self.hop_size :]
            self.is_first_frame = False

        # Drain remaining output without exceeding input length
        remaining_needed = self._total_input_samples - self._total_output_samples
        if remaining_needed <= 0:
            return b""

        output_len = min(len(self.output_buffer), remaining_needed)
        if output_len <= 0:
            return b""

        output_samples = self.output_buffer[:output_len]
        self.output_buffer = self.output_buffer[output_len:]
        self._total_output_samples += len(output_samples)

        output_samples = np.clip(output_samples, a_min=-1.0, a_max=1.0)
        output_int16 = (output_samples * 32768.0).astype(np.int16)
        return output_int16.tobytes()

    def _enhance_remote(self, pcm_bytes: bytes) -> bytes:
        """Enhance one PCM payload by calling the FastEnhancer HTTP service."""
        response = requests.post(
            f"{self.base_url}/v1/enhance/pcm",
            params={"input_dtype": "int16", "response_format": "pcm"},
            data=pcm_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        response.raise_for_status()
        return response.content



def _is_tcp_port_open(host: str, port: int, timeout: float) -> bool:
    """Return whether a TCP listener accepts connections at host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _build_test_pcm(sample_rate: int, duration_seconds: float) -> bytes:
    """Build a small PCM s16le test tone for the remote smoke test."""
    sample_count = int(sample_rate * duration_seconds)
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    samples = 0.2 * np.sin(2.0 * np.pi * 440.0 * time_axis)
    return (samples * 32767.0).astype(np.int16).tobytes()


def _run_remote_smoke_test() -> int:
    """Run a simple client request against a running FastEnhancer service."""
    parser = argparse.ArgumentParser(
        description="Smoke test the FastEnhancer HTTP client."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--duration-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if not _is_tcp_port_open(args.host, args.port, timeout=2.0):
        print(
            "No service is listening on "
            f"{args.host}:{args.port}. Please start fastenhancer first.",
            file=sys.stderr,
        )
        return 2

    base_url = f"http://{args.host}:{args.port}"
    enhancer = FastEnhancer(base_url=base_url)
    pcm_bytes = _build_test_pcm(enhancer.sample_rate, args.duration_seconds)
    try:
        enhanced_bytes = enhancer.enhance(pcm_bytes)
    except requests.RequestException as exc:
        print(
            f"FastEnhancer client request failed for {base_url}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not enhanced_bytes or len(enhanced_bytes) % 2 != 0:
        print(
            "FastEnhancer client request failed: invalid PCM response "
            f"({len(enhanced_bytes)} bytes).",
            file=sys.stderr,
        )
        return 1

    print(
        "FastEnhancer client request succeeded: "
        f"input_bytes={len(pcm_bytes)}, output_bytes={len(enhanced_bytes)}, "
        f"base_url={base_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_remote_smoke_test())
