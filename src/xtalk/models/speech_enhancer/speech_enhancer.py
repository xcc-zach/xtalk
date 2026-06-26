# -*- coding: utf-8 -*-
"""
Speech enhancement module.

Implements streaming enhancement based on the FastEnhancer-S ONNX model:
- Input: 16 kHz PCM s16le audio frames
- Output: enhanced 16 kHz PCM s16le audio frames
- Maintains ONNX cache state for streaming processing
"""

import os
from typing import Optional
import numpy as np
import onnxruntime

from .interfaces import SpeechEnhancer
from ..registry import model


@model
class FastEnhancerS(SpeechEnhancer):
    """Streaming speech enhancer using FastEnhancer-S ONNX."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_fft: int = 512,
        hop_size: int = 256,
        _shared_session: Optional[onnxruntime.InferenceSession] = None,
    ):
        """Initialize the enhancer."""
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.sample_rate = 16000
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
        self._init_cache()
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)
        self.is_first_frame = True
        # Reset sample counters for alignment
        self._total_input_samples = 0
        self._total_output_samples = 0

    def clone(self) -> "FastEnhancerS":
        """Clone enhancer sharing session but keeping independent state."""
        return FastEnhancerS(
            model_path=self.model_path,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            _shared_session=self.session,  # Shared ONNX session
        )

    def enhance(self, pcm_bytes: bytes) -> bytes:
        """Enhance audio frames in streaming mode."""
        if not pcm_bytes:
            return b""

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
        """Flush remaining buffers at the end of the stream."""
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
