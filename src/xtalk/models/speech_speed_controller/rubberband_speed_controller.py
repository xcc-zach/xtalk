# -*- coding: utf-8 -*-
"""TTS speed controller using pyrubberband time-stretching."""
import numpy as np

from .interfaces import SpeechSpeedController
from ..registry import model

try:
    import pyrubberband as pyrb

    PYRUBBERBAND_AVAILABLE = True
except ImportError:
    PYRUBBERBAND_AVAILABLE = False


@model
class RubberbandSpeedController(SpeechSpeedController):
    """Adjust audio speed via time-stretching while preserving pitch."""

    def __init__(self, sample_rate: int = 48000):
        """Initialize speed controller."""
        self.sample_rate = sample_rate
        self.enabled = PYRUBBERBAND_AVAILABLE

        if not self.enabled:
            import warnings

            warnings.warn(
                "pyrubberband is not installed. Audio speed control will be disabled. "
                "Install it with: pip install pyrubberband",
                ImportWarning,
            )

    def process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes:
        """Apply time-stretch with optional speed change."""
        # Fast path when no processing is needed
        if not self.enabled or speed == 1.0 or not audio_bytes:
            return audio_bytes

        # Clamp speed range
        speed = max(0.5, min(1.5, speed))

        try:
            # Step 1: convert bytes (int16) to numpy array (mono assumed)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # Skip very short audio (<50 ms)
            min_samples = int(self.sample_rate * 0.05)  # 50ms
            if len(audio_int16) < min_samples:
                return audio_bytes

            # Step 2: time-stretch with pitch preservation
            y_stretched = pyrb.time_stretch(audio_int16, self.sample_rate, speed)

            # Step 3: convert back to int16 (pyrubberband may return float)
            if y_stretched.dtype != np.int16:
                # Ensure int16 range to avoid clipping
                y_stretched = np.clip(y_stretched, -32768, 32767).astype(np.int16)

            return y_stretched.tobytes()

        except Exception as e:
            # On any failure, return original audio to keep pipeline alive
            import warnings

            warnings.warn(
                f"Audio speed processing failed: {e}. Returning original audio.",
                RuntimeWarning,
            )
            return audio_bytes
