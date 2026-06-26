import os
import struct
from typing import Optional
from .interfaces import TTS
from ..registry import model


@model
class DummyTTS(TTS):
    """
    Dummy TTS model for testing purposes.

    This model ignores the input text and returns predefined audio data.
    By default, it generates silence audio unless a specific audio file is provided.
    """

    def __init__(self, audio_file: Optional[str] = None, sample_rate: int = 16000):
        """
        Initialize the DummyTTS model.

        Args:
            audio_file (Optional[str]): Path to an audio file to return. If None, generates silence audio
            sample_rate (int): Sample rate of the audio (default: 16000 Hz)
        """
        self.audio_file = audio_file
        self._sample_rate = sample_rate

        if audio_file is not None and os.path.exists(audio_file):
            # Load audio data from specified file
            with open(audio_file, "rb") as f:
                self._audio_data = f.read()
        else:
            # Generate silence audio (default behavior)
            self._audio_data = self._generate_silence_audio()

    def clone(self):
        return DummyTTS(audio_file=self.audio_file, sample_rate=self._sample_rate)

    def _generate_silence_audio(self, duration: float = 1.0) -> bytes:
        """
        Generate silence audio with WAV format.

        Args:
            duration (float): Duration of silence in seconds (default: 1.0)

        Returns:
            bytes: WAV formatted silence audio data
        """
        sample_rate = self._sample_rate
        num_samples = int(sample_rate * duration)

        # WAV header
        header = struct.pack("<4sI4s", b"RIFF", 36 + num_samples * 2, b"WAVE")
        header += struct.pack(
            "<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16
        )
        header += struct.pack("<4sI", b"data", num_samples * 2)

        # Silent audio data (all zeros)
        audio_data = b"\x00" * (num_samples * 2)

        return header + audio_data

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech (dummy implementation).

        This method ignores the text input and returns the predefined audio data.

        Args:
            text (str): Text to synthesize (ignored in dummy implementation)

        Returns:
            bytes: The predefined audio data (silence or custom audio file)
        """
        return self._audio_data
