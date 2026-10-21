"""Host-owned WAV recording from mixed Realtime audio events."""

import base64
import wave
from pathlib import Path


class RecordingStore:
    """Write one conversation's mixed PCM stream to a host-owned WAV file.

    Parameters
    ----------
    path : pathlib.Path
        Recording file selected by the application.
    """

    def __init__(self, path: Path):
        self.path = path
        self._writer = None

    def append(self, *, event: dict) -> None:
        """Append a session.audio.mix.delta event to the recording.

        Parameters
        ----------
        event : dict
            Mixed PCM event including sample rate and channel count.
        """
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = wave.open(str(self.path), "wb")
            self._writer.setnchannels(event["channels"])
            self._writer.setsampwidth(2)
            self._writer.setframerate(event["sample_rate"])
        self._writer.writeframes(base64.b64decode(event["delta"], validate=True))

    def close(self) -> None:
        """Finalize the WAV header and release the host file handle."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
