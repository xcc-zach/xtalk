from .interfaces import ASR
from ..registry import model


@model
class DummyASR(ASR):
    """
    Dummy ASR model for testing purposes.

    This model ignores the input audio and returns a predefined test text.
    """

    def __init__(self, default_text: str = "This is a dummy ASR test text"):
        """
        Initialize the DummyASR model.

        Args:
            default_text (str): The default text to return for any audio input
        """
        self.default_text = default_text
        self._stream_text = ""

    def recognize(self, audio: bytes) -> str:
        """Convert audio to text (stub implementation)."""
        return self.default_text

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Streaming stub: maintain accumulated text and return it."""
        del chat_history
        if audio or is_final:
            self._stream_text = self.default_text

        result = self._stream_text
        return result

    def reset(self):
        """Reset streaming text buffer."""
        self._stream_text = ""

    def clone(self) -> "DummyASR":
        """Clone the dummy ASR."""
        return DummyASR(default_text=self.default_text)
