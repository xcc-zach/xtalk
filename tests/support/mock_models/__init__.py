"""Mock models for the conversation experience tests.

Importing this package registers ``MockASR``, ``MockAgent``, ``MockTTS`` and
``MockVAD`` with the xtalk model registry.
"""

from .mock_agent import MockAgent, echo_reply, summary_reply
from .mock_asr import MockASR
from .mock_tools import MockLookupTool, MockTimerTool
from .mock_tts import MockTTS
from .mock_vad import MockVAD
from .scripted_chat_model import ScriptedChatModel

__all__ = [
    "MockASR",
    "MockAgent",
    "MockLookupTool",
    "MockTTS",
    "MockTimerTool",
    "MockVAD",
    "ScriptedChatModel",
    "echo_reply",
    "summary_reply",
]
