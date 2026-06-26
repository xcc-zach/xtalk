"""Agent tools for controlling TTS voice/emotion parameters.

Includes tool definitions/factories for LLM tool-calling usage or prompt docs
that help the model produce structured tool-call outputs.
"""

from .pipeline_control import (
    build_set_voice_tool,
    build_set_emotion_tool,
    build_silence_tool,
    build_set_speed_tool,
    AVAILABLE_EMOTIONS,
)
from .retrievers import (
    build_web_search_tool,
    build_time_tool,
)

__all__ = [
    "build_set_voice_tool",
    "build_set_emotion_tool",
    "build_silence_tool",
    "build_set_speed_tool",
    "AVAILABLE_EMOTIONS",
    "build_web_search_tool",
    "build_time_tool",
]
