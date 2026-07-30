"""Agent tools for controlling TTS voice/emotion parameters.

Includes tool definitions/factories for LLM tool-calling usage or prompt docs
that help the model produce structured tool-call outputs.
"""

from .core import (
    AsyncTool,
    Finished,
    Running,
    SyncTool,
    Tool,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
    ToolRun,
    AsyncToolRun,
    ToolEngine,
)

from .thinking import (
    ThinkInput,
    ThinkOutput,
    build_think_tool,
)
from .async_web_search import (
    AsyncWebSearchInput,
    AsyncWebSearchOutput,
    build_async_web_search_tool,
)
from .deepresearch import (
    DeepResearchInput,
    DeepResearchOutput,
    DeepResearchState,
    build_deep_research_tool,
)

from .speech_control import (
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
    "AsyncTool",
    "Finished",
    "Running",
    "SyncTool",
    "Tool",
    "ToolEngineState",
    "ToolInput",
    "ToolOutput",
    "ToolResult",
    "ToolState",
    "ToolRun",
    "AsyncToolRun",
    "ToolEngine",
    "build_set_voice_tool",
    "build_set_emotion_tool",
    "build_silence_tool",
    "build_set_speed_tool",
    "AVAILABLE_EMOTIONS",
    "build_web_search_tool",
    "build_time_tool",
    "ThinkInput",
    "ThinkOutput",
    "build_think_tool",
    "AsyncWebSearchInput",
    "AsyncWebSearchOutput",
    "build_async_web_search_tool",
    "DeepResearchInput",
    "DeepResearchOutput",
    "DeepResearchState",
    "build_deep_research_tool",
]
