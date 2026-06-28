from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain.tools import tool

AVAILABLE_EMOTIONS: List[str] = [
    "happy",
    "angry",
    "sad",
    "fear",
    "disgust",
    "depressed",
    "surprised",
    "calm",
    "normal",
]


def _make_enum_schema(enum_values: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": enum_values,
                "description": "Target name to switch to.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }


SET_VOICE_TOOL = "set_voice"


def build_set_voice_tool(available_voice_names: Optional[List[str]] = None):
    """Create a compact tool for switching TTS voice."""

    enum_values = list(available_voice_names or [])
    schema = (
        _make_enum_schema(enum_values)
        if enum_values
        else {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Target voice name.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        }
    )

    # Small-model friendly: list names and explicit trigger phrases
    voice_list_str = f"Options: {', '.join(enum_values)}." if enum_values else ""

    tool_description = f"""Change the speaker's voice identity. 
Trigger: User asks to 'change voice', 'sound like [name]', or 'be a man/woman'.
{voice_list_str}
Must call BEFORE speaking."""

    @tool(SET_VOICE_TOOL, args_schema=schema, return_direct=False)
    def set_voice(name: str) -> str:
        """Switch the TTS voice."""
        return f"voice switched to: {name}"

    set_voice.description = tool_description
    return set_voice


SET_EMOTION_TOOL = "set_emotion"


def build_set_emotion_tool(available_emotions: Optional[List[str]] = None):
    """Create a compact tool for switching speech emotion."""

    values = available_emotions or AVAILABLE_EMOTIONS
    schema = _make_enum_schema(values)
    emotion_list = ", ".join(values)

    emotion_description = f"""Switch speaking mood/tone. 
You must detect context: 
- Good news/Jokes -> 'happy'
- Bad news/Comforting -> 'sad'
- Secrets/Soothing -> 'calm'
- Anger/Warning -> 'angry'
- Default -> 'normal'
Allowed values: {emotion_list}.
Call BEFORE generating response text."""

    @tool(SET_EMOTION_TOOL, args_schema=schema, return_direct=False)
    def set_emotion(name: str) -> str:
        """Set the speaking emotion."""
        return f"emotion set to: {name}"

    set_emotion.description = emotion_description
    return set_emotion


SILENCE_TOOL = "silence"


def build_silence_tool():
    """Create a tool that only displays text without audio."""

    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to display silently.",
            }
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    # Clarify which content should stay silent to help smaller models
    silence_description = """Output text visually ONLY (NO AUDIO). 
Use this for: mathematical formulas, code blocks, complex symbols, or emojis that should not be read aloud."""

    @tool(SILENCE_TOOL, args_schema=args_schema, return_direct=False)
    def silence(text: str) -> str:
        """Display text without speaking it aloud."""
        return f"silence applied."

    silence.description = silence_description
    return silence


SET_SPEED_TOOL = "set_speed"


def build_set_speed_tool(*, min_speed: float = 0.5, max_speed: float = 2.0):
    """Create a compact tool for adjusting speaking speed."""

    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "speed": {
                "type": "number",
                "minimum": float(min_speed),
                "maximum": float(max_speed),
                "description": f"Speed multiplier ({min_speed}-{max_speed}).",
            }
        },
        "required": ["speed"],
        "additionalProperties": False,
    }

    # Provide direct mappings instead of verbose strategy text for smaller models
    speed_description = f"""Set speaking rate. Range: {min_speed} to {max_speed}.
Default: 1.0.
Mappings:
- 'Faster' -> 1.25 or 1.5
- 'Slower' -> 0.75 or 0.5
- 'Normal' -> 1.0"""

    @tool(SET_SPEED_TOOL, args_schema=args_schema, return_direct=False)
    def set_speed(speed: float) -> str:
        """Change speaking speed."""
        return f"speed set to: {speed:.2f}x"

    set_speed.description = speed_description
    return set_speed
