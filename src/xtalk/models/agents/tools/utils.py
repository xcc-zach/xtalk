from __future__ import annotations

"""
Helper utilities unrelated to agent decision logic.

This module defines a constructor for a pseudo tool named "tool_call_result".
It wraps each finished tool invocation into a consistent payload so upstream
components (e.g., Pipeline/TTSManager) can consume the result and downstream
modules (e.g., RetrievalManager) can react accordingly.

Notes:
- The tool is NOT registered with the LLM (models never call it directly).
- The emitted tool call simply tags a tool result with:
  name: original tool name (e.g., "web_search")
  args: original arguments (verbatim)
  content: textual output from the tool
"""

from typing import Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import ToolCall


class ToolCallResultArgs(TypedDict):
    """Serialized result for one completed tool invocation.

    Notes
    -----
    ``name`` stores the original tool name, ``args`` stores the original tool
    arguments, and ``content`` stores the textual tool output.
    """

    name: str
    args: dict[str, Any]
    content: str


class ToolCallResult(ToolCall):
    """Structured tool-call event emitted after a tool finishes."""

    name: Literal["tool_call_result"]
    args: ToolCallResultArgs


def build_tool_call_result(
    *, tool_call: ToolCall, result_content: str
) -> ToolCallResult:
    """Build a normalized tool-call result event.

    Parameters
    ----------
    tool_call : ToolCall
        Original tool call.
    result_content : str
        Textual tool result.

    Returns
    -------
    ToolCallResult
        Structured tool call named ``"tool_call_result"``.
    """
    args = tool_call.get("args", {})
    return ToolCallResult(
        name="tool_call_result",
        args={
            "name": str(tool_call.get("name", "") or ""),
            "args": dict(args) if isinstance(args, dict) else {},
            "content": result_content or "",
        },
        id=f"tool_call_result_{uuid4().hex}",
    )
