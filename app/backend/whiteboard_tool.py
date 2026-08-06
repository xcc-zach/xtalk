"""Built-in whiteboard tools that maintain one global Markdown text board.

This module mirrors ``resources/tools/whiteboard/whiteboard_tool.py`` so unit
tests can import the tool directly; the manifest loader runs the resource copy
inside a synthetic package.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from backend.whiteboard_store import (
    configure_whiteboard_data_directory,
    get_whiteboard_store,
)
from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)


MAX_WHITEBOARD_TEXT_CHUNK = 50_000
WHITEBOARD_COMMON_DESCRIPTION = (
    "在需要教学、演示的场景考虑调用该工具。"
    "Consider using this tool when teaching or demonstrating."
)

_data_directory = os.environ.get("XTALK_TOOL_DATA_DIR")
if _data_directory:
    configure_whiteboard_data_directory(Path(_data_directory) / "whiteboards")


class WhiteboardFetchInput(ToolInput):
    """Empty input for reading the whole whiteboard document."""


class WhiteboardAddInput(ToolInput):
    """Input for appending one text block to the whiteboard."""

    text: str = Field(
        min_length=1,
        max_length=MAX_WHITEBOARD_TEXT_CHUNK,
        description="Markdown text appended to the end of the whiteboard.",
    )


class WhiteboardDeleteInput(ToolInput):
    """Input for deleting one exact text block from the whiteboard."""

    text: str = Field(
        min_length=1,
        max_length=MAX_WHITEBOARD_TEXT_CHUNK,
        description="Exact whiteboard text to delete; every match is removed.",
    )


class WhiteboardUpdateInput(ToolInput):
    """Input for replacing one exact text block on the whiteboard."""

    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(
        alias="from",
        min_length=1,
        max_length=MAX_WHITEBOARD_TEXT_CHUNK,
        description="Exact whiteboard text to find; every match is replaced.",
    )
    to: str = Field(
        max_length=MAX_WHITEBOARD_TEXT_CHUNK,
        description="Replacement Markdown text; empty removes the block.",
    )


class WhiteboardOutput(ToolOutput):
    """Structured result returned by every whiteboard tool."""

    call_id: str = ""
    action: str
    success: bool
    text: str
    revision: int
    message: str


@dataclass
class WhiteboardState(ToolState):
    """Mutable state for one whiteboard tool call."""

    status_text: str = "Preparing whiteboard"
    output: WhiteboardOutput | None = None


class _WhiteboardTool(AsyncTool):
    """在需要教学、演示的场景考虑调用该工具。Consider using this tool when teaching or demonstrating.

    The whiteboard is one global Markdown text document shared by every
    conversation. Calls read or mutate the document and always return the full
    normalized snapshot so the independent whiteboard window can re-render.
    """

    subscribe_by_default = False
    structured_payload: ClassVar[bool] = True
    input_type = ToolInput
    state_type = WhiteboardState
    output_type = WhiteboardOutput
    initial_status = "Reading whiteboard"

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: ToolInput,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> Running:
        """Apply the concrete operation and return the updated snapshot."""

        del global_state
        tool_state.call_id = tool_call_id
        tool_state.status_text = cls.initial_status
        output = cls.execute(tool_input, tool_state)
        # ToolEngine requires the initial Running content to embed the call id,
        # otherwise every whiteboard call is rejected as an engine error.
        output.call_id = tool_call_id
        tool_state.output = output
        return Running(output.to_content())

    @classmethod
    def emit_updates(
        cls,
        tool_input: ToolInput,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[WhiteboardOutput]]:
        """Complete the call with the same snapshot produced initially."""

        del tool_input, global_state
        if tool_state.output is None:
            tool_state.output = cls.execute(cls.input_type(), tool_state)
        yield Finished(tool_state.output)

    @classmethod
    def status(
        cls,
        tool_input: ToolInput,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> str:
        """Return the current phase for the live panel."""

        del tool_input, global_state
        return tool_state.status_text

    @classmethod
    def stop(
        cls,
        tool_input: ToolInput,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> None:
        """Mark the call as stopped."""

        del tool_input, global_state
        tool_state.status_text = "Whiteboard stopped"

    @classmethod
    def execute(
        cls,
        tool_input: ToolInput,
        tool_state: WhiteboardState,
    ) -> WhiteboardOutput:
        """Execute the concrete operation implemented by a subclass."""

        raise NotImplementedError


class WhiteboardFetchTool(_WhiteboardTool):
    """在需要教学、演示的场景考虑调用该工具。Consider using this tool when teaching or demonstrating.
    获取白板上全部文本，用于查看当前内容。"""

    name = "fetch_text"
    initial_status = "Reading whiteboard"
    input_type = WhiteboardFetchInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardFetchInput,
        tool_state: WhiteboardState,
    ) -> WhiteboardOutput:
        """Return the full whiteboard document without mutating it."""

        del tool_input
        snapshot: dict[str, Any] = get_whiteboard_store(
            tool_state.metadata.get("session_id")
        ).snapshot()
        text = str(snapshot["text"])
        message = (
            "白板内容为空。"
            if not text
            else f"白板当前共 {len(text)} 个字符，revision {snapshot['revision']}。"
        )
        return WhiteboardOutput(
            action=cls.name,
            success=True,
            text=text,
            revision=int(snapshot["revision"]),
            message=message,
        )


class WhiteboardAddTool(_WhiteboardTool):
    """在需要教学、演示的场景考虑调用该工具。Consider using this tool when teaching or demonstrating.
    把一段文本追加到白板末尾；适合把新知识点、步骤或示例持续写到白板上。"""

    name = "add_text"
    initial_status = "Appending whiteboard text"
    input_type = WhiteboardAddInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardAddInput,
        tool_state: WhiteboardState,
    ) -> WhiteboardOutput:
        """Append one Markdown text block and return the new snapshot."""

        snapshot: dict[str, Any] = get_whiteboard_store(
            tool_state.metadata.get("session_id")
        ).add_text(tool_input.text)
        return WhiteboardOutput(
            action=cls.name,
            success=True,
            text=str(snapshot["text"]),
            revision=int(snapshot["revision"]),
            message=f"已追加文本，白板 revision {snapshot['revision']}。",
        )


class WhiteboardDeleteTool(_WhiteboardTool):
    """在需要教学、演示的场景考虑调用该工具。Consider using this tool when teaching or demonstrating.
    删除白板上匹配的文本；删除的内容必须与白板现有文本完全一致。"""

    name = "delete_text"
    initial_status = "Deleting whiteboard text"
    input_type = WhiteboardDeleteInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardDeleteInput,
        tool_state: WhiteboardState,
    ) -> WhiteboardOutput:
        """Delete every exact match and return the new snapshot."""

        snapshot: dict[str, Any] = get_whiteboard_store(
            tool_state.metadata.get("session_id")
        ).delete_text(tool_input.text)
        return WhiteboardOutput(
            action=cls.name,
            success=True,
            text=str(snapshot["text"]),
            revision=int(snapshot["revision"]),
            message=f"已删除匹配文本，白板 revision {snapshot['revision']}。",
        )


class WhiteboardUpdateTool(_WhiteboardTool):
    """在需要教学、演示的场景考虑调用该工具。Consider using this tool when teaching or demonstrating.
    把白板上 from 匹配的文本更新为 to；from 必须与现有文本完全一致。"""

    name = "update_text"
    initial_status = "Updating whiteboard text"
    input_type = WhiteboardUpdateInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardUpdateInput,
        tool_state: WhiteboardState,
    ) -> WhiteboardOutput:
        """Replace every exact match and return the new snapshot."""

        snapshot: dict[str, Any] = get_whiteboard_store(
            tool_state.metadata.get("session_id")
        ).update_text(
            tool_input.from_,
            tool_input.to,
        )
        return WhiteboardOutput(
            action=cls.name,
            success=True,
            text=str(snapshot["text"]),
            revision=int(snapshot["revision"]),
            message=f"已更新匹配文本，白板 revision {snapshot['revision']}。",
        )


def create_tools() -> list[type[AsyncTool]]:
    """Create the text whiteboard tools exported by this directory.

    Returns
    -------
    list[type[AsyncTool]]
        The four whiteboard operations sharing one global text document.
    """

    return [
        WhiteboardFetchTool,
        WhiteboardAddTool,
        WhiteboardDeleteTool,
        WhiteboardUpdateTool,
    ]
