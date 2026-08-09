"""Built-in whiteboard tools that maintain per-session Markdown text boards."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from backend.whiteboard_store import (
    configure_whiteboard_data_directory,
    get_whiteboard_store,
)
from xtalk.models.agents.tools import (
    SyncTool,
    ToolEngineState,
    ToolInput,
    ToolOutput,
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

    action: str
    success: bool
    text: str
    revision: int
    message: str


class _WhiteboardTool(SyncTool):
    """在需要教学、演示的场景考虑调用该工具。Consider using this tool when teaching or demonstrating.

    Each conversation owns one Markdown document. Calls complete immediately
    and return the full normalized snapshot so the independent whiteboard
    window can re-render.
    """

    structured_payload: ClassVar[bool] = True
    input_type = ToolInput
    output_type = WhiteboardOutput

    @classmethod
    def invoke(
        cls,
        tool_input: ToolInput,
        global_state: ToolEngineState,
    ) -> WhiteboardOutput:
        """Execute one whiteboard operation for the active backend session."""

        session_id = None
        if isinstance(global_state, dict):
            value = global_state.get("session_id")
            if isinstance(value, str) and value:
                session_id = value
        return cls.execute(tool_input, session_id)

    @classmethod
    def execute(
        cls,
        tool_input: ToolInput,
        session_id: str | None,
    ) -> WhiteboardOutput:
        """Execute the concrete operation implemented by a subclass."""

        raise NotImplementedError


class WhiteboardFetchTool(_WhiteboardTool):
    """在需要教学、演示的场景考虑调用该工具。Consider using this tool when teaching or demonstrating.
    获取白板上全部文本，用于查看当前内容。"""

    name = "fetch_text"
    input_type = WhiteboardFetchInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardFetchInput,
        session_id: str | None,
    ) -> WhiteboardOutput:
        """Return the full whiteboard document without mutating it."""

        del tool_input
        snapshot: dict[str, Any] = get_whiteboard_store(session_id).snapshot()
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
    input_type = WhiteboardAddInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardAddInput,
        session_id: str | None,
    ) -> WhiteboardOutput:
        """Append one Markdown text block and return the new snapshot."""

        snapshot: dict[str, Any] = get_whiteboard_store(session_id).add_text(
            tool_input.text
        )
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
    input_type = WhiteboardDeleteInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardDeleteInput,
        session_id: str | None,
    ) -> WhiteboardOutput:
        """Delete every exact match and return the new snapshot."""

        snapshot: dict[str, Any] = get_whiteboard_store(session_id).delete_text(
            tool_input.text
        )
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
    input_type = WhiteboardUpdateInput

    @classmethod
    def execute(
        cls,
        tool_input: WhiteboardUpdateInput,
        session_id: str | None,
    ) -> WhiteboardOutput:
        """Replace every exact match and return the new snapshot."""

        snapshot: dict[str, Any] = get_whiteboard_store(session_id).update_text(
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


def create_tools() -> list[type[SyncTool]]:
    """Create the text whiteboard tools exported by this directory.

    Returns
    -------
    list[type[SyncTool]]
        The four immediate whiteboard operations scoped by backend session.
    """

    return [
        WhiteboardFetchTool,
        WhiteboardAddTool,
        WhiteboardDeleteTool,
        WhiteboardUpdateTool,
    ]
