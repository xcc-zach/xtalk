"""Built-in whiteboard tool that streams structured board snapshots.

This module mirrors ``resources/tools/whiteboard/whiteboard_tool.py`` so unit
tests can import the tool directly; the manifest loader runs the resource copy
inside a synthetic package.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
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


MAX_WHITEBOARD_NOTES = 200
MAX_WHITEBOARD_NOTE_TEXT = 2000
MAX_WHITEBOARD_TITLE = 100
MAX_WHITEBOARD_OPS = 50

_WHITEBOARD_STATE_KEY = "_xtalk_whiteboard_board"

WhiteboardColor = Literal["yellow", "blue", "green", "pink", "purple"]


class WhiteboardNoteInput(BaseModel):
    """One sticky note supplied by the model."""

    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Optional stable note id used by later update or remove operations.",
    )
    text: str = Field(
        min_length=1,
        max_length=MAX_WHITEBOARD_NOTE_TEXT,
        description="Note text displayed on the whiteboard.",
    )
    color: WhiteboardColor = Field(
        default="yellow",
        description="Sticky-note accent color.",
    )


class SetTitleOp(BaseModel):
    """Replace the whiteboard title."""

    op: Literal["set_title"] = "set_title"
    title: str = Field(
        max_length=MAX_WHITEBOARD_TITLE,
        description="New whiteboard title shown above the notes.",
    )


class AddNoteOp(BaseModel):
    """Append one sticky note, or replace the note with the same id."""

    op: Literal["add_note"] = "add_note"
    note: WhiteboardNoteInput | str

    @model_validator(mode="before")
    @classmethod
    def _accept_flattened_note(cls, value: Any) -> Any:
        """Accept models that flatten id/text/color onto the operation."""

        if (
            isinstance(value, dict)
            and "note" not in value
            and isinstance(value.get("text"), str)
        ):
            return {
                **value,
                "note": {
                    "id": value.get("id"),
                    "text": value["text"],
                    "color": value.get("color"),
                },
            }
        return value

    @field_validator("note")
    @classmethod
    def _coerce_note(cls, value: WhiteboardNoteInput | str) -> WhiteboardNoteInput:
        """Wrap a plain-text note into a note object for robustness."""

        if isinstance(value, str):
            return WhiteboardNoteInput(text=value)
        return value


class UpdateNoteOp(BaseModel):
    """Patch an existing sticky note by id."""

    op: Literal["update_note"] = "update_note"
    id: str = Field(
        min_length=1,
        max_length=64,
        description="Id of an existing note returned by a previous snapshot.",
    )
    text: str | None = Field(
        default=None,
        max_length=MAX_WHITEBOARD_NOTE_TEXT,
        description="Optional replacement note text.",
    )
    color: WhiteboardColor | None = Field(
        default=None,
        description="Optional replacement accent color.",
    )


class RemoveNoteOp(BaseModel):
    """Remove one sticky note by id."""

    op: Literal["remove_note"] = "remove_note"
    id: str = Field(
        min_length=1,
        max_length=64,
        description="Id of the note to remove.",
    )


class ClearOp(BaseModel):
    """Remove every sticky note while keeping the title."""

    op: Literal["clear"] = "clear"


WhiteboardOp = Annotated[
    Union[
        SetTitleOp,
        AddNoteOp,
        UpdateNoteOp,
        RemoveNoteOp,
        ClearOp,
    ],
    Field(discriminator="op"),
]


class WhiteboardInput(ToolInput):
    """Input accepted by the whiteboard tool."""

    ops: list[WhiteboardOp] = Field(
        default_factory=list,
        max_length=MAX_WHITEBOARD_OPS,
        description=(
            "Operations applied to the conversation whiteboard. Every call "
            "returns the full normalized board snapshot in its tool result."
        ),
    )


class WhiteboardOutput(ToolOutput):
    """Full whiteboard snapshot returned by the whiteboard tool."""

    content: str

    def to_content(self) -> str:
        """Return the serialized board snapshot."""

        return self.content


@dataclass
class WhiteboardState(ToolState):
    """Mutable whiteboard state for one tool call."""

    title: str = ""
    revision: int = 0
    notes: dict[str, dict[str, str]] = field(default_factory=dict)


class WhiteboardTool(AsyncTool):
    """Shared whiteboard shown to the user as sticky notes; editable across turns.

    Teaching rule (MUST, highest priority): for any request to teach, explain,
    or tutor a topic — for example "教我牛顿莱布尼茨公式", "解释一下贝叶斯
    定理", "讲讲光合作用", or "教我怎么求导" — your reply MUST begin with a
    whiteboard_update tool call that puts the topic title, the key points or
    steps, and one concrete example on the board, followed by the verbal
    explanation. Never describe putting content on the whiteboard without
    actually calling this tool, and never answer a teaching request with text
    alone.

    Other triggers:
    - The user asks for a plan, outline, brainstorm, checklist, agenda,
      comparison, or a step-by-step breakdown with multiple items.
    - Structured content needs to stay visible and keep evolving over several
      turns (track progress, revise items, mark things done as the discussion
      moves on).
    - A compact visual summary of 3+ distinct items would help the user follow
      along or make a decision.

    Incremental behavior:
    - Add one note per meaningful item and keep note text short, one idea per
      note.
    - Prefer add_note / update_note / remove_note with existing note ids over
      replacing the whole board; reuse the board already shown in this
      conversation instead of creating a fresh one.
    - Use set_title for a short board title; use clear only when the user asks
      to reset.

    Do not call it for one-off factual questions that are not teaching
    requests, such as asking for the time or a single fact.
    """

    name = "whiteboard_update"
    subscribe_by_default = False
    structured_payload: ClassVar[bool] = True
    _state_key: ClassVar[str] = _WHITEBOARD_STATE_KEY

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: WhiteboardInput,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> Running:
        """Apply the requested operations and return the full board snapshot."""

        cls._load_board(tool_state, global_state)
        cls._apply_ops(tool_input.ops, tool_state)
        cls._save_board(tool_state, global_state)
        return Running(cls._snapshot_json(tool_state))

    @classmethod
    def emit_updates(
        cls,
        tool_input: WhiteboardInput,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[WhiteboardOutput]]:
        """Complete the call with the same full board snapshot."""

        del tool_input, global_state
        yield Finished(WhiteboardOutput(content=cls._snapshot_json(tool_state)))

    @classmethod
    def status(
        cls,
        tool_input: WhiteboardInput,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> str:
        """Return the human-readable board status for the live panel."""

        del tool_input, global_state
        return (
            f"Whiteboard revision {tool_state.revision} with "
            f"{len(tool_state.notes)} notes"
        )

    @classmethod
    def _load_board(
        cls,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> None:
        """Load the persisted board into a fresh call's tool state."""

        if tool_state.revision > 0 or not isinstance(global_state, dict):
            return
        persisted = global_state.get(cls._state_key)
        if not isinstance(persisted, dict):
            return
        title = persisted.get("title")
        if isinstance(title, str):
            tool_state.title = title[:MAX_WHITEBOARD_TITLE]
        revision = persisted.get("revision")
        if isinstance(revision, int) and revision >= 0:
            tool_state.revision = revision
        notes = persisted.get("notes")
        if isinstance(notes, dict):
            tool_state.notes = {
                str(note_id): {
                    "text": str(note["text"])[:MAX_WHITEBOARD_NOTE_TEXT],
                    "color": (
                        str(note["color"])
                        if isinstance(note.get("color"), str)
                        else "yellow"
                    ),
                }
                for note_id, note in notes.items()
                if isinstance(note, dict) and "text" in note
            }

    @classmethod
    def _save_board(
        cls,
        tool_state: WhiteboardState,
        global_state: ToolEngineState,
    ) -> None:
        """Store the board so subsequent calls can apply incremental ops."""

        if isinstance(global_state, dict):
            global_state[cls._state_key] = {
                "title": tool_state.title,
                "revision": tool_state.revision,
                "notes": dict(tool_state.notes),
            }

    @classmethod
    def _apply_ops(
        cls,
        ops: list[WhiteboardOp],
        tool_state: WhiteboardState,
    ) -> None:
        """Apply one operation batch and advance the revision counter."""

        for op in ops:
            if isinstance(op, SetTitleOp):
                tool_state.title = op.title
            elif isinstance(op, AddNoteOp):
                note_id = op.note.id or f"note-{uuid4().hex[:8]}"
                if (
                    note_id not in tool_state.notes
                    and len(tool_state.notes) >= MAX_WHITEBOARD_NOTES
                ):
                    raise ValueError(
                        f"whiteboard already holds {MAX_WHITEBOARD_NOTES} notes"
                    )
                tool_state.notes[note_id] = {
                    "text": op.note.text,
                    "color": op.note.color,
                }
            elif isinstance(op, UpdateNoteOp):
                note = tool_state.notes.get(op.id)
                if note is None:
                    raise ValueError(
                        f"whiteboard note {op.id!r} does not exist"
                    )
                if op.text is not None:
                    note["text"] = op.text
                if op.color is not None:
                    note["color"] = op.color
            elif isinstance(op, RemoveNoteOp):
                tool_state.notes.pop(op.id, None)
            elif isinstance(op, ClearOp):
                tool_state.notes.clear()
        tool_state.revision += 1

    @classmethod
    def _snapshot(cls, tool_state: WhiteboardState) -> dict[str, Any]:
        """Build the normalized full-board snapshot for one call."""

        return {
            "version": 1,
            "call_id": tool_state.call_id,
            "title": tool_state.title,
            "revision": tool_state.revision,
            "notes": [
                {"id": note_id, **note}
                for note_id, note in tool_state.notes.items()
            ],
            "updated_at": _utc_now(),
        }

    @classmethod
    def _snapshot_json(cls, tool_state: WhiteboardState) -> str:
        """Serialize the full board snapshot for one emit."""

        return json.dumps(
            cls._snapshot(tool_state),
            ensure_ascii=False,
            separators=(",", ":"),
        )


def create_tools() -> list[type[AsyncTool]]:
    """Create the tools exported by this directory.

    Returns
    -------
    list[type[AsyncTool]]
        Native XTalk tool classes registered with the configured Agent.
    """

    return [WhiteboardTool]


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()
