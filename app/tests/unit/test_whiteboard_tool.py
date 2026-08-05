"""Contract tests for the desktop whiteboard tool."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError
from xtalk.models.agents.tools import Finished, Running

from backend.tool_ui import ToolUIBinding, wrap_tools_with_ui
from backend.whiteboard_tool import (
    MAX_WHITEBOARD_NOTES,
    MAX_WHITEBOARD_OPS,
    AddNoteOp,
    ClearOp,
    RemoveNoteOp,
    SetTitleOp,
    UpdateNoteOp,
    WhiteboardInput,
    WhiteboardNoteInput,
    WhiteboardState,
    WhiteboardTool,
)


def _snapshot(tool_state: WhiteboardState) -> dict[str, Any]:
    """Parse the serialized snapshot of one call's tool state."""

    return json.loads(WhiteboardTool._snapshot_json(tool_state))


def _run_initial(
    ops: list[Any],
    *,
    global_state: Any = None,
    call_id: str = "whiteboard-call",
) -> tuple[Running, WhiteboardState, Any]:
    """Run one whiteboard call and return its initial result and state."""

    engine_state: Any = {} if global_state is None else global_state
    tool_state = WhiteboardState(call_id=call_id)
    initial = WhiteboardTool.emit_initial(
        call_id,
        WhiteboardInput(ops=ops),
        tool_state,
        engine_state,
    )
    return initial, tool_state, engine_state


def _collect_updates(
    tool_input: WhiteboardInput,
    tool_state: WhiteboardState,
) -> list[Any]:
    """Collect every whiteboard update into a list."""

    return list(
        WhiteboardTool.emit_updates(
            tool_input,
            tool_state,
            object(),
        )
    )


def test_whiteboard_applies_operations_and_returns_full_snapshot() -> None:
    """Apply title, add, update, and remove operations in one call."""

    initial, tool_state, _ = _run_initial(
        [
            SetTitleOp(title="本周计划"),
            AddNoteOp(
                note=WhiteboardNoteInput(
                    id="n1",
                    text="写方案",
                    color="blue",
                )
            ),
            AddNoteOp(note=WhiteboardNoteInput(text="自动编号")),
            UpdateNoteOp(id="n1", text="写白板方案"),
            RemoveNoteOp(id="n1"),
        ]
    )

    assert isinstance(initial, Running)
    assert "whiteboard-call" in initial.content
    snapshot = json.loads(initial.content)
    assert snapshot["version"] == 1
    assert snapshot["call_id"] == "whiteboard-call"
    assert snapshot["title"] == "本周计划"
    assert snapshot["revision"] == 1
    assert len(snapshot["notes"]) == 1
    assert snapshot["notes"][0]["text"] == "自动编号"
    assert "updated_at" in snapshot

    updates = _collect_updates(WhiteboardInput(ops=[]), tool_state)
    assert len(updates) == 1
    finished = updates[0]
    assert isinstance(finished, Finished)
    assert json.loads(finished.content.to_content())["revision"] == 1


def test_whiteboard_persists_board_across_calls_in_engine_state() -> None:
    """Later calls apply incremental operations against earlier snapshots."""

    engine_state: dict[str, Any] = {}
    _, first_state, _ = _run_initial(
        [AddNoteOp(note=WhiteboardNoteInput(id="n1", text="第一张"))],
        global_state=engine_state,
        call_id="call-1",
    )
    _, second_state, _ = _run_initial(
        [AddNoteOp(note=WhiteboardNoteInput(id="n2", text="第二张"))],
        global_state=engine_state,
        call_id="call-2",
    )

    assert _snapshot(first_state)["revision"] == 1
    snapshot = _snapshot(second_state)
    assert snapshot["revision"] == 2
    assert [note["id"] for note in snapshot["notes"]] == ["n1", "n2"]


def test_whiteboard_works_without_engine_state_for_single_call() -> None:
    """Degrade gracefully when the session engine state is unavailable."""

    initial, tool_state, _ = _run_initial(
        [AddNoteOp(note=WhiteboardNoteInput(text="独立便签"))],
        global_state=object(),
    )

    assert isinstance(initial, Running)
    assert _snapshot(tool_state)["revision"] == 1


def test_whiteboard_clear_keeps_title() -> None:
    """Clear removes every note while preserving the board title."""

    _, tool_state, _ = _run_initial(
        [
            SetTitleOp(title="保留标题"),
            AddNoteOp(note=WhiteboardNoteInput(id="n1", text="待清理")),
            ClearOp(),
        ]
    )

    snapshot = _snapshot(tool_state)
    assert snapshot["title"] == "保留标题"
    assert snapshot["notes"] == []


def test_whiteboard_update_requires_existing_note() -> None:
    """Reject updates for ids that are not on the board."""

    with pytest.raises(ValueError, match="does not exist"):
        _run_initial([UpdateNoteOp(id="missing", text="无效更新")])


def test_whiteboard_accepts_flattened_add_note_args() -> None:
    """Tolerate models that put note fields directly on the operation."""

    tool_input = WhiteboardInput.model_validate(
        {
            "ops": [
                {
                    "op": "add_note",
                    "id": "flat-id",
                    "text": "展平便签",
                    "color": "blue",
                }
            ]
        }
    )

    assert tool_input.ops[0].note.text == "展平便签"
    assert tool_input.ops[0].note.id == "flat-id"
    assert tool_input.ops[0].note.color == "blue"


def test_whiteboard_accepts_plain_text_add_note_args() -> None:
    """Tolerate models that pass a note as a plain string."""

    tool_input = WhiteboardInput.model_validate(
        {
            "ops": [
                {
                    "op": "add_note",
                    "note": "纯文本便签",
                }
            ]
        }
    )

    assert tool_input.ops[0].note.text == "纯文本便签"
    assert tool_input.ops[0].note.id is None


def test_whiteboard_enforces_note_limit() -> None:
    """Refuse new notes when the board is already full."""

    tool_state = WhiteboardState(call_id="full-board")
    tool_state.notes = {
        f"note-{index}": {"text": f"note {index}", "color": "yellow"}
        for index in range(MAX_WHITEBOARD_NOTES)
    }

    with pytest.raises(ValueError, match="already holds"):
        WhiteboardTool._apply_ops(
            [AddNoteOp(note=WhiteboardNoteInput(text="溢出"))],
            tool_state,
        )
    WhiteboardTool._apply_ops(
        [
            UpdateNoteOp(
                id="note-0",
                text="满员时仍可修改",
            )
        ],
        tool_state,
    )
    assert tool_state.notes["note-0"]["text"] == "满员时仍可修改"


def test_whiteboard_input_validates_operation_limits() -> None:
    """Reject oversized operation batches and note text through the schema."""

    with pytest.raises(ValidationError):
        WhiteboardInput(
            ops=[
                AddNoteOp(note=WhiteboardNoteInput(text="x"))
                for _ in range(MAX_WHITEBOARD_OPS + 1)
            ]
        )
    with pytest.raises(ValidationError):
        WhiteboardInput(
            ops=[
                AddNoteOp(
                    note=WhiteboardNoteInput(text="长" * 2001)
                )
            ]
        )


def test_whiteboard_wrapper_emits_structured_payload() -> None:
    """The tool UI observer attaches the parsed board snapshot as payload."""

    class _RecordingBroker:
        """Capture publish arguments without opening a WebSocket."""

        def __init__(self) -> None:
            """Initialize an empty emit list."""

            self.emits: list[dict[str, Any]] = []

        def register_ui_tool(self, tool_name: str) -> None:
            """Ignore tool registration in this fixture."""

        async def publish_status(self, **payload: Any) -> None:
            """Ignore status observations in this fixture."""

        async def publish_emit(self, **payload: Any) -> None:
            """Record one emit payload."""

            self.emits.append(payload)

        def finish_call(self, call_id: str) -> None:
            """Ignore terminal cleanup in this fixture."""

    async def scenario() -> tuple[Any, _RecordingBroker]:
        broker = _RecordingBroker()
        wrapped = wrap_tools_with_ui(
            [WhiteboardTool],
            binding=ToolUIBinding(
                tool_id="builtin:whiteboard",
                update_every_s=-1,
            ),
            broker=broker,  # type: ignore[arg-type]
        )[0]
        tool_state = WhiteboardState(call_id="wrapped-call")
        initial = await wrapped.aemit_initial(
            "wrapped-call",
            WhiteboardInput(
                ops=[AddNoteOp(note=WhiteboardNoteInput(text="包装链路"))]
            ),
            tool_state,
            {},
        )
        return initial, broker

    initial, broker = asyncio.run(scenario())

    assert isinstance(initial, Running)
    assert len(broker.emits) == 1
    emit = broker.emits[0]
    assert emit["payload"]["title"] == ""
    assert emit["payload"]["revision"] == 1
    assert emit["payload"]["notes"][0]["text"] == "包装链路"
