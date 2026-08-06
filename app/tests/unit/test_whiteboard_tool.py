"""Contract tests for the per-session desktop text whiteboard tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError
from xtalk.models.agents.tools import Finished, Running

from backend.tool_ui import ToolUIBinding, wrap_tools_with_ui
from backend.whiteboard_store import (
    configure_whiteboard_data_directory,
    get_whiteboard_store,
    reset_whiteboard_stores,
)
from backend.whiteboard_tool import (
    MAX_WHITEBOARD_TEXT_CHUNK,
    WhiteboardAddInput,
    WhiteboardAddTool,
    WhiteboardDeleteInput,
    WhiteboardDeleteTool,
    WhiteboardFetchTool,
    WhiteboardState,
    WhiteboardUpdateInput,
    WhiteboardUpdateTool,
)


@pytest.fixture(autouse=True)
def _reset_whiteboard_stores() -> Any:
    """Restore a fresh in-memory registry around every test."""

    reset_whiteboard_stores()
    yield
    reset_whiteboard_stores()


def _run_initial(
    tool: type[Any],
    tool_input: Any,
    *,
    session_id: str = "session-1",
    call_id: str = "whiteboard-call",
) -> tuple[Running, WhiteboardState]:
    """Run one whiteboard call and return its initial result and state."""

    tool_state = WhiteboardState(call_id=call_id)
    tool_state.metadata["session_id"] = session_id
    initial = tool.emit_initial(
        call_id,
        tool_input,
        tool_state,
        object(),
    )
    return initial, tool_state


def _collect_updates(
    tool: type[Any],
    tool_input: Any,
    tool_state: WhiteboardState,
) -> list[Any]:
    """Collect every whiteboard update into a list."""

    return list(tool.emit_updates(tool_input, tool_state, object()))


def test_whiteboard_add_appends_text_and_advances_revision() -> None:
    """Appending text joins blocks with one newline and bumps the revision."""

    initial, tool_state = _run_initial(
        WhiteboardAddTool,
        WhiteboardAddInput(text="# 第一段"),
    )

    assert isinstance(initial, Running)
    assert "whiteboard-call" in initial.content
    snapshot = json.loads(initial.content)
    assert snapshot["action"] == "add_text"
    assert snapshot["success"] is True
    assert snapshot["text"] == "# 第一段"
    assert snapshot["revision"] == 1
    assert snapshot["call_id"] == "whiteboard-call"
    assert snapshot["message"]

    _, second_state = _run_initial(
        WhiteboardAddTool,
        WhiteboardAddInput(text="第二段"),
        call_id="whiteboard-call-2",
    )
    second_snapshot = json.loads(second_state.output.to_content())  # type: ignore[union-attr]
    assert second_snapshot["text"] == "# 第一段\n第二段"
    assert second_snapshot["revision"] == 2

    updates = _collect_updates(
        WhiteboardAddTool,
        WhiteboardAddInput(text="第三段"),
        tool_state,
    )
    assert len(updates) == 1
    assert isinstance(updates[0], Finished)


def test_whiteboard_fetch_returns_full_text_without_mutating() -> None:
    """Fetch reports the whole document and never changes the revision."""

    get_whiteboard_store("session-1").add_text("演示内容")
    initial, tool_state = _run_initial(WhiteboardFetchTool, WhiteboardFetchTool.input_type())

    assert isinstance(initial, Running)
    snapshot = json.loads(initial.content)
    assert snapshot["text"] == "演示内容"
    assert snapshot["revision"] == 1
    assert json.loads(tool_state.output.to_content())["action"] == "fetch_text"  # type: ignore[union-attr]
    assert get_whiteboard_store("session-1").snapshot()["revision"] == 1


def test_whiteboard_delete_removes_exact_match() -> None:
    """Deleting one block removes it while preserving the rest."""

    store = get_whiteboard_store("session-1")
    store.add_text("# 标题")
    store.add_text("需要删除的内容")
    store.add_text("保留内容")

    initial, _ = _run_initial(
        WhiteboardDeleteTool,
        WhiteboardDeleteInput(text="需要删除的内容"),
    )

    snapshot = json.loads(initial.content)
    assert snapshot["revision"] == 4
    assert snapshot["text"] == "# 标题\n保留内容"


def test_whiteboard_delete_rejects_missing_text() -> None:
    """Deleting text that is not on the board raises a clear error."""

    with pytest.raises(ValueError, match="not found"):
        _run_initial(
            WhiteboardDeleteTool,
            WhiteboardDeleteInput(text="不存在的内容"),
        )


def test_whiteboard_update_replaces_every_match() -> None:
    """Updating text replaces each occurrence and keeps the rest intact."""

    get_whiteboard_store("session-1").add_text("旧文本 保留 旧文本")

    initial, _ = _run_initial(
        WhiteboardUpdateTool,
        WhiteboardUpdateInput(from_="旧文本", to="新文本"),
    )

    snapshot = json.loads(initial.content)
    assert snapshot["revision"] == 2
    assert snapshot["text"] == "新文本 保留 新文本"


def test_whiteboard_update_accepts_json_from_alias() -> None:
    """The model-facing schema exposes ``from`` instead of ``from_``."""

    tool_input = WhiteboardUpdateInput.model_validate(
        {"from": "旧", "to": "新"}
    )

    assert tool_input.from_ == "旧"
    assert tool_input.to == "新"
    assert WhiteboardUpdateInput.model_json_schema()["properties"]["from"]


def test_whiteboard_update_rejects_missing_from() -> None:
    """Updating absent text raises a clear error."""

    with pytest.raises(ValueError, match="not found"):
        _run_initial(
            WhiteboardUpdateTool,
            WhiteboardUpdateInput(from_="不存在", to="替换"),
        )


def test_whiteboard_sessions_are_independent() -> None:
    """Each conversation owns its own board and revision counter."""

    _, first_state = _run_initial(
        WhiteboardAddTool,
        WhiteboardAddInput(text="会话一内容"),
        session_id="session-a",
        call_id="call-a",
    )
    _, second_state = _run_initial(
        WhiteboardAddTool,
        WhiteboardAddInput(text="会话二内容"),
        session_id="session-b",
        call_id="call-b",
    )

    first_snapshot = json.loads(first_state.output.to_content())  # type: ignore[union-attr]
    second_snapshot = json.loads(second_state.output.to_content())  # type: ignore[union-attr]
    assert first_snapshot["text"] == "会话一内容"
    assert first_snapshot["revision"] == 1
    assert second_snapshot["text"] == "会话二内容"
    assert second_snapshot["revision"] == 1

    _, first_again = _run_initial(
        WhiteboardAddTool,
        WhiteboardAddInput(text="追加到会话一"),
        session_id="session-a",
        call_id="call-a-2",
    )
    first_again_snapshot = json.loads(first_again.output.to_content())  # type: ignore[union-attr]
    assert first_again_snapshot["text"] == "会话一内容\n追加到会话一"
    assert first_again_snapshot["revision"] == 2
    assert get_whiteboard_store("session-b").snapshot()["revision"] == 1


def test_whiteboard_persists_per_session_across_store_restarts(
    tmp_path: Any,
) -> None:
    """Session boards survive a sidecar-style restart on the same directory."""

    store_directory = tmp_path / "data" / "whiteboards"
    configure_whiteboard_data_directory(store_directory)
    get_whiteboard_store("session-a").add_text("会话一内容")
    get_whiteboard_store("session-b").add_text("会话二内容")

    configure_whiteboard_data_directory(store_directory)
    assert get_whiteboard_store("session-a").snapshot()["text"] == "会话一内容"
    assert get_whiteboard_store("session-b").snapshot()["text"] == "会话二内容"
    assert (store_directory / "session-a.json").is_file()
    assert (store_directory / "session-b.json").is_file()


def test_whiteboard_input_validates_text_limits() -> None:
    """Reject oversized text chunks through the input schema."""

    with pytest.raises(ValidationError):
        WhiteboardAddInput(text="长" * (MAX_WHITEBOARD_TEXT_CHUNK + 1))
    with pytest.raises(ValidationError):
        WhiteboardDeleteInput(text="")


def test_whiteboard_wrapper_emits_structured_payload() -> None:
    """The tool UI observer attaches the parsed text snapshot as payload."""

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
            [WhiteboardAddTool],
            binding=ToolUIBinding(
                tool_id="builtin:whiteboard",
                update_every_s=-1,
            ),
            broker=broker,  # type: ignore[arg-type]
        )[0]
        tool_state = WhiteboardState(call_id="wrapped-call")
        tool_state.metadata["session_id"] = "session-1"
        initial = await wrapped.aemit_initial(
            "wrapped-call",
            WhiteboardAddInput(text="包装链路"),
            tool_state,
            {},
        )
        return initial, broker

    initial, broker = asyncio.run(scenario())

    assert isinstance(initial, Running)
    assert len(broker.emits) == 1
    emit = broker.emits[0]
    assert emit["payload"]["action"] == "add_text"
    assert emit["payload"]["revision"] == 1
    assert emit["payload"]["text"] == "包装链路"
