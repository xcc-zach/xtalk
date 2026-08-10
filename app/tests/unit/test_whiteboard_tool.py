"""Contract tests for the per-session desktop text whiteboard tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError
from xtalk.models.agents.tools import SyncTool, ToolEngine

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
    WhiteboardOutput,
    WhiteboardUpdateInput,
    WhiteboardUpdateTool,
)


@pytest.fixture(autouse=True)
def _reset_whiteboard_stores() -> Any:
    """Restore a fresh in-memory registry around every test."""

    reset_whiteboard_stores()
    yield
    reset_whiteboard_stores()


def _run_tool(
    tool: type[Any],
    tool_input: Any,
    *,
    session_id: str = "session-1",
) -> WhiteboardOutput:
    """Run one immediate whiteboard operation for a backend session."""

    return tool.invoke(
        tool_input,
        {"session_id": session_id},
    )


def test_whiteboard_add_appends_text_and_advances_revision() -> None:
    """Appending text joins blocks with one newline and bumps the revision."""

    output = _run_tool(
        WhiteboardAddTool,
        WhiteboardAddInput(text="# 第一段"),
    )

    assert issubclass(WhiteboardAddTool, SyncTool)
    snapshot = json.loads(output.to_content())
    assert snapshot["action"] == "add_text"
    assert snapshot["success"] is True
    assert snapshot["text"] == "# 第一段"
    assert snapshot["revision"] == 1
    assert "call_id" not in snapshot
    assert snapshot["message"]

    second_output = _run_tool(
        WhiteboardAddTool,
        WhiteboardAddInput(text="第二段"),
    )
    second_snapshot = json.loads(second_output.to_content())
    assert second_snapshot["text"] == "# 第一段\n第二段"
    assert second_snapshot["revision"] == 2


def test_whiteboard_fetch_returns_full_text_without_mutating() -> None:
    """Fetch reports the whole document and never changes the revision."""

    get_whiteboard_store("session-1").add_text("演示内容")
    output = _run_tool(WhiteboardFetchTool, WhiteboardFetchTool.input_type())

    snapshot = json.loads(output.to_content())
    assert snapshot["text"] == "演示内容"
    assert snapshot["revision"] == 1
    assert snapshot["action"] == "fetch_text"
    assert get_whiteboard_store("session-1").snapshot()["revision"] == 1


def test_whiteboard_delete_removes_exact_match() -> None:
    """Deleting one block removes it while preserving the rest."""

    store = get_whiteboard_store("session-1")
    store.add_text("# 标题")
    store.add_text("需要删除的内容")
    store.add_text("保留内容")

    output = _run_tool(
        WhiteboardDeleteTool,
        WhiteboardDeleteInput(text="需要删除的内容"),
    )

    snapshot = json.loads(output.to_content())
    assert snapshot["revision"] == 4
    assert snapshot["text"] == "# 标题\n保留内容"


def test_whiteboard_delete_rejects_missing_text() -> None:
    """Deleting text that is not on the board raises a clear error."""

    with pytest.raises(ValueError, match="not found"):
        _run_tool(
            WhiteboardDeleteTool,
            WhiteboardDeleteInput(text="不存在的内容"),
        )


def test_whiteboard_update_replaces_every_match() -> None:
    """Updating text replaces each occurrence and keeps the rest intact."""

    get_whiteboard_store("session-1").add_text("旧文本 保留 旧文本")

    output = _run_tool(
        WhiteboardUpdateTool,
        WhiteboardUpdateInput(from_="旧文本", to="新文本"),
    )

    snapshot = json.loads(output.to_content())
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
        _run_tool(
            WhiteboardUpdateTool,
            WhiteboardUpdateInput(from_="不存在", to="替换"),
        )


def test_whiteboard_sessions_are_independent() -> None:
    """Each conversation owns its own board and revision counter."""

    first_output = _run_tool(
        WhiteboardAddTool,
        WhiteboardAddInput(text="会话一内容"),
        session_id="session-a",
    )
    second_output = _run_tool(
        WhiteboardAddTool,
        WhiteboardAddInput(text="会话二内容"),
        session_id="session-b",
    )

    first_snapshot = json.loads(first_output.to_content())
    second_snapshot = json.loads(second_output.to_content())
    assert first_snapshot["text"] == "会话一内容"
    assert first_snapshot["revision"] == 1
    assert second_snapshot["text"] == "会话二内容"
    assert second_snapshot["revision"] == 1

    first_again = _run_tool(
        WhiteboardAddTool,
        WhiteboardAddInput(text="追加到会话一"),
        session_id="session-a",
    )
    first_again_snapshot = json.loads(first_again.to_content())
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

    async def scenario() -> tuple[WhiteboardOutput, _RecordingBroker]:
        broker = _RecordingBroker()
        wrapped = wrap_tools_with_ui(
            [WhiteboardAddTool],
            binding=ToolUIBinding(
                tool_id="builtin:whiteboard",
                update_every_s=-1,
            ),
            broker=broker,  # type: ignore[arg-type]
        )[0]
        output = await wrapped.ainvoke(
            WhiteboardAddInput(text="包装链路"),
            {"session_id": "session-1"},
        )
        return output, broker

    output, broker = asyncio.run(scenario())

    assert isinstance(output, WhiteboardOutput)
    assert get_whiteboard_store("session-1").snapshot()["text"] == "包装链路"
    assert get_whiteboard_store().snapshot()["text"] == ""
    assert len(broker.emits) == 1
    emit = broker.emits[0]
    assert emit["running"] is False
    assert emit["status"] == "complete"
    assert emit["session_id"] == "session-1"
    assert emit["payload"]["action"] == "add_text"
    assert emit["payload"]["revision"] == 1
    assert emit["payload"]["text"] == "包装链路"


def test_whiteboard_tool_engine_returns_one_final_result() -> None:
    """Finish in the initial call without scheduling an asynchronous update."""

    async def scenario() -> tuple[Any, list[tuple[Any, Any]]]:
        engine = ToolEngine(
            tools=[WhiteboardAddTool],
            state={"session_id": "session-1"},
        )
        updates: list[tuple[Any, Any]] = []
        engine.on_async_tool_update(
            lambda tool_call, tool_message: updates.append(
                (tool_call, tool_message)
            )
        )
        try:
            result = await engine.ainvoke(
                {
                    "id": "whiteboard-engine-call",
                    "name": "add_text",
                    "args": {"text": "单次最终结果"},
                }
            )
            await asyncio.sleep(0)
            return result, updates
        finally:
            await engine.shutdown()

    result, updates = asyncio.run(scenario())
    snapshot = json.loads(str(result.content))

    assert result.tool_call_id == "whiteboard-engine-call"
    assert result.name == "add_text"
    assert snapshot["text"] == "单次最终结果"
    assert snapshot["revision"] == 1
    assert updates == []
