"""Contract tests for read-only App observation of asynchronous tools."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)

from backend.tool_ui import ToolUIBinding, ToolUIBroker, wrap_tools_with_ui


class _Input(ToolInput):
    """Input used by the observed test tool."""

    value: str


@dataclass
class _State(ToolState):
    """State used by the observed test tool."""

    progress: int = 0


class _Output(ToolOutput):
    """Final output used by the observed test tool."""

    value: str

    def to_content(self) -> str:
        """Return the output text."""

        return self.value


class _ObservedTool(AsyncTool):
    """Small asynchronous-tool contract fixture."""

    name = "observed"

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> Running:
        """Return the initial unchanged tool result."""

        del cls, tool_call_id, tool_input, tool_state, global_state
        return Running("initial")

    @classmethod
    def emit_updates(
        cls,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> Iterator[ToolResult[_Output]]:
        """Yield one running and one finished unchanged tool result."""

        del cls, tool_input, global_state
        tool_state.progress = 1
        yield Running("progress")
        tool_state.progress = 2
        yield Finished(_Output(value="finished"))

    @classmethod
    def status(
        cls,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> str:
        """Return deterministic fixture progress."""

        del cls, tool_input, global_state
        return f"progress={tool_state.progress}"


class _RecordingBroker:
    """Capture published observations without opening a WebSocket."""

    def __init__(self) -> None:
        """Initialize empty event lists."""

        self.statuses: list[dict[str, Any]] = []
        self.emits: list[dict[str, Any]] = []
        self.finished_calls: list[str] = []

    async def publish_status(self, **payload: Any) -> None:
        """Record one status payload."""

        self.statuses.append(payload)

    async def publish_emit(self, **payload: Any) -> None:
        """Record one emit payload."""

        self.emits.append(payload)

    def finish_call(self, call_id: str) -> None:
        """Record explicit terminal cleanup."""

        self.finished_calls.append(call_id)


async def _exercise_wrapped_tool() -> tuple[
    list[ToolResult[_Output]],
    _RecordingBroker,
]:
    """Run the complete wrapped lifecycle with periodic polling disabled."""

    broker = _RecordingBroker()
    wrapped = wrap_tools_with_ui(
        [_ObservedTool],
        binding=ToolUIBinding(tool_id="tool-1", update_every_s=-1),
        broker=broker,  # type: ignore[arg-type]
    )[0]
    tool_input = _Input(value="input")
    tool_state = _State(call_id="call-1")
    initial = await wrapped.aemit_initial(
        "call-1",
        tool_input,
        tool_state,
        object(),
    )
    updates = [
        update
        async for update in wrapped.aemit_updates(
            tool_input,
            tool_state,
            object(),
        )
    ]
    return [initial, *updates], broker


def test_wrapper_preserves_results_and_observes_each_emit() -> None:
    """Keep tool logic unchanged while producing display-only observations."""

    results, broker = asyncio.run(_exercise_wrapped_tool())

    assert results[0] == Running("initial")
    assert results[1] == Running("progress")
    assert results[2] == Finished(_Output(value="finished"))
    assert [event["message"] for event in broker.emits] == [
        "initial",
        "progress",
        "finished",
    ]
    assert [event["running"] for event in broker.emits] == [
        True,
        True,
        False,
    ]
    assert [event["status"] for event in broker.statuses] == [
        "progress=0",
        "progress=2",
    ]


async def _consume_one_time_frame() -> tuple[str | None, str | None]:
    """Create and consume one frame ticket twice."""

    broker = ToolUIBroker()
    ticket = await broker.create_frame_ticket("<!doctype html><p>frame</p>")
    first = await broker.consume_frame_ticket(ticket)
    second = await broker.consume_frame_ticket(ticket)
    return first, second


def test_frame_ticket_is_one_time() -> None:
    """Expose prepared HTML without putting the launch token in the frame URL."""

    first, second = asyncio.run(_consume_one_time_frame())

    assert first == "<!doctype html><p>frame</p>"
    assert second is None


def test_timer_example_uses_unlabeled_live_and_history_ui() -> None:
    """Keep mode annotations out of the user-visible timer card."""

    source = (
        Path(__file__).parents[2]
        / "examples"
        / "tools"
        / "timer"
        / "ui"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert "window.xtalkToolUI.status" in source
    assert "window.xtalkToolUI.emit" in source
    assert "History UI" not in source
    assert "Live UI" not in source


def test_chat_topbar_uses_collapsible_live_tool_status() -> None:
    """Keep the chat header contextual instead of repeating the product name."""

    app_root = Path(__file__).parents[2]
    markup = (app_root / "ui" / "index.html").read_text(encoding="utf-8")
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    assert '<div class="brand">XTalk</div>' not in markup
    assert 'id="live-tool-status-toggle"' in markup
    assert 'id="live-tool-content"' in markup
    assert "renderLiveToolPanel" in logic
