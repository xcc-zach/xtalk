"""Contract tests for read-only App observation of asynchronous tools."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import WebSocketDisconnect
from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)

from backend.tool_ui import (
    MAX_TOOL_UI_FRAME_TICKETS,
    ToolUIBinding,
    ToolUIBroker,
    wrap_tools_with_ui,
)


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


async def _read_reloadable_frame() -> list[str | None]:
    """Read one frame ticket through repeated WKWebView reloads."""

    broker = ToolUIBroker()
    ticket = await broker.create_frame_ticket("<!doctype html><p>frame</p>")
    return [await broker.consume_frame_ticket(ticket) for _ in range(12)]


def test_frame_ticket_allows_repeated_webkit_reloads_during_runtime() -> None:
    """Permit every internal reload while the backend runtime is alive."""

    reads = asyncio.run(_read_reloadable_frame())

    assert reads == ["<!doctype html><p>frame</p>"] * 12


def test_frame_ticket_store_evicts_the_oldest_document() -> None:
    """Bound retained runtime documents while keeping recent frames readable."""

    async def scenario() -> tuple[str | None, str | None]:
        broker = ToolUIBroker()
        tickets = [
            await broker.create_frame_ticket(f"<p>{index}</p>")
            for index in range(MAX_TOOL_UI_FRAME_TICKETS + 1)
        ]
        return (
            await broker.consume_frame_ticket(tickets[0]),
            await broker.consume_frame_ticket(tickets[-1]),
        )

    oldest, newest = asyncio.run(scenario())

    assert oldest is None
    assert newest == f"<p>{MAX_TOOL_UI_FRAME_TICKETS}</p>"


class _ReplayWebSocket:
    """Minimal WebSocket that binds once and records replayed events."""

    def __init__(self) -> None:
        """Initialize an empty outbound event list."""

        self.accepted = False
        self.messages: list[dict[str, Any]] = []
        self._received = False

    async def accept(self) -> None:
        """Record WebSocket acceptance."""

        self.accepted = True

    async def receive_json(self) -> dict[str, Any]:
        """Bind once, then close the fake connection."""

        if self._received:
            raise WebSocketDisconnect()
        self._received = True
        return {"type": "bind_session", "sessionId": "session-1"}

    async def send_json(self, payload: dict[str, Any]) -> None:
        """Record one replayed status payload."""

        self.messages.append(payload)


def test_broker_replays_running_status_after_app_channel_reconnects() -> None:
    """Show active tool UI even when its initial event preceded the App socket."""

    async def scenario() -> _ReplayWebSocket:
        broker = ToolUIBroker()
        await broker.publish_status(
            binding=ToolUIBinding(tool_id="builtin:codex", update_every_s=1),
            tool_name="codex_session_create",
            call_id="call-1",
            status="Codex is working",
            running=True,
        )
        websocket = _ReplayWebSocket()
        await broker.serve(websocket)  # type: ignore[arg-type]
        return websocket

    websocket = asyncio.run(scenario())

    assert websocket.accepted is True
    assert websocket.messages == [
        {
            "type": "tool_ui.status",
            "toolId": "builtin:codex",
            "toolName": "codex_session_create",
            "callId": "call-1",
            "sessionId": "session-1",
            "sequence": 1,
            "status": "Codex is working",
            "running": True,
            "updatedAt": websocket.messages[0]["updatedAt"],
        }
    ]


def test_broker_replays_history_emit_after_app_channel_connects() -> None:
    """Recover history UI emitted before the App WebSocket was ready."""

    async def scenario() -> _ReplayWebSocket:
        broker = ToolUIBroker()
        await broker.publish_emit(
            binding=ToolUIBinding(tool_id="builtin:codex", update_every_s=1),
            tool_name="codex_session_create",
            call_id="call-1",
            message="Codex completed",
            status="Complete",
            running=False,
        )
        websocket = _ReplayWebSocket()
        await broker.serve(websocket)  # type: ignore[arg-type]
        return websocket

    websocket = asyncio.run(scenario())

    assert websocket.messages == [
        {
            "type": "tool_ui.emit",
            "toolId": "builtin:codex",
            "toolName": "codex_session_create",
            "callId": "call-1",
            "sessionId": "session-1",
            "sequence": 1,
            "message": "Codex completed",
            "status": "Complete",
            "running": False,
            "emittedAt": websocket.messages[0]["emittedAt"],
        }
    ]


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
    assert "timelineItems = [...toolUIHistory]" in logic
    assert "...toolUILive.values()" in logic
    live_render = logic.split(
        "function renderLiveToolPanel()", maxsplit=1
    )[1].split("function getOrCreateToolUIRow", maxsplit=1)[0]
    assert "reconcileStableChildren" in live_render
    assert "...items.map((item) => getOrCreateToolUIRow" not in live_render
    assert "container.insertBefore(child, current)" in logic
    assert "reconcileStableChildren(elements.messages, timelineElements)" in logic
    assert "elements.messages.replaceChildren(...timelineElements)" not in logic


def test_tool_ui_capabilities_settle_and_can_recover() -> None:
    """Do not destroy a frame on an intermediate hook-registration report."""

    app_root = Path(__file__).parents[2]
    frame_logic = (app_root / "ui" / "tool-ui-frame.ts").read_text(
        encoding="utf-8"
    )
    app_logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    assert "let capabilitiesReady = false" in frame_logic
    assert "let capabilitiesQueued = false" in frame_logic
    assert "capabilitiesReady = true" in frame_logic
    assert "queueMicrotask" in frame_logic
    capability_branch = frame_logic.split(
        'if (event.data.type === "tool_ui.capabilities") {', maxsplit=1
    )[1].split("return;", maxsplit=1)[0]
    assert "this.#loaded = true" in capability_branch
    assert "this.#flush();" in capability_branch
    host_constructor = frame_logic.split(
        "constructor(", maxsplit=1
    )[1].split("/** Loads the runtime-scoped frame URL", maxsplit=1)[0]
    assert 'this.element.addEventListener("load"' not in host_constructor
    assert 'event.data.type === "tool_ui.received"' in frame_logic
    assert 'type: "tool_ui.received"' in frame_logic
    assert "event.data.callId === pending.callId" in frame_logic
    assert "event.data.sequence === pending.sequence" in frame_logic
    assert "this.#retryAttempts < 20" in frame_logic
    assert "}, 100);" in frame_logic
    unsupported_branch = app_logic.split(
        "if (!capabilities[requiredCapability]) {",
        maxsplit=1,
    )[1].split("return;", maxsplit=1)[0]
    assert "row.element.hidden = true" in unsupported_branch
    assert "frame.destroy()" not in unsupported_branch


def test_tool_ui_iframe_loads_only_after_entering_the_document() -> None:
    """Avoid unnecessary WebKit reads before the frame enters the document."""

    app_root = Path(__file__).parents[2]
    frame_logic = (app_root / "ui" / "tool-ui-frame.ts").read_text(
        encoding="utf-8"
    )
    app_logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    constructor = frame_logic.split("constructor(", maxsplit=1)[1].split(
        "/** Loads the runtime-scoped frame URL",
        maxsplit=1,
    )[0]
    assert "this.element.src = frameUrl" not in constructor
    assert "mount(): void" in frame_logic
    assert "!this.element.isConnected" in frame_logic
    assert "this.element.src = this.#frameUrl" in frame_logic
    assert "fallback.replaceWith(frame.element);\n    frame.mount();" in app_logic


def test_desktop_adapter_polls_authenticated_tool_ui_snapshots() -> None:
    """Avoid the failing custom-scheme WebKit WebSocket handshake."""

    app_root = Path(__file__).parents[2]
    adapter = (
        app_root / "ui" / "adapters" / "tool-ui-adapter.ts"
    ).read_text(encoding="utf-8")

    assert 'new URL("/app/api/tool-ui/events", this.#origin)' in adapter
    assert '"X-XTalk-App-Token": this.#launchToken' in adapter
    assert "parseToolUIEvents" in adapter
    assert "new WebSocket" not in adapter


def test_terminal_history_emit_removes_matching_live_card() -> None:
    """Do not leave a completed tool pinned in the top live status area."""

    app_root = Path(__file__).parents[2]
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")
    emit_branch = logic.split(
        'if (event.type === "tool_ui.emit") {',
        maxsplit=1,
    )[1].split("const item:", maxsplit=1)[0]

    assert "if (!event.running" in emit_branch
    assert "toolUILive.delete(event.callId)" in emit_branch
    assert "removeToolUIRow(`live:${event.callId}`)" in emit_branch
    assert "historyItem.event.callId === event.callId" in emit_branch
    assert "removeToolUIRow(historyItem.id)" in emit_branch
    assert "candidate.event.callId !== item.event.callId" in logic
