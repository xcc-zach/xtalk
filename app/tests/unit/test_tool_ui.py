"""Contract tests for read-only App observation of native tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import WebSocketDisconnect
from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    SyncTool,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)

from backend.tool_ui import (
    MAX_TOOL_UI_EMIT_PAYLOAD_BYTES,
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


class _ObservedSyncTool(SyncTool):
    """Small immediate-tool contract fixture."""

    name = "observed_sync"
    structured_payload = True

    @classmethod
    def invoke(
        cls,
        tool_input: _Input,
        global_state: Any,
    ) -> _Output:
        """Return one structured final result immediately."""

        del cls, global_state
        return _Output(value=json.dumps({"value": tool_input.value}))


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


class _CancelledTool(AsyncTool):
    """Asynchronous tool that remains active until its task is cancelled."""

    name = "cancelled"

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> Running:
        """Return the initial running result."""

        del cls, tool_call_id, tool_input, tool_state, global_state
        return Running("initial")

    @classmethod
    async def aemit_updates(
        cls,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> AsyncIterator[ToolResult[_Output]]:
        """Wait indefinitely so the test can cancel the update task."""

        del cls, tool_input, tool_state, global_state
        await asyncio.Event().wait()
        yield Running("unreachable")

    @classmethod
    def status(
        cls,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> str:
        """Return the status retained in the cancellation history emit."""

        del cls, tool_input, tool_state, global_state
        return "Tool stopped"


class _StructuredTool(AsyncTool):
    """Asynchronous tool that emits JSON content and declares payloads."""

    name = "structured"
    structured_payload = True

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> Running:
        """Return the first JSON snapshot containing the call id."""

        del cls, tool_input, tool_state, global_state
        return Running(
            json.dumps(
                {
                    "kind": "board",
                    "call_id": tool_call_id,
                    "revision": 1,
                }
            )
        )

    @classmethod
    def emit_updates(
        cls,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> Iterator[ToolResult[_Output]]:
        """Yield one running and one finished JSON snapshot."""

        del cls, tool_input, global_state
        yield Running(
            json.dumps(
                {
                    "kind": "board",
                    "call_id": tool_state.call_id,
                    "revision": 2,
                }
            )
        )
        yield Finished(
            _Output(
                value=json.dumps(
                    {
                        "kind": "board",
                        "call_id": tool_state.call_id,
                        "revision": 3,
                    }
                )
            )
        )

    @classmethod
    def status(
        cls,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> str:
        """Return a deterministic status string."""

        del cls, tool_input, global_state
        return f"revision={tool_state.progress}"


class _MalformedStructuredTool(AsyncTool):
    """Structured-declaring tool whose content is not valid JSON."""

    name = "malformed_structured"
    structured_payload = True

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> Running:
        """Return plain text that contains the required call id."""

        del cls, tool_input, tool_state, global_state
        return Running(f"Working on {tool_call_id}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: _Input,
        tool_state: _State,
        global_state: Any,
    ) -> Iterator[ToolResult[_Output]]:
        """Yield one finished plain-text result."""

        del cls, tool_input, tool_state, global_state
        yield Finished(_Output(value="Finished without JSON"))


class _RecordingBroker:
    """Capture published observations without opening a WebSocket."""

    def __init__(self) -> None:
        """Initialize empty event lists."""

        self.statuses: list[dict[str, Any]] = []
        self.emits: list[dict[str, Any]] = []
        self.finished_calls: list[str] = []
        self.ui_tool_names: set[str] = set()

    def register_ui_tool(self, tool_name: str) -> None:
        """Record the wrapped tool name like the production broker."""

        self.ui_tool_names.add(tool_name)

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
    global_state = {"session_id": "session-1"}
    initial = await wrapped.aemit_initial(
        "call-1",
        tool_input,
        tool_state,
        global_state,
    )
    updates = [
        update
        async for update in wrapped.aemit_updates(
            tool_input,
            tool_state,
            global_state,
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
    assert all(
        event["session_id"] == "session-1"
        for event in [*broker.statuses, *broker.emits]
    )


def test_wrapper_observes_sync_tool_with_one_terminal_emit() -> None:
    """Publish one completed UI event for an immediate native tool."""

    async def scenario() -> tuple[_Output, _RecordingBroker]:
        broker = _RecordingBroker()
        wrapped = wrap_tools_with_ui(
            [_ObservedSyncTool],
            binding=ToolUIBinding(tool_id="tool-sync", update_every_s=-1),
            broker=broker,  # type: ignore[arg-type]
        )[0]
        result = await wrapped.ainvoke(
            _Input(value="finished"),
            {"session_id": "session-1"},
        )
        return result, broker

    result, broker = asyncio.run(scenario())

    assert result == _Output(value=json.dumps({"value": "finished"}))
    assert broker.ui_tool_names == {"observed_sync"}
    assert broker.statuses == []
    assert len(broker.emits) == 1
    emit = broker.emits[0]
    assert emit["tool_name"] == "observed_sync"
    assert emit["message"] == json.dumps({"value": "finished"})
    assert emit["status"] == "complete"
    assert emit["running"] is False
    assert emit["payload"] == {"value": "finished"}
    assert emit["session_id"] == "session-1"
    assert isinstance(emit["call_id"], str)
    assert emit["call_id"]


def test_wrapper_publishes_terminal_history_emit_when_cancelled() -> None:
    """Represent cancellation as history UI instead of only clearing live UI."""

    async def scenario() -> _RecordingBroker:
        broker = _RecordingBroker()
        wrapped = wrap_tools_with_ui(
            [_CancelledTool],
            binding=ToolUIBinding(tool_id="tool-1", update_every_s=-1),
            broker=broker,  # type: ignore[arg-type]
        )[0]
        tool_input = _Input(value="input")
        tool_state = _State(call_id="cancelled-call")
        await wrapped.aemit_initial(
            "cancelled-call",
            tool_input,
            tool_state,
            object(),
        )

        async def consume() -> None:
            async for _ in wrapped.aemit_updates(
                tool_input,
                tool_state,
                object(),
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return broker

    broker = asyncio.run(scenario())

    assert broker.emits[-1] == {
        "binding": ToolUIBinding(tool_id="tool-1", update_every_s=-1),
        "tool_name": "cancelled",
        "call_id": "cancelled-call",
        "message": "Tool stopped",
        "status": "Tool stopped",
        "running": False,
        "outcome": "cancelled",
        "session_id": None,
    }


def test_wrapper_attaches_structured_payload_when_declared() -> None:
    """Expose parsed JSON content as a structured emit payload."""

    async def scenario() -> _RecordingBroker:
        broker = _RecordingBroker()
        wrapped = wrap_tools_with_ui(
            [_StructuredTool],
            binding=ToolUIBinding(
                tool_id="tool-structured",
                update_every_s=-1,
            ),
            broker=broker,  # type: ignore[arg-type]
        )[0]
        tool_input = _Input(value="input")
        tool_state = _State(call_id="structured-call")
        await wrapped.aemit_initial(
            "structured-call",
            tool_input,
            tool_state,
            object(),
        )
        async for _ in wrapped.aemit_updates(
            tool_input,
            tool_state,
            object(),
        ):
            pass
        return broker

    broker = asyncio.run(scenario())

    assert [event["payload"] for event in broker.emits] == [
        {"kind": "board", "call_id": "structured-call", "revision": 1},
        {"kind": "board", "call_id": "structured-call", "revision": 2},
        {"kind": "board", "call_id": "structured-call", "revision": 3},
    ]
    assert [event["message"] for event in broker.emits] == [
        json.dumps({"kind": "board", "call_id": "structured-call", "revision": 1}),
        json.dumps({"kind": "board", "call_id": "structured-call", "revision": 2}),
        json.dumps({"kind": "board", "call_id": "structured-call", "revision": 3}),
    ]


def test_wrapper_degrades_when_structured_content_is_not_json() -> None:
    """Keep plain-text emits intact without inventing a payload."""

    async def scenario() -> _RecordingBroker:
        broker = _RecordingBroker()
        wrapped = wrap_tools_with_ui(
            [_MalformedStructuredTool],
            binding=ToolUIBinding(
                tool_id="tool-malformed",
                update_every_s=-1,
            ),
            broker=broker,  # type: ignore[arg-type]
        )[0]
        tool_input = _Input(value="input")
        tool_state = _State(call_id="malformed-call")
        await wrapped.aemit_initial(
            "malformed-call",
            tool_input,
            tool_state,
            object(),
        )
        async for _ in wrapped.aemit_updates(
            tool_input,
            tool_state,
            object(),
        ):
            pass
        return broker

    broker = asyncio.run(scenario())

    assert [event["message"] for event in broker.emits] == [
        "Working on malformed-call",
        "Finished without JSON",
    ]
    assert all(event.get("payload") is None for event in broker.emits)


def test_broker_retains_structured_payload_in_history() -> None:
    """Replay emits with their structured payloads after reconnects."""

    async def scenario() -> list[dict[str, Any]]:
        broker = ToolUIBroker()
        await broker.publish_emit(
            binding=ToolUIBinding(
                tool_id="builtin:whiteboard",
                update_every_s=-1,
            ),
            tool_name="fetch_text",
            call_id="board-call",
            message=json.dumps(
                {
                    "kind": "board",
                    "call_id": "board-call",
                    "revision": 4,
                }
            ),
            status="complete",
            running=False,
            session_id="session-1",
            payload={
                "kind": "board",
                "call_id": "board-call",
                "revision": 4,
            },
        )
        return await broker.snapshot("session-1")

    events = asyncio.run(scenario())

    assert events[0]["message"] == json.dumps(
        {
            "kind": "board",
            "call_id": "board-call",
            "revision": 4,
        }
    )
    assert events[0]["payload"] == {
        "kind": "board",
        "call_id": "board-call",
        "revision": 4,
    }


def test_broker_drops_oversized_payload_but_keeps_message() -> None:
    """Bound structured content without losing the human-readable emit."""

    async def scenario() -> list[dict[str, Any]]:
        broker = ToolUIBroker()
        await broker.publish_emit(
            binding=ToolUIBinding(
                tool_id="builtin:whiteboard",
                update_every_s=-1,
            ),
            tool_name="fetch_text",
            call_id="large-call",
            message="small message",
            status="complete",
            running=False,
            session_id="session-1",
            payload={"blob": "x" * (MAX_TOOL_UI_EMIT_PAYLOAD_BYTES + 1)},
        )
        return await broker.snapshot("session-1")

    events = asyncio.run(scenario())

    assert events[0]["message"] == "small message"
    assert "payload" not in events[0]


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
            session_id="session-1",
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
            session_id="session-1",
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
            "outcome": "complete",
            "emittedAt": websocket.messages[0]["emittedAt"],
        }
    ]


def test_cancelled_emit_replaces_live_status_in_broker_snapshot() -> None:
    """Keep the cancellation history card and remove the matching live card."""

    async def scenario() -> list[dict[str, Any]]:
        broker = ToolUIBroker()
        binding = ToolUIBinding(tool_id="builtin:timer", update_every_s=0.5)
        await broker.publish_status(
            binding=binding,
            tool_name="timer",
            call_id="timer-call",
            status="10 of 60 seconds",
            running=True,
            session_id="session-1",
        )
        await broker.publish_emit(
            binding=binding,
            tool_name="timer",
            call_id="timer-call",
            message="Timer stopped after 10 seconds out of 60 seconds.",
            status="Timer stopped after 10 seconds out of 60 seconds.",
            running=False,
            outcome="cancelled",
            session_id="session-1",
        )
        return await broker.snapshot("session-1")

    events = asyncio.run(scenario())

    assert len(events) == 1
    assert events[0]["type"] == "tool_ui.emit"
    assert events[0]["callId"] == "timer-call"
    assert events[0]["running"] is False
    assert events[0]["outcome"] == "cancelled"


def test_timer_example_uses_unlabeled_live_and_history_ui() -> None:
    """Keep history compact without adding visible mode annotations."""

    app_root = Path(__file__).parents[2]
    sources = [
        (app_root / root / "tools" / "timer" / "ui" / "index.html")
        .read_text(encoding="utf-8")
        for root in ("examples", "resources")
    ]

    for source in sources:
        assert "window.xtalkToolUI.status" in source
        assert "window.xtalkToolUI.emit" in source
        assert "window.xtalkToolUI.context" in source
        assert "const { language, mode }" in source
        assert ':root[data-mode="history"] .status' in source
        assert ':root[data-mode="history"] .message' in source
        assert ':root[data-mode="history"] .progress' in source
        assert ':root[data-mode="history"] #elapsed' in source
        assert ':root[data-mode="history"] .badge' in source
        assert 'title: "计时器"' in source
        assert 'running: "运行中"' in source
        assert 'complete: "已完成"' in source
        assert 'cancelled: "已取消"' in source
        assert '=== "cancelled"' in source
        assert 'seconds: "秒"' in source
        assert "copy.seconds" in source
        assert "History UI" not in source
        assert "Live UI" not in source


def test_tool_ui_runtime_injects_resolved_application_language() -> None:
    """Give each sandbox the App language and rebuild it after a change."""

    app_root = Path(__file__).parents[2]
    frame_logic = (app_root / "ui" / "tool-ui-frame.ts").read_text(
        encoding="utf-8"
    )
    app_logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    assert "language: SupportedLanguage" in frame_logic
    assert "context: Object.freeze({ mode, language })" in frame_logic
    assert "document.documentElement.lang = language" in frame_logic
    assert "resetToolUIRowsIfLanguageChanged();" in app_logic
    reset_logic = app_logic.split(
        "function resetToolUIRowsIfLanguageChanged()", maxsplit=1
    )[1].split("function renderLiveToolPanel", maxsplit=1)[0]
    assert "getResolvedLanguage()" in reset_logic
    assert "row.frame?.destroy()" in reset_logic
    assert "row.element.remove()" in reset_logic
    assert "toolUIRows.clear()" in reset_logic
    assert "row.mode,\n        getResolvedLanguage()," in app_logic


def test_codex_tool_ui_localizes_actions_and_runtime_status() -> None:
    """Keep the shipped Codex card consistent with the selected language."""

    source = (
        Path(__file__).parents[2]
        / "resources"
        / "tools"
        / "codex"
        / "ui"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert "window.xtalkToolUI.context" in source
    assert 'codex_session_search: "搜索会话"' in source
    assert 'codex_session_create: "新建会话"' in source
    assert '"Codex is working": "Codex 正在处理"' in source
    assert '"Codex completed": "Codex 已完成"' in source
    assert 'failed: "失败"' in source
    assert 'cancelled: "已取消"' in source
    assert 'outcome === "cancelled"' in source
    assert "const isInternalAsyncNotice" in source
    assert "The final result will be delivered asynchronously" in source
    assert 'isInternalAsyncNotice(message) ? "" : message' in source


def test_codex_tool_ui_renders_safe_offline_markdown() -> None:
    """Render common Markdown and GFM tables without injecting raw HTML."""

    source = (
        Path(__file__).parents[2]
        / "resources"
        / "tools"
        / "codex"
        / "ui"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert "const renderMarkdown" in source
    assert "const appendInlineMarkdown" in source
    assert "const tableAlignment" in source
    assert 'document.createElement("table")' in source
    assert 'document.createElement("pre")' in source
    assert 'renderMarkdown(byId("message"), displayMessage)' in source
    assert "container.replaceChildren(fragment)" in source
    assert "innerHTML" not in source


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


def test_history_tool_ui_is_anchored_before_the_current_assistant_reply() -> None:
    """Keep delayed tool observations ahead of the reply they produced."""

    logic = (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")

    assert "findToolUIAnchorMessageIndex(latestSnapshot.messages)" in logic
    anchor_logic = logic.split(
        "function findToolUIAnchorMessageIndex", maxsplit=1
    )[1].split("function switchToolUISession", maxsplit=1)[0]
    assert 'messages[index]?.role === "user"' in anchor_logic
    assert 'messages[index]?.role === "assistant"' in anchor_logic
    assert "return index;" in anchor_logic
    assert "item.anchorMessageIndex = Math.min(" in logic


def test_history_tool_ui_can_embed_inside_assistant_message() -> None:
    """Render tool rows inside a message at the recorded generation offset."""

    logic = (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")

    render = logic.split(
        "function renderSnapshot(", maxsplit=1
    )[1].split("function createMessageActionIcon", maxsplit=1)[0]
    assert "message-content-group" in render
    assert 'typeof item.event.textOffset === "number"' in render
    assert "embeddedToolItemIds" in render
    assert "clampMessageOffset(" in render
    assert "!embeddedToolItemIds.has(item.id)" in render
    assert "createMessageContentSpan(" in render

    clamp = logic.split(
        "function clampMessageOffset(", maxsplit=1
    )[1].split("function createMessageCopyButton", maxsplit=1)[0]
    assert "Math.max(0, Math.min(length, Math.trunc(offset)))" in clamp


def test_tool_card_offsets_advance_to_sentence_boundary() -> None:
    """Place tool cards after the sentence being spoken when the call fired."""

    app_root = Path(__file__).parents[2]
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    boundary = logic.split(
        "function advanceToSentenceBoundary(", maxsplit=1
    )[1].split("function createMessageRowState", maxsplit=1)[0]
    assert "offset >= text.length" in boundary
    assert "isSentenceBoundary(text[offset - 1]!)" in boundary
    assert "return index + 1;" in boundary
    assert '"。！？；!?;\\n".includes(character)' in logic
    assert "advanceToSentenceBoundary(" in logic

    styles = (app_root / "ui" / "styles.css").read_text(encoding="utf-8")
    assert (
        '.message[data-final="false"] .message-content:last-of-type::after'
        in styles
    )
    assert ".message[data-final=\"false\"]::after" not in styles


def test_embedded_tool_rows_stay_mounted_across_renders() -> None:
    """Reuse message rows so embedded tool iframes are not reloaded."""

    logic = (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")

    render = logic.split(
        "function renderSnapshot(", maxsplit=1
    )[1].split("function createMessageRowState", maxsplit=1)[0]
    assert "messageRowStates.get(message.id)" in render
    assert "createMessageRowState(message)" in render
    assert "updateMessageRowState(" in render

    reconcile = logic.split(
        "function reconcileMessageContent(", maxsplit=1
    )[1].split("function createMessageActionIcon", maxsplit=1)[0]
    assert "span.textContent = part.text" in reconcile
    assert "host.insertBefore(element, current)" in reconcile
    assert "getOrCreateToolUIRow(item).element" in logic

    switch = logic.split(
        "function switchToolUISession(", maxsplit=1
    )[1].split("function resetToolUIRowsIfLanguageChanged", maxsplit=1)[0]
    assert "messageRowStates.clear()" in switch


def test_chat_auto_scroll_yields_to_user_scroll_position() -> None:
    """Renders must not yank the viewport down while the user scrolls up."""

    logic = (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")

    render = logic.split(
        "function renderSnapshot(", maxsplit=1
    )[1].split("function createMessageContentSpan", maxsplit=1)[0]
    assert "isScrolledNearBottom(elements.messages)" in render
    assert "elements.messages.scrollTop = elements.messages.scrollHeight" in render

    helper = logic.split(
        "function isScrolledNearBottom(", maxsplit=1
    )[1].split("/**\n * Creates one plain-text content span", maxsplit=1)[0]
    assert "scrollHeight - container.scrollTop - container.clientHeight < 64" in helper


def test_sidebar_delete_flow_confirms_before_deleting() -> None:
    """Hover delete buttons ask for confirmation before removing a chat."""

    app_root = Path(__file__).parents[2]
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")
    markup = (app_root / "ui" / "index.html").read_text(encoding="utf-8")
    styles = (app_root / "ui" / "styles.css").read_text(encoding="utf-8")
    i18n = (app_root / "ui" / "i18n.ts").read_text(encoding="utf-8")

    render = logic.split(
        "function renderChatSessions()", maxsplit=1
    )[1].split("function updateSessionControls", maxsplit=1)[0]
    assert 'className = "chat-session-row"' in render
    assert 'row.classList.toggle("is-active"' in render
    assert 'className = "chat-session-delete"' in render
    assert "chatSessionRowStates.get(session.id)" in render
    assert "reconcileStableChildren(elements.chatSessionList, rows)" in render
    assert "openDeleteSessionDialog(session.id, currentSession?.title ?? null)" in render

    delete_logic = logic.split(
        "async function deletePendingSession()", maxsplit=1
    )[1].split("async function refreshChatSessions", maxsplit=1)[0]
    assert 'method: "DELETE"' in delete_logic
    assert "X-XTalk-App-Token" in delete_logic
    assert "sidebar.deleteConfirm" in logic
    assert "showModal()" in logic
    assert "localStorage.removeItem" in delete_logic

    assert 'id="delete-session-dialog"' in markup
    assert 'id="delete-session-confirm-button"' in markup
    assert "settings-dialog-button-danger" in markup
    assert ".chat-session-delete" in styles
    assert ".chat-session-row.is-active" in styles
    assert '"sidebar.deleteConfirm"' in i18n


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


def test_tool_ui_frame_redelivers_last_event_after_reload() -> None:
    """A reloaded iframe must not fall back to the tool's initial HTML."""

    frame_logic = (
        Path(__file__).parents[2] / "ui" / "tool-ui-frame.ts"
    ).read_text(encoding="utf-8")

    status_logic = frame_logic.split(
        "status(event: ToolUIStatusEvent): void {",
        maxsplit=1,
    )[1].split("emit(event: ToolUIEmitEvent): void {", maxsplit=1)[0]
    emit_logic = frame_logic.split(
        "emit(event: ToolUIEmitEvent): void {",
        maxsplit=1,
    )[1].split("/** Removes global listeners", maxsplit=1)[0]
    assert "this.#lastStatus = event" in status_logic
    assert "this.#lastEmit = event" in emit_logic

    flush_logic = frame_logic.split(
        "#flush(): void {",
        maxsplit=1,
    )[1]
    assert "this.#lastStatus !== null" in flush_logic
    assert "this.#lastEmit !== null" in flush_logic
    assert 'type: "tool_ui.emit"' in flush_logic
    assert 'type: "tool_ui.status"' in flush_logic


def test_tool_ui_frame_measures_content_without_viewport_feedback() -> None:
    """Size history cards to body content instead of the iframe viewport."""

    app_root = Path(__file__).parents[2]
    frame_logic = (app_root / "ui" / "tool-ui-frame.ts").read_text(
        encoding="utf-8"
    )
    styles = (app_root / "ui" / "styles.css").read_text(encoding="utf-8")

    assert "measuredElement.getBoundingClientRect().height" in frame_logic
    assert (
        "new ResizeObserver(reportHeight).observe(measuredElement)"
        in frame_logic
    )
    assert "document.documentElement.scrollHeight" not in frame_logic
    assert "#messages > .tool-ui-row + .tool-ui-row" in styles
    assert "#messages > .tool-ui-row + .message-row" in styles
    assert "#messages > .message-row + .tool-ui-row" in styles
    assert "margin-top: -6px" in styles


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


def test_terminal_emit_updates_history_row_in_place() -> None:
    """Keep one stable row per call so tool iframes never reload on emit."""

    app_root = Path(__file__).parents[2]
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")
    emit_branch = logic.split(
        'if (event.type === "tool_ui.emit") {',
        maxsplit=1,
    )[1].split("} else if (sessionId === activeToolUISessionId) {", maxsplit=1)[0]

    assert "if (!event.running" in emit_branch
    assert "toolUILive.delete(event.callId)" in emit_branch
    assert "removeToolUIRow(`live:${event.callId}`)" in emit_branch
    assert "historyItem.event.callId === event.callId" not in emit_branch
    assert 'id: `history:${event.callId}`' in emit_branch
    assert "toolUIRows.get(item.id)?.frame?.emit(item.event)" in emit_branch

    append = logic.split(
        "function appendToolUIHistory(", maxsplit=1
    )[1].split("function readToolUIHistory", maxsplit=1)[0]
    assert "existingIndex !== -1" in append
    assert "item.anchorMessageIndex = Math.min(" in append
    assert "history[existingIndex] = item" in append
