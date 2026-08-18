"""Unit tests for tool-call text offsets delivered to the desktop tool UI."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.desktop_tool_bridge import DesktopToolCallBridge
from backend.tool_ui import ToolUIBinding, ToolUIBroker


def test_bridge_tracks_only_ui_tools_in_fifo_order() -> None:
    """Non-UI engine calls never disturb the per-session offset queue."""

    bridge = DesktopToolCallBridge()
    bridge.register_ui_tool("timer")

    bridge.record_tool_call(session_id="s", name="current_time", offset=5)
    bridge.record_tool_call(session_id="s", name="timer", offset=26)
    bridge.record_tool_call(session_id="s", name="subscribe_async_tool", offset=58)

    assert bridge.consume_tool_offset(session_id="s") == 26
    assert bridge.consume_tool_offset(session_id="s") is None


def test_broker_attaches_offset_to_first_and_terminal_emits() -> None:
    """Every emit of one call carries the offset, including the final one."""

    bridge = DesktopToolCallBridge()
    bridge.register_ui_tool("timer")
    bridge.record_tool_call(session_id="s", name="timer", offset=26)
    broker = ToolUIBroker(bridge=bridge)
    binding = ToolUIBinding(tool_id="builtin:timer", update_every_s=1.0)

    async def scenario() -> None:
        broker._current_session_id = "s"
        await broker.publish_emit(
            binding=binding,
            tool_name="timer",
            call_id="call_1",
            message="Timer started.",
            status="running",
            running=True,
            session_id="s",
        )
        await broker.publish_emit(
            binding=binding,
            tool_name="timer",
            call_id="call_1",
            message="Timer stopped.",
            status="stopped",
            running=False,
            outcome="cancelled",
            session_id="s",
        )

    asyncio.run(scenario())

    history = broker._history_payloads["s"]
    assert [payload["textOffset"] for payload in history] == [26, 26]
    assert broker._call_offsets.get("call_1") is None


def test_broker_without_bridge_omits_text_offset() -> None:
    """Brokers without a bridge keep emitting legacy payloads unchanged."""

    broker = ToolUIBroker()
    binding = ToolUIBinding(tool_id="builtin:timer", update_every_s=1.0)

    async def scenario() -> None:
        broker._current_session_id = "s"
        await broker.publish_emit(
            binding=binding,
            tool_name="timer",
            call_id="call_2",
            message="Timer started.",
            status="running",
            running=True,
            session_id="s",
        )

    asyncio.run(scenario())
    payload: dict[str, Any] = broker._history_payloads["s"][0]
    assert "textOffset" not in payload
