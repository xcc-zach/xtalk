"""Unit tests for the desktop text-projection output gateway."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    ErrorOccurred,
    LLMAgentResponseFinish,
    LLMAgentResponseUpdate,
    ResponseFinish,
    ResponseUpdate,
)

from backend.desktop_gateway import DesktopTextProjectionGateway


class _RecordingWebSocket:
    """WebSocket stub recording every JSON frame the gateway sends."""

    def __init__(self) -> None:
        """Initialize an empty frame list with an open connection state."""

        self.client_state = SimpleNamespace(value=1)
        self.frames: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        """Record one outbound text frame."""

        self.frames.append(json.loads(payload))


def _build_gateway() -> tuple[DesktopTextProjectionGateway, _RecordingWebSocket]:
    """Create a projection gateway bound to a recording websocket."""

    websocket = _RecordingWebSocket()
    gateway = DesktopTextProjectionGateway(
        EventBus(),
        "session-1",
        websocket,
        config={},
    )
    return gateway, websocket


def _actions(websocket: _RecordingWebSocket) -> list[tuple[str, str]]:
    """Return the ``(action, text)`` pairs recorded so far."""

    return [
        (frame["action"], frame["data"]["text"])
        for frame in websocket.frames
        if frame["action"] in {"update_resp", "finish_resp"}
    ]


def test_normal_turn_streams_prefixes_then_complete_finish() -> None:
    """One uninterrupted turn shows growing text and a complete final bubble."""

    async def scenario() -> None:
        gateway, websocket = _build_gateway()
        full = "好的，我来帮你设置一个1分钟的计时器，每20秒提醒你一次。"

        await gateway._desktop_track_llm_update(
            LLMAgentResponseUpdate(session_id="s", text=full[:8])
        )
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text=full[:8])
        )
        await gateway._desktop_track_llm_update(
            LLMAgentResponseUpdate(session_id="s", text=full)
        )
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text=full[:16])
        )
        await gateway._desktop_track_llm_finish(
            LLMAgentResponseFinish(session_id="s", text=full)
        )
        await gateway._send_finish_resp_signal(
            ResponseFinish(session_id="s", text=full)
        )

        assert _actions(websocket) == [
            ("update_resp", full[:8]),
            ("update_resp", full[:16]),
            ("update_resp", full),
            ("finish_resp", full),
        ]

    asyncio.run(scenario())


def test_restart_closes_previous_message_with_full_text() -> None:
    """A mid-turn LLM restart finalizes the old bubble instead of fragmenting."""

    async def scenario() -> None:
        gateway, websocket = _build_gateway()
        first = "好嘞，这是第二个计时器，同样是1分钟、每20秒提醒一次。现在"
        second = "第一个计时器已经结束了，还剩20秒。"

        await gateway._desktop_track_llm_update(
            LLMAgentResponseUpdate(session_id="s", text=first)
        )
        # Playback confirmed only the first half of the response before restart.
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text=first[:16])
        )
        await gateway._desktop_track_llm_update(
            LLMAgentResponseUpdate(session_id="s", text=second)
        )
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text=second[:10])
        )
        await gateway._desktop_track_llm_finish(
            LLMAgentResponseFinish(session_id="s", text=second)
        )
        await gateway._send_finish_resp_signal(
            ResponseFinish(session_id="s", text=second)
        )

        assert _actions(websocket) == [
            ("update_resp", first[:16]),
            ("update_resp", first),
            ("finish_resp", first),
            ("update_resp", second[:10]),
            ("update_resp", second),
            ("finish_resp", second),
        ]

    asyncio.run(scenario())


def test_finish_prefers_full_llm_text_over_lagging_playback() -> None:
    """A lagging playback prefix cannot truncate the final displayed text."""

    async def scenario() -> None:
        gateway, websocket = _build_gateway()
        full = "第二个计时器的时间也到了，总共60秒已经走完。"

        await gateway._desktop_track_llm_update(
            LLMAgentResponseUpdate(session_id="s", text=full)
        )
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text=full[:10])
        )
        await gateway._desktop_track_llm_finish(
            LLMAgentResponseFinish(session_id="s", text=full)
        )
        # Playback manager committed only a short prefix at playback finish.
        await gateway._send_finish_resp_signal(
            ResponseFinish(session_id="s", text=full[:10])
        )

        assert _actions(websocket) == [
            ("update_resp", full[:10]),
            ("update_resp", full),
            ("finish_resp", full),
        ]

    asyncio.run(scenario())


def test_non_prefix_playback_jump_catches_up_to_llm_text() -> None:
    """A playback tracking reset cannot create a second fragment bubble."""

    async def scenario() -> None:
        gateway, websocket = _build_gateway()
        full = "好的，先说第一个计时器的进度：目前已经过了60秒，它结束了。"

        await gateway._desktop_track_llm_update(
            LLMAgentResponseUpdate(session_id="s", text=full)
        )
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text=full[:12])
        )
        # Playback tracking reset and produced an unrelated longer prefix.
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text="不相关的新片段，比之前更长")
        )

        assert _actions(websocket) == [
            ("update_resp", full[:12]),
            ("update_resp", full),
        ]

    asyncio.run(scenario())


def test_error_closes_open_message_with_complete_text() -> None:
    """An error mid-turn finalizes the open message with its full text."""

    async def scenario() -> None:
        gateway, websocket = _build_gateway()
        full = "正在回复的完整文本。"

        await gateway._desktop_track_llm_update(
            LLMAgentResponseUpdate(session_id="s", text=full)
        )
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="s", text=full[:5])
        )
        await gateway._desktop_handle_error(
            ErrorOccurred(session_id="s", error_type="test", error_message="boom")
        )

        assert _actions(websocket) == [
            ("update_resp", full[:5]),
            ("update_resp", full),
            ("finish_resp", full),
        ]

    asyncio.run(scenario())
