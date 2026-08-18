"""Contract tests for the desktop asynchronous timer tool."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from xtalk.models.agents.tools import Finished, Running, ToolEngine

from backend import timer_tool
from backend.timer_tool import TimerInput, TimerOutput, TimerState, TimerTool


@dataclass
class _Clock:
    """Deterministic monotonic clock used by timer contract tests."""

    now: float

    def monotonic(self) -> float:
        """Return the current deterministic timestamp."""

        return self.now

    async def sleep(self, delay: float) -> None:
        """Advance time by the requested non-negative delay."""

        self.now += delay


async def _collect_updates(
    tool_input: TimerInput,
    tool_state: TimerState,
) -> list[Any]:
    """Collect every asynchronous timer update into a list."""

    return [
        update
        async for update in TimerTool.aemit_updates(
            tool_input,
            tool_state,
            object(),
        )
    ]


async def _run_timer_through_tool_engine() -> tuple[Any, Any, Any]:
    """Run the real timer lifecycle through XTalk's public tool engine."""

    engine = ToolEngine(tools=[TimerTool], state={})
    updates: list[tuple[Any, Any]] = []
    finished = asyncio.Event()

    def record_update(tool_call: Any, tool_message: Any) -> None:
        """Record the engine's proactive final update."""

        updates.append((tool_call, tool_message))
        finished.set()

    engine.on_async_tool_update(record_update)
    try:
        initial = await engine.ainvoke(
            {
                "id": "timer-engine-call",
                "name": "timer",
                "args": {
                    "duration_seconds": 0.1,
                    "reminder_interval_seconds": None,
                },
            }
        )
        await asyncio.wait_for(finished.wait(), timeout=2.0)
        update_call, update_message = updates[-1]
        return initial, update_call, update_message
    finally:
        await engine.shutdown()


def test_timer_transitions_from_running_updates_to_finished(
    monkeypatch,
) -> None:
    """Emit an initial Running result, progress, and one final Finished result."""

    clock = _Clock(now=100.0)
    monkeypatch.setattr(timer_tool.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(timer_tool.asyncio, "sleep", clock.sleep)
    tool_input = TimerInput(
        duration_seconds=5.0,
        reminder_interval_seconds=2.0,
    )
    tool_state = TimerState()

    initial = TimerTool.emit_initial(
        "timer-call-1",
        tool_input,
        tool_state,
        object(),
    )
    updates = asyncio.run(_collect_updates(tool_input, tool_state))

    assert isinstance(initial, Running)
    assert "timer-call-1" in initial.content
    assert "subscribe_async_tool" in initial.content
    assert [type(update) for update in updates] == [
        Running,
        Running,
        Finished,
    ]
    assert "2 seconds elapsed" in updates[0].content
    assert "4 seconds elapsed" in updates[1].content
    finished = updates[2]
    assert isinstance(finished, Finished)
    assert isinstance(finished.content, TimerOutput)
    assert finished.content.elapsed_seconds == pytest.approx(5.0)
    assert finished.content.to_content() == (
        "Timer finished after the configured duration of 5 seconds."
    )
    assert tool_state.elapsed_seconds == pytest.approx(5.0)
    assert tool_state.stopped is False


def test_timer_reaches_finished_state_through_public_tool_engine() -> None:
    """Complete a real asynchronous timer through XTalk's public engine."""

    initial, update_call, update_message = asyncio.run(
        _run_timer_through_tool_engine()
    )
    payload = json.loads(str(update_message.content))

    assert initial.name == "timer"
    assert "timer-engine-call" in str(initial.content)
    assert update_call["name"] == "async_tool_updated"
    assert update_call["args"] == {"source_call_id": "timer-engine-call"}
    assert payload == {
        "running": False,
        "tool_output": (
            "Timer finished after the configured duration of 0.1 seconds."
        ),
    }


def test_timer_stop_records_elapsed_time_and_suppresses_completion(
    monkeypatch,
) -> None:
    """Stop a running timer without yielding a Finished result."""

    clock = _Clock(now=200.0)
    monkeypatch.setattr(timer_tool.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(timer_tool.asyncio, "sleep", clock.sleep)
    tool_input = TimerInput(
        duration_seconds=10.0,
        reminder_interval_seconds=2.0,
    )
    tool_state = TimerState()
    initial = TimerTool.emit_initial(
        "timer-call-2",
        tool_input,
        tool_state,
        object(),
    )
    clock.now = 203.25

    result = TimerTool.stop(tool_input, tool_state, object())
    status = TimerTool.status(tool_input, tool_state, object())
    updates = asyncio.run(_collect_updates(tool_input, tool_state))

    assert isinstance(initial, Running)
    assert result is None
    assert tool_state.stopped is True
    assert tool_state.elapsed_seconds == pytest.approx(3.25)
    assert status == "Timer stopped after 3.2 seconds out of 10 seconds."
    assert updates == []
