"""Native tools used by the tool scenarios."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from pydantic import Field

from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    SyncTool,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)

LOOKUP_ANSWER = "今天晴"


def format_seconds(seconds: float) -> str:
    """Format a duration compactly."""
    return f"{seconds:.1f}".rstrip("0").rstrip(".")


class LookupInput(ToolInput):
    """Query accepted by the lookup tool."""

    query: str


class LookupOutput(ToolOutput):
    """Fixed lookup answer."""

    content: str

    def to_content(self) -> str:
        return self.content


class MockLookupTool(SyncTool):
    """Return a fixed answer for any query."""

    name = "mock_lookup"

    @classmethod
    def invoke(cls, tool_input: LookupInput, global_state: ToolEngineState) -> LookupOutput:
        del tool_input, global_state
        return LookupOutput(content=LOOKUP_ANSWER)


class TimerInput(ToolInput):
    """Timer duration with an optional reminder interval."""

    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    reminder_interval_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


@dataclass
class TimerState(ToolState):
    """Mutable state of one timer call."""

    started_at: float = 0.0
    stopped: bool = False


class TimerOutput(ToolOutput):
    """Final timer result."""

    content: str

    def to_content(self) -> str:
        return self.content


class MockTimerTool(AsyncTool):
    """Start a background timer that reports progress and completion."""

    name = "mock_timer"
    subscribe_by_default = False

    @classmethod
    def _elapsed(cls, tool_input: TimerInput, tool_state: TimerState) -> float:
        if tool_state.started_at <= 0:
            return 0.0
        return min(tool_input.duration_seconds, time.monotonic() - tool_state.started_at)

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Running:
        del global_state
        tool_state.started_at = time.monotonic()
        duration = format_seconds(tool_input.duration_seconds)
        return Running(f"计时开始，共{duration}秒。Tool call ID: {tool_call_id}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[TimerOutput]]:
        del tool_input, tool_state, global_state
        return iter(())

    @classmethod
    async def aemit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> AsyncIterator[ToolResult[TimerOutput]]:
        del global_state
        duration = tool_input.duration_seconds
        interval = tool_input.reminder_interval_seconds
        next_reminder = interval
        while next_reminder is not None and next_reminder < duration:
            await asyncio.sleep(max(0.0, tool_state.started_at + next_reminder - time.monotonic()))
            if tool_state.stopped:
                return
            yield Running(f"已过{format_seconds(cls._elapsed(tool_input, tool_state))}秒")
            next_reminder += interval
        await asyncio.sleep(max(0.0, tool_state.started_at + duration - time.monotonic()))
        if tool_state.stopped:
            return
        yield Finished(TimerOutput(content=f"计时完成，共{format_seconds(duration)}秒"))

    @classmethod
    def status(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> str:
        del global_state
        return f"已过{format_seconds(cls._elapsed(tool_input, tool_state))}秒"

    @classmethod
    def stop(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.stopped = True
