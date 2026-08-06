"""Built-in asynchronous timer tool."""

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
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)


class TimerInput(ToolInput):
    """Input accepted by the asynchronous timer."""

    duration_seconds: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Total timer duration in seconds. Must be greater than zero.",
    )
    reminder_interval_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Optional reminder interval in seconds. When provided, immediately "
            "call subscribe_async_tool after starting the timer to subscribe to "
            "progress updates."
        ),
    )


@dataclass
class TimerState(ToolState):
    """Mutable state for one asynchronous timer invocation."""

    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    stopped: bool = False


class TimerOutput(ToolOutput):
    """Final result returned when an asynchronous timer expires."""

    content: str
    elapsed_seconds: float

    def to_content(self) -> str:
        """Return the human-readable timer completion message."""

        return self.content


class TimerTool(AsyncTool):
    """Start a background timer with optional periodic progress reminders.

    No extra reply is needed before calling this tool, such as "Let me set a
    timer for you".
    """

    name = "timer"
    subscribe_by_default = False

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format a duration compactly for a human-readable message."""

        return f"{seconds:.1f}".rstrip("0").rstrip(".")

    @classmethod
    def _elapsed_seconds(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
    ) -> float:
        """Return the timer's current elapsed time, clamped to its duration."""

        if tool_state.started_at <= 0:
            return tool_state.elapsed_seconds
        current = max(
            tool_state.elapsed_seconds,
            time.monotonic() - tool_state.started_at,
        )
        return min(tool_input.duration_seconds, current)

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Running:
        """Start the timer and immediately return its protocol result."""

        del global_state
        tool_state.started_at = time.monotonic()
        duration = cls._format_seconds(tool_input.duration_seconds)
        message = (
            f"Timer started and will finish in {duration} seconds. "
            f"Tool call ID: {tool_call_id}."
        )
        if tool_input.reminder_interval_seconds is not None:
            interval = cls._format_seconds(tool_input.reminder_interval_seconds)
            message += (
                f" The user requested a reminder every {interval} seconds. "
                "Immediately call subscribe_async_tool with source_call_id set to "
                f"{tool_call_id}."
            )
        return Running(message)

    @classmethod
    def emit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[TimerOutput]]:
        """Yield no synchronous updates because the timer is natively async."""

        del tool_input, tool_state, global_state
        return iter(())

    @classmethod
    async def aemit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> AsyncIterator[ToolResult[TimerOutput]]:
        """Emit optional interval reminders and finish at the requested time."""

        del global_state
        duration = tool_input.duration_seconds
        interval = tool_input.reminder_interval_seconds
        next_reminder = interval

        while next_reminder is not None and next_reminder < duration:
            wake_at = tool_state.started_at + next_reminder
            await asyncio.sleep(max(0.0, wake_at - time.monotonic()))
            if tool_state.stopped:
                return
            tool_state.elapsed_seconds = cls._elapsed_seconds(
                tool_input,
                tool_state,
            )
            elapsed = cls._format_seconds(tool_state.elapsed_seconds)
            total = cls._format_seconds(duration)
            yield Running(
                f"Timer progress: {elapsed} seconds elapsed out of {total} seconds."
            )
            next_reminder += interval

        finish_at = tool_state.started_at + duration
        await asyncio.sleep(max(0.0, finish_at - time.monotonic()))
        if tool_state.stopped:
            return
        tool_state.elapsed_seconds = duration
        total = cls._format_seconds(duration)
        yield Finished(
            TimerOutput(
                content=(
                    f"Timer finished after the configured duration of {total} seconds."
                ),
                elapsed_seconds=duration,
            )
        )

    @classmethod
    def status(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> str:
        """Return how many seconds have elapsed for the timer."""

        del global_state
        elapsed_seconds = cls._elapsed_seconds(tool_input, tool_state)
        elapsed = cls._format_seconds(elapsed_seconds)
        total = cls._format_seconds(tool_input.duration_seconds)
        if tool_state.stopped:
            return (
                f"Timer stopped after {elapsed} seconds out of {total} seconds."
            )
        return f"Timer progress: {elapsed} seconds elapsed out of {total} seconds."

    @classmethod
    def stop(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        """Record elapsed time before XTalk cancels the timer task."""

        del global_state
        tool_state.elapsed_seconds = cls._elapsed_seconds(
            tool_input,
            tool_state,
        )
        tool_state.stopped = True


def create_tools() -> list[type[AsyncTool]]:
    """Create the tools exported by this directory.

    Returns
    -------
    list[type[AsyncTool]]
        Native XTalk tool classes registered with the configured Agent.
    """

    return [TimerTool]
