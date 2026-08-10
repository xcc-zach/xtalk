"""Tests for event dispatch waiting and propagation control."""

from __future__ import annotations

import asyncio
import unittest

from xtalk.serving.event_bus import (
    EventBus,
    EventDispatchMode,
    EventPropagation,
)
from xtalk.serving.events import Event, create_event_class


ProbeEvent = create_event_class(name="ProbeEvent", type_name="test.probe")


class EventDispatchModeTests(unittest.TestCase):
    """Verify canonical dispatch modes and supported string aliases."""

    def test_parses_long_and_short_mode_strings(self) -> None:
        """Normalize every public string spelling to its enum member."""

        cases = {
            "return_after_dispatch": EventDispatchMode.RETURN_AFTER_DISPATCH,
            "dispatch": EventDispatchMode.RETURN_AFTER_DISPATCH,
            "wait_until_complete": EventDispatchMode.WAIT_UNTIL_COMPLETE,
            "wait": EventDispatchMode.WAIT_UNTIL_COMPLETE,
            "wait_until_complete_or_stopped": (
                EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED
            ),
            "wait_stoppable": EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
            " WAIT-STOPPABLE ": (
                EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED
            ),
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertIs(EventDispatchMode.parse(value), expected)
        self.assertIs(
            EventDispatchMode.parse(EventDispatchMode.WAIT_UNTIL_COMPLETE),
            EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    def test_rejects_unknown_or_non_string_modes(self) -> None:
        """Fail explicitly instead of silently changing dispatch behavior."""

        for value in ("", "later", 1, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EventDispatchMode.parse(value)  # type: ignore[arg-type]


class EventBusPropagationTests(unittest.IsolatedAsyncioTestCase):
    """Verify waiting, background dispatch, and explicit propagation stops."""

    async def test_return_after_dispatch_does_not_wait_for_handler(self) -> None:
        """Return after scheduling a background handler that is still blocked."""

        event_bus = EventBus()
        self.addAsyncCleanup(event_bus.shutdown)
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(_event: Event) -> None:
            started.set()
            await release.wait()

        event_bus.subscribe(ProbeEvent, handler)
        await event_bus.publish(
            ProbeEvent(session_id="session"),
            mode="dispatch",
        )
        await started.wait()
        self.assertFalse(release.is_set())
        release.set()

    async def test_wait_until_complete_preserves_priority_order(self) -> None:
        """Await every handler and ignore STOP outside stoppable mode."""

        event_bus = EventBus()
        self.addAsyncCleanup(event_bus.shutdown)
        calls: list[str] = []

        async def low(_event: Event) -> None:
            calls.append("low")

        async def high(_event: Event) -> EventPropagation:
            calls.append("high")
            return EventPropagation.STOP

        event_bus.subscribe(ProbeEvent, low, priority=0)
        event_bus.subscribe(ProbeEvent, high, priority=10)
        await event_bus.publish(
            ProbeEvent(session_id="session"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

        self.assertEqual(calls, ["high", "low"])

    async def test_publish_rejects_unknown_mode(self) -> None:
        """Surface invalid mode input to the caller as a ValueError."""

        event_bus = EventBus()
        self.addAsyncCleanup(event_bus.shutdown)

        with self.assertRaises(ValueError):
            await event_bus.publish(
                ProbeEvent(session_id="session"),
                mode="eventually",
            )

    async def test_stoppable_dispatch_skips_lower_handlers_and_history(self) -> None:
        """Stop at the first explicit STOP and omit the intercepted event."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        calls: list[str] = []

        async def low(_event: Event) -> None:
            calls.append("low")

        async def gate(_event: Event) -> EventPropagation:
            calls.append("gate")
            return EventPropagation.STOP

        event_bus.subscribe(ProbeEvent, low, priority=0)
        event_bus.subscribe(ProbeEvent, gate, priority=1000)
        await event_bus.publish(
            ProbeEvent(session_id="session"),
            mode="wait_stoppable",
        )

        self.assertEqual(calls, ["gate"])
        self.assertEqual(event_bus.get_history(), [])
        self.assertEqual(event_bus.get_stats()["events_stopped"], 1)

    async def test_stoppable_dispatch_records_event_when_not_stopped(self) -> None:
        """Record a stoppable event after every handler continues."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)

        async def handler(_event: Event) -> None:
            return None

        event_bus.subscribe(ProbeEvent, handler)
        await event_bus.publish(
            ProbeEvent(session_id="session"),
            mode="wait_until_complete_or_stopped",
        )

        self.assertEqual(len(event_bus.get_history()), 1)

    async def test_stoppable_dispatch_continues_after_handler_error(self) -> None:
        """Treat handler errors as CONTINUE without an on-error stop policy."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        calls: list[str] = []

        async def later(_event: Event) -> None:
            calls.append("later")

        async def failing(_event: Event) -> None:
            calls.append("failing")
            raise RuntimeError("gate failed")

        event_bus.subscribe(ProbeEvent, later, priority=0)
        event_bus.subscribe(ProbeEvent, failing, priority=1000)
        await event_bus.publish(
            ProbeEvent(session_id="session"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )

        self.assertEqual(calls, ["failing", "later"])
        self.assertEqual(len(event_bus.get_history(event_type="test.probe")), 1)
        self.assertEqual(event_bus.get_stats()["events_stopped"], 0)
