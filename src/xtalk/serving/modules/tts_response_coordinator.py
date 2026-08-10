"""Serialize client delivery while allowing the next response to prepare."""

from __future__ import annotations

import asyncio
from typing import Any

from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    TTSResponseClosed,
    TTSStopped,
    TurnTTSDeliveryStartRequested,
    TurnTTSStartRequested,
    TurnTTSStopRequested,
)
from ..interfaces import Manager

class TTSResponseCoordinator(Manager):
    """Gate all response delivery through one session-scoped state machine."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize response-delivery state for one session.

        Parameters
        ----------
        event_bus : EventBus
            Session event bus.
        session_id : str
            Session identifier.
        config : dict[str, Any] | None, optional
            Shared service configuration.
        """

        self.event_bus = event_bus
        self.session_id = session_id
        self.config = config or {}
        self._delivering_response_id: str | None = None
        self._stopping_response_id: str | None = None
        self._preparing_response_id: str | None = None
        self._preparing_ready = False
        self._switch_lock = asyncio.Lock()

    @Manager.event_handler(TurnTTSStartRequested, priority=90)
    async def _handle_response_start(self, event: TurnTTSStartRequested) -> None:
        """Prepare, preempt, or immediately release one response."""

        response_id = event.response_id
        if not response_id:
            return

        delivery_response_id: str | None = None
        restart_response_id: str | None = None
        stop_response_id: str | None = None
        superseded_response_id: str | None = None

        async with self._switch_lock:
            if response_id == self._delivering_response_id:
                return
            if response_id == self._preparing_response_id:
                self._preparing_ready = True
                if self._delivering_response_id is None:
                    delivery_response_id = self._take_prepared_response_locked()
            elif self._delivering_response_id is None:
                self._delivering_response_id = response_id
                delivery_response_id = response_id
            else:
                if self._preparing_response_id is not None:
                    superseded_response_id = self._preparing_response_id
                self._preparing_response_id = response_id
                self._preparing_ready = False
                restart_response_id = response_id
                if self._stopping_response_id is None:
                    self._stopping_response_id = self._delivering_response_id
                    stop_response_id = self._delivering_response_id

        if superseded_response_id is not None:
            await self.event_bus.publish(
                TurnTTSStopRequested(
                    session_id=self.session_id,
                    response_id=superseded_response_id,
                    reason="superseded",
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

        if stop_response_id is not None:
            await self.event_bus.publish(
                TurnTTSStopRequested(
                    session_id=self.session_id,
                    response_id=stop_response_id,
                    reason="response_preempted",
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

        if restart_response_id is not None:
            await self.event_bus.publish(
                TurnTTSStartRequested(
                    session_id=self.session_id,
                    response_id=restart_response_id,
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

        if delivery_response_id is not None:
            await self._start_delivery(delivery_response_id)

    @Manager.event_handler(TTSStopped, priority=15)
    async def _handle_tts_stopped(self, event: TTSStopped) -> None:
        """Remember which delivered response is awaiting playback settlement."""

        async with self._switch_lock:
            if event.response_id == self._delivering_response_id:
                self._stopping_response_id = event.response_id

    @Manager.event_handler(TurnTTSStopRequested, priority=90)
    async def _handle_stop_request(self, event: TurnTTSStopRequested) -> None:
        """Discard preparing responses targeted by a session or response stop."""

        async with self._switch_lock:
            if self._preparing_response_id is None:
                return
            if event.response_id not in (None, self._preparing_response_id):
                return
            self._preparing_response_id = None
            self._preparing_ready = False

    @Manager.event_handler(TTSResponseClosed, priority=90)
    async def _handle_response_closed(self, event: TTSResponseClosed) -> None:
        """Release the latest prepared response after the old response closes."""

        delivery_response_id: str | None = None
        async with self._switch_lock:
            if event.response_id == self._delivering_response_id:
                self._delivering_response_id = None
            if event.response_id == self._stopping_response_id:
                self._stopping_response_id = None
            if self._delivering_response_id is None and self._preparing_ready:
                delivery_response_id = self._take_prepared_response_locked()

        if delivery_response_id is not None:
            await self._start_delivery(delivery_response_id)

    def _take_prepared_response_locked(self) -> str | None:
        """Promote the ready preparing response while holding the switch lock."""

        if not self._preparing_ready:
            return None
        response_id = self._preparing_response_id
        if response_id is None:
            return None
        self._preparing_response_id = None
        self._preparing_ready = False
        self._delivering_response_id = response_id
        return response_id

    async def _start_delivery(self, response_id: str) -> None:
        """Open the client-delivery gate for one prepared response."""

        await self.event_bus.publish(
            TurnTTSDeliveryStartRequested(
                session_id=self.session_id,
                response_id=response_id,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    async def shutdown(self) -> None:
        """Clear coordinator state during session shutdown."""

        async with self._switch_lock:
            self._delivering_response_id = None
            self._stopping_response_id = None
            self._preparing_response_id = None
            self._preparing_ready = False
