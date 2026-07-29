# -*- coding: utf-8 -*-
"""
Session concurrency limiter.

Restricts the number of active WebSocket sessions. Excess clients are queued and
woken when a slot becomes available.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)



@dataclass
class QueueWaiter:
    """Queue entry for a waiting WebSocket connection."""

    websocket: WebSocket
    granted: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    holds_slot: bool = False  # True once a slot has been granted


class SessionLimiter:
    """
    Manage concurrent WebSocket session limits.

    - max_sessions: maximum active sessions (<=0 disables limiting)
    - acquire(): obtain a slot or enqueue
    - release(): release slot and wake next waiter
    """

    def __init__(self, max_sessions: int = 0):
        """
        Args:
            max_sessions: maximum concurrent sessions (<=0 disables limiting)
        """
        self.max_sessions = max_sessions
        self.active_sessions = 0
        self.wait_queue: Deque[QueueWaiter] = deque()
        self._lock = asyncio.Lock()

    @property
    def is_limited(self) -> bool:
        """Return True if limiting is enabled."""
        return self.max_sessions > 0

    @property
    def queue_length(self) -> int:
        """Return number of queued waiters."""
        return len(self.wait_queue)

    async def acquire(self, websocket: WebSocket) -> Optional[QueueWaiter]:
        """
        Acquire a session slot, queueing if necessary.

        Returns:
            QueueWaiter for later release, or None if the websocket disconnected.
        """
        waiter = QueueWaiter(websocket=websocket)

        # Skip limiting when disabled
        if not self.is_limited:
            waiter.granted.set()
            return waiter

        async with self._lock:
            if self.active_sessions < self.max_sessions:
                # Grant slot immediately
                self.active_sessions += 1
                waiter.granted.set()
                waiter.holds_slot = True
                return waiter
            else:
                self.wait_queue.append(waiter)
                position = len(self.wait_queue)

        # Notify client of queue position
        try:
            await websocket.send_json(
                {
                    "action": "queue_status",
                    "data": {
                        "position": position,
                        "limit": self.max_sessions,
                        "active": self.active_sessions,
                    },
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send queue_status: {e}")
            # Remove from queue when notification fails
            await self._remove_waiter(waiter)
            return None

        # Periodically report queue position
        update_task = asyncio.create_task(
            self._periodic_queue_update(waiter), name="queue_update"
        )

        try:
            await waiter.granted.wait()
        except asyncio.CancelledError:
            waiter.cancelled = True
            await self._remove_waiter(waiter)
            update_task.cancel()
            return None
        finally:
            update_task.cancel()

        if waiter.cancelled:
            return None

        # Defensive: ensure slot tracking matches
        if not waiter.holds_slot:
            logger.error("BUG: waiter was granted but holds_slot is False")
            waiter.holds_slot = True

        # Notify client it has been granted
        try:
            await websocket.send_json({"action": "queue_granted"})
        except Exception as e:
            logger.warning(f"Failed to send queue_granted: {e}")
            # Client disconnected, release slot
            await self.release(waiter)
            return None

        return waiter

    async def release(self, waiter: Optional[QueueWaiter]) -> None:
        """Release a slot and wake the next queued waiter."""
        if waiter is None:
            return

        if not self.is_limited:
            return

        async with self._lock:
            if waiter in self.wait_queue:
                self.wait_queue.remove(waiter)
                return

            if not waiter.holds_slot:
                return

            waiter.holds_slot = False

            if self.wait_queue:
                next_waiter = self.wait_queue.popleft()
                next_waiter.holds_slot = True
                next_waiter.granted.set()
            else:
                self.active_sessions -= 1

    async def _remove_waiter(self, waiter: QueueWaiter) -> None:
        """Remove a waiter from the queue."""
        async with self._lock:
            if waiter in self.wait_queue:
                self.wait_queue.remove(waiter)

    async def _periodic_queue_update(self, waiter: QueueWaiter) -> None:
        """Send periodic queue position updates to waiting clients."""
        while not waiter.granted.is_set() and not waiter.cancelled:
            await asyncio.sleep(5)
            if waiter.granted.is_set() or waiter.cancelled:
                break

            # Compute queue position
            async with self._lock:
                if waiter not in self.wait_queue:
                    break
                position = list(self.wait_queue).index(waiter) + 1

            try:
                await waiter.websocket.send_json(
                    {
                        "action": "queue_status",
                        "position": position,
                        "limit": self.max_sessions,
                        "active": self.active_sessions,
                    }
                )
            except Exception:
                # Connection lost; cancel and wake acquire() to avoid leaks
                waiter.cancelled = True
                waiter.granted.set()
                async with self._lock:
                    if waiter in self.wait_queue:
                        self.wait_queue.remove(waiter)
                break

    def get_status(self) -> dict:
        """Return limiter status for monitoring APIs."""
        return {
            "max_sessions": self.max_sessions,
            "active_sessions": self.active_sessions,
            "queue_length": self.queue_length,
            "is_limited": self.is_limited,
        }
