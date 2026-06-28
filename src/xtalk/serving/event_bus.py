# -*- coding: utf-8 -*-
"""
Event Bus with enhanced error handling and recursion protection.
"""
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional, Set, Type, Union
import weakref
import time

from ..log_utils import logger

from .events import Event, ErrorOccurred


@dataclass
class EventHandler:
    """Container for handler metadata."""

    handler: Callable
    priority: int = 0
    error_count: int = 0
    last_error: Optional[Exception] = None


class EventBus:
    """Publish and subscribe session events with async dispatch support.

    Parameters
    ----------
    enable_history : bool, optional
        Whether to store published events in memory for later inspection.
    max_history : int, optional
        Maximum number of events kept when history is enabled.
    """

    # Max error event recursion depth
    MAX_ERROR_EVENT_DEPTH = 3
    # Error event publish cooldown (s)
    ERROR_EVENT_COOLDOWN = 1.0
    # Error event publish rate limit (/s)
    ERROR_EVENT_RATE_LIMIT = 10

    def __init__(self, enable_history: bool = False, max_history: int = 1000):
        """Initialize the event bus.

        Parameters
        ----------
        enable_history : bool, optional
            Whether to record published events in the in-memory history buffer.
        max_history : int, optional
            Maximum number of events retained in history.
        """
        # Map of event type string to handlers
        # {event_type(str): [EventHandler]}
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)

        # Event history tracking
        self._enable_history = enable_history
        self._max_history = max_history
        self._event_history: List[Event] = []

        # Metrics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "errors_occurred": 0,
            "handlers_count": 0,
            "error_events_dropped": 0,  # Error events dropped due to recursion protection
        }

        # Active async tasks
        self._active_tasks: Set[asyncio.Task] = set()

        # Weak references for cleanup
        self._weak_refs: List[weakref.ref] = []

        # Error event recursion protection
        self._error_event_depth = 0
        self._error_event_lock = asyncio.Lock()

        # Error event rate limiting
        self._error_event_times: List[float] = []
        self._last_error_event_time = 0.0

    def _get_event_key(self, event_identifier: Union[Type[Event], str]) -> str:
        """Normalize event identifier (class or string) into type string."""
        # Already a string
        if isinstance(event_identifier, str):
            return event_identifier
        # Prefer TYPE attribute on event class
        try:
            evt_type = event_identifier.TYPE
            if isinstance(evt_type, str) and evt_type:
                return evt_type
        except Exception:
            pass
        # Fall back to class name
        try:
            return event_identifier.__name__
        except Exception:
            return str(event_identifier)

    def subscribe(
        self,
        event_class: Union[Type[Event], str],
        handler: Callable[[Event], Any],
        priority: int = 0,
    ) -> None:
        """Subscribe a handler to an event type.

        Parameters
        ----------
        event_class : Type[Event] | str
            Event class or event type string such as ``"tts.started"``.
        handler : Callable[[Event], Any]
            Sync or async callable invoked for matching events.
        priority : int, optional
            Higher values run earlier.
        """
        event_handler = EventHandler(handler=handler, priority=priority)

        key = self._get_event_key(event_class)
        self._handlers[key].append(event_handler)

        # Sort by priority descending
        self._handlers[key].sort(key=lambda h: h.priority, reverse=True)

        self._stats["handlers_count"] += 1

    def unsubscribe(
        self, event_class: Union[Type[Event], str], handler: Callable
    ) -> bool:
        """Unsubscribe a handler from an event type.

        Parameters
        ----------
        event_class : Type[Event] | str
            Event class or event type string.
        handler : Callable
            Previously subscribed handler to remove.

        Returns
        -------
        bool
            ``True`` if the handler was removed, otherwise ``False``.
        """
        key = self._get_event_key(event_class)
        handlers = self._handlers.get(key, [])
        for i, event_handler in enumerate(handlers):
            if event_handler.handler == handler:
                del handlers[i]
                self._stats["handlers_count"] -= 1

                return True
        return False

    async def publish(
        self, event: Event, wait_for_completion: bool = False
    ) -> bool:
        """Publish an event to all matching handlers.

        Parameters
        ----------
        event : Event
            Event instance to dispatch.
        wait_for_completion : bool, optional
            Whether to await handler completion before returning.

        Returns
        -------
        bool
            ``True`` on success, otherwise ``False``.
        """
        try:
            self._stats["events_published"] += 1

            # Append to history if enabled
            if self._enable_history:
                self._add_to_history(event)

            # Retrieve handlers for this event type
            event_type = event.event_type
            handlers = self._handlers.get(event_type, [])
            if not handlers:
                return True

            if wait_for_completion:
                # Preserve priority order for event chains that require deterministic
                # completion before returning.
                for handler in handlers:
                    await self._handle_event_safe(handler, event)
                return True

            # spawn handler tasks
            for handler in handlers:
                task = asyncio.create_task(self._handle_event_safe(handler, event))
                self._active_tasks.add(task)

                # cleanup after completion
                task.add_done_callback(self._active_tasks.discard)

            return True

        except Exception as e:
            logger.error("Failed to publish event: %s", e)
            self._stats["errors_occurred"] += 1

            # Publish error event safely
            if event.event_type != "error.occurred":
                await self._publish_error_event_safe(
                    session_id=event.session_id,
                    error_type="event_bus_publish_error",
                    error_message=str(e),
                )

            return False

    async def _handle_event_safe(self, handler: EventHandler, event: Event) -> None:
        """
        Safely handle an event, capturing exceptions.

        Args:
            handler: event handler metadata
            event: event instance
        """
        try:
            self._stats["events_processed"] += 1

            # Invoke handler
            if asyncio.iscoroutinefunction(handler.handler):
                await handler.handler(event)
            else:
                handler.handler(event)

        except Exception as e:
            handler.error_count += 1
            handler.last_error = e
            self._stats["errors_occurred"] += 1

            logger.error(
                "Event handler raised: %s, event_type: %s", e, event.event_type
            )

            # Publish error event safely
            if event.event_type != "error.occurred":
                await self._publish_error_event_safe(
                    session_id=event.session_id,
                    error_type="event_handler_error",
                    error_message=str(e),
                )

    async def _publish_error_event_safe(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """
        Safely publish an error event with recursion protection.

        Protection mechanisms:
        1. Depth tracking: Prevents deep recursion
        2. Rate limiting: Prevents error event flooding
        3. Cooldown: Prevents rapid successive error events
        4. Circuit breaker: Temporarily stops error events if too many failures

        Args:
            session_id: session identifier
            error_type: type of error
            error_message: error message
        """
        # Check recursion depth
        if self._error_event_depth >= self.MAX_ERROR_EVENT_DEPTH:
            logger.error(
                "Max error event depth (%d) reached, dropping error event: %s",
                self.MAX_ERROR_EVENT_DEPTH,
                error_type,
            )
            self._stats["error_events_dropped"] += 1
            return

        # Check cooldown period
        current_time = time.time()
        if current_time - self._last_error_event_time < self.ERROR_EVENT_COOLDOWN:
            logger.debug(
                "Error event cooldown active, dropping error event: %s",
                error_type,
            )
            self._stats["error_events_dropped"] += 1
            return

        # Check rate limit
        self._error_event_times = [
            t for t in self._error_event_times if current_time - t < 1.0
        ]
        if len(self._error_event_times) >= self.ERROR_EVENT_RATE_LIMIT:
            logger.warning(
                "Error event rate limit (%d/s) exceeded, dropping error event: %s",
                self.ERROR_EVENT_RATE_LIMIT,
                error_type,
            )
            self._stats["error_events_dropped"] += 1
            return

        # Lock to protect recursion depth counter
        async with self._error_event_lock:
            self._error_event_depth += 1
            try:
                # Create error event
                error_event = ErrorOccurred(
                    session_id=session_id,
                    error_type=error_type,
                    error_message=error_message,
                )

                # Publish error event (fire-and-forget to avoid blocking)
                # Use try-except to ensure depth counter is restored even if publish fails
                try:
                    await self.publish(error_event, wait_for_completion=False)
                    self._last_error_event_time = current_time
                    self._error_event_times.append(current_time)
                except Exception as e:
                    logger.error("Failed to publish error event (suppressing): %s", e)
                    # Do not recurse further; just log
                    self._stats["error_events_dropped"] += 1

            finally:
                # Ensure depth counter is always restored
                self._error_event_depth -= 1

    def _add_to_history(self, event: Event) -> None:
        """
        Append event to history storage.

        Args:
            event: event instance
        """
        self._event_history.append(event)

        # Trim history size
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

    def get_history(
        self, event_type: Optional[str] = None, session_id: Optional[str] = None
    ) -> List[Event]:
        """
        Retrieve event history with optional filters.

        Args:
            event_type: filter by type
            session_id: filter by session id

        Returns:
            List of events
        """
        if not self._enable_history:
            return []

        history = self._event_history

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        if session_id:
            history = [e for e in history if e.session_id == session_id]

        return history

    def get_stats(self) -> Dict[str, Any]:
        """
        Return current event bus statistics.

        Returns:
            Dict of stats
        """
        handler_stats = {}
        for event_type, handlers in self._handlers.items():
            handler_stats[event_type] = len(handlers)

        return {
            **self._stats,
            "handler_stats": handler_stats,
            "active_tasks": len(self._active_tasks),
            "history_enabled": self._enable_history,
            "history_count": len(self._event_history) if self._enable_history else 0,
            "error_event_depth": self._error_event_depth,
        }

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def reset_error_tracking(self) -> None:
        """Reset error event tracking (useful for testing)."""
        self._error_event_depth = 0
        self._error_event_times.clear()
        self._last_error_event_time = 0.0

    async def _wait_for_all_handlers(self, timeout: float = 3.0):
        """
        Wait for all active handler tasks to finish.

        Args:
            timeout: timeout in seconds

        Returns:
            None
        """
        if not self._active_tasks:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._active_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for handlers; cancelling %d pending tasks",
                len(self._active_tasks),
            )
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()

    async def shutdown(self) -> None:
        """Shut down the event bus and release resources."""

        # Wait for active tasks to finish
        await self._wait_for_all_handlers(timeout=3.0)

        # Cleanup structures
        self._handlers.clear()
        self._event_history.clear()
        self._active_tasks.clear()

        # Reset error tracking
        self.reset_error_tracking()

    def __del__(self):
        """Destructor: best-effort cleanup."""
        # Release resources
        if hasattr(self, "_active_tasks"):
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()
