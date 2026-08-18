# -*- coding: utf-8 -*-
"""Shared per-session tool-call text offsets for the desktop tool UI."""

from __future__ import annotations

import threading


class DesktopToolCallBridge:
    """Match tool invocations to their character offsets in assistant text.

    The LLM stream interleaves text chunks and tool-call markers. The desktop
    output gateway sees both and can therefore record, for every tool call,
    how many characters of the current assistant response had already been
    generated. The Tool UI broker consumes those offsets in call order and
    attaches them to its emit payloads so the desktop chat can render tool UI
    rows inside a message at the exact generation point.

    Only tools that own a UI timeline row (registered by the tool wrapper) are
    tracked, so built-in engine calls such as ``subscribe_async_tool`` never
    disturb the per-session FIFO order.
    """

    def __init__(self) -> None:
        """Initialize an empty UI-tool registry and per-session offset queues."""

        self._lock = threading.Lock()
        self._ui_tool_names: set[str] = set()
        self._offsets: dict[str, list[int]] = {}

    def register_ui_tool(self, tool_name: str) -> None:
        """Record one tool that owns a UI timeline row.

        Parameters
        ----------
        tool_name : str
            Exported tool name used by the LLM tool-call marker.
        """

        if not tool_name:
            return
        with self._lock:
            self._ui_tool_names.add(tool_name)

    def record_tool_call(
        self,
        *,
        session_id: str,
        name: str,
        offset: int,
    ) -> None:
        """Queue the text offset of one UI tool call for a session.

        Parameters
        ----------
        session_id : str
            Session identifier owning the assistant message.
        name : str
            Tool-call name emitted by the agent.
        offset : int
            Character offset within the accumulated assistant text where the
            tool call was emitted.
        """

        if not name:
            return
        with self._lock:
            if name not in self._ui_tool_names:
                return
            self._offsets.setdefault(session_id, []).append(offset)

    def consume_tool_offset(self, *, session_id: str) -> int | None:
        """Return and remove the oldest queued offset for one session.

        Parameters
        ----------
        session_id : str
            Session identifier whose oldest UI tool call should be consumed.

        Returns
        -------
        int | None
            The recorded character offset, or ``None`` when the queue is empty.
        """

        with self._lock:
            offsets = self._offsets.get(session_id)
            if not offsets:
                return None
            return offsets.pop(0)

    def discard_session(self, session_id: str) -> None:
        """Drop all queued offsets for a terminated session.

        Parameters
        ----------
        session_id : str
            Session identifier whose queued offsets should be removed.
        """

        with self._lock:
            self._offsets.pop(session_id, None)
