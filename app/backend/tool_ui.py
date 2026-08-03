"""Read-only UI observation for developer-installed asynchronous tools."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from xtalk.models.agents.tools import AsyncTool, Finished, Running


MAX_TOOL_UI_FRAME_BYTES = 2 * 1024 * 1024
MAX_TOOL_UI_FRAME_TICKETS = 400
MAX_REPLAYED_TOOL_UI_HISTORY_ITEMS = 200


@dataclass(frozen=True)
class ToolUIBinding:
    """Installed read-only UI metadata for one developer tool directory.

    Parameters
    ----------
    tool_id : str
        Stable App-generated installed tool identifier.
    update_every_s : float
        Live status polling interval, or ``-1`` to disable polling.
    """

    tool_id: str
    update_every_s: float


@dataclass
class _ToolUIFrameTicket:
    """Runtime-scoped reload ticket for one sandboxed Tool UI document."""

    source: str


class ToolUIBroker:
    """Broadcast observed tool status and emit events to the trusted App UI."""

    def __init__(self) -> None:
        """Initialize empty connection and tool-call state."""

        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._current_session_id: str | None = None
        self._call_sessions: dict[str, str | None] = {}
        self._status_sequences: dict[str, int] = {}
        self._emit_sequences: dict[str, int] = {}
        self._last_status: dict[str, tuple[str, bool]] = {}
        self._live_status_payloads: dict[str, dict[str, Any]] = {}
        self._history_payloads: dict[str | None, list[dict[str, Any]]] = {}
        self._frame_tickets: dict[str, _ToolUIFrameTicket] = {}

    async def serve(self, websocket: WebSocket) -> None:
        """Serve one authenticated read-only Tool UI WebSocket.

        Parameters
        ----------
        websocket : fastapi.WebSocket
            Authorized App connection used for event delivery and session
            binding. Incoming messages cannot operate tools.
        """

        await websocket.accept()
        async with self._lock:
            self._clients.append(websocket)
        try:
            while True:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") != "bind_session":
                    continue
                session_id = payload.get("sessionId")
                if session_id is not None and (
                    not isinstance(session_id, str) or not session_id
                ):
                    continue
                async with self._lock:
                    self._current_session_id = session_id
                    history_payloads = self._bind_history_payloads(session_id)
                    live_statuses = []
                    for call_id, status in self._live_status_payloads.items():
                        if (
                            status.get("sessionId") is None
                            and session_id is not None
                        ):
                            status["sessionId"] = session_id
                            self._call_sessions[call_id] = session_id
                        if status.get("sessionId") == session_id:
                            live_statuses.append(dict(status))
                for replay in [*history_payloads, *live_statuses]:
                    await websocket.send_json(replay)
        except WebSocketDisconnect:
            pass
        finally:
            async with self._lock:
                self._clients = [
                    client for client in self._clients if client is not websocket
                ]

    async def snapshot(self, session_id: str) -> list[dict[str, Any]]:
        """Return replayable history and live UI events for one App session.

        Parameters
        ----------
        session_id : str
            Persisted chat session currently displayed by the desktop App.

        Returns
        -------
        list[dict[str, Any]]
            Immutable event copies safe to serialize through the authenticated
            loopback HTTP endpoint.

        Raises
        ------
        ValueError
            If ``session_id`` is empty.
        """

        if not session_id:
            raise ValueError("tool UI session ID must not be empty")
        async with self._lock:
            self._current_session_id = session_id
            history_payloads = self._bind_history_payloads(session_id)
            live_statuses = []
            for call_id, status in self._live_status_payloads.items():
                if status.get("sessionId") is None:
                    status["sessionId"] = session_id
                    self._call_sessions[call_id] = session_id
                if status.get("sessionId") == session_id:
                    live_statuses.append(dict(status))
        return [*history_payloads, *live_statuses]

    async def publish_status(
        self,
        *,
        binding: ToolUIBinding,
        tool_name: str,
        call_id: str,
        status: str,
        running: bool,
    ) -> None:
        """Publish one deduplicated live status observation."""

        normalized_status = str(status)
        latest = (normalized_status, running)
        if self._last_status.get(call_id) == latest:
            return
        self._last_status[call_id] = latest
        sequence = self._status_sequences.get(call_id, 0) + 1
        self._status_sequences[call_id] = sequence
        payload = (
            self._event_base(
                event_type="tool_ui.status",
                binding=binding,
                tool_name=tool_name,
                call_id=call_id,
            )
            | {
                "sequence": sequence,
                "status": normalized_status,
                "running": running,
                "updatedAt": _utc_now(),
            }
        )
        if running:
            self._live_status_payloads[call_id] = payload
        else:
            self._live_status_payloads.pop(call_id, None)
        await self._broadcast(payload)
        if not running:
            self._last_status.pop(call_id, None)

    async def publish_emit(
        self,
        *,
        binding: ToolUIBinding,
        tool_name: str,
        call_id: str,
        message: str,
        status: str,
        running: bool,
    ) -> None:
        """Publish one immutable tool emit observation."""

        sequence = self._emit_sequences.get(call_id, 0) + 1
        self._emit_sequences[call_id] = sequence
        payload = (
            self._event_base(
                event_type="tool_ui.emit",
                binding=binding,
                tool_name=tool_name,
                call_id=call_id,
            )
            | {
                "sequence": sequence,
                "message": str(message),
                "status": str(status),
                "running": running,
                "emittedAt": _utc_now(),
            }
        )
        self._retain_history_payload(payload)
        await self._broadcast(payload)
        if not running:
            self.finish_call(call_id)

    def finish_call(self, call_id: str) -> None:
        """Release broker bookkeeping after a terminal call observation."""

        self._call_sessions.pop(call_id, None)
        self._status_sequences.pop(call_id, None)
        self._emit_sequences.pop(call_id, None)
        self._last_status.pop(call_id, None)
        self._live_status_payloads.pop(call_id, None)

    async def create_frame_ticket(self, source: str) -> str:
        """Create a runtime-scoped reloadable ticket for a UI document.

        Parameters
        ----------
        source : str
            Complete sandbox frame document prepared by the trusted App UI.

        Returns
        -------
        str
            Cryptographically random ticket safe to place in an iframe URL.

        Raises
        ------
        ValueError
            If the rendered document exceeds the bounded frame size.
        """

        if len(source.encode("utf-8")) > MAX_TOOL_UI_FRAME_BYTES:
            raise ValueError("tool UI frame exceeds the two MiB size limit")
        ticket = secrets.token_urlsafe(32)
        async with self._lock:
            self._frame_tickets[ticket] = _ToolUIFrameTicket(
                source=source,
            )
            self._prune_frame_tickets()
        return ticket

    async def consume_frame_ticket(self, ticket: str) -> str | None:
        """Return one sandbox frame document retained by this runtime.

        Parameters
        ----------
        ticket : str
            Opaque reloadable ticket created by
            :meth:`create_frame_ticket`.

        Returns
        -------
        str | None
            Frame HTML, or ``None`` when the ticket is invalid or evicted.
        """

        async with self._lock:
            frame = self._frame_tickets.get(ticket)
            if frame is None:
                return None
            return frame.source

    def _prune_frame_tickets(self) -> None:
        while len(self._frame_tickets) > MAX_TOOL_UI_FRAME_TICKETS:
            oldest_ticket = next(iter(self._frame_tickets))
            self._frame_tickets.pop(oldest_ticket, None)

    def _bind_history_payloads(
        self,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        """Bind pending history events and return replay copies for a session."""

        if session_id is None:
            return []
        pending = self._history_payloads.pop(None, [])
        if pending:
            for payload in pending:
                payload["sessionId"] = session_id
                call_id = str(payload["callId"])
                if self._call_sessions.get(call_id) is None:
                    self._call_sessions[call_id] = session_id
            existing = self._history_payloads.get(session_id, [])
            known_ids = {
                (payload["callId"], payload["sequence"])
                for payload in existing
            }
            existing.extend(
                payload
                for payload in pending
                if (payload["callId"], payload["sequence"]) not in known_ids
            )
            self._history_payloads[session_id] = existing[
                -MAX_REPLAYED_TOOL_UI_HISTORY_ITEMS:
            ]
        return [
            dict(payload)
            for payload in self._history_payloads.get(session_id, [])
        ]

    def _retain_history_payload(self, payload: dict[str, Any]) -> None:
        """Retain one bounded immutable emit for reconnect recovery."""

        session_id = payload.get("sessionId")
        if session_id is not None and not isinstance(session_id, str):
            return
        history = self._history_payloads.setdefault(session_id, [])
        history.append(dict(payload))
        del history[:-MAX_REPLAYED_TOOL_UI_HISTORY_ITEMS]

    def _event_base(
        self,
        *,
        event_type: str,
        binding: ToolUIBinding,
        tool_name: str,
        call_id: str,
    ) -> dict[str, Any]:
        if call_id not in self._call_sessions:
            self._call_sessions[call_id] = self._current_session_id
        return {
            "type": event_type,
            "toolId": binding.tool_id,
            "toolName": tool_name,
            "callId": call_id,
            "sessionId": self._call_sessions[call_id],
        }

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        failed: list[WebSocket] = []
        for client in clients:
            try:
                await client.send_json(payload)
            except Exception:
                failed.append(client)
        if failed:
            async with self._lock:
                self._clients = [
                    client for client in self._clients if client not in failed
                ]


def wrap_tools_with_ui(
    tools: list[Any],
    *,
    binding: ToolUIBinding,
    broker: ToolUIBroker,
) -> list[Any]:
    """Wrap native asynchronous tools with read-only App observers.

    Parameters
    ----------
    tools : list[Any]
        Values returned by the unchanged developer tool factory.
    binding : ToolUIBinding
        Installed UI metadata used only by the App wrapper.
    broker : ToolUIBroker
        Event destination for status and emit observations.

    Returns
    -------
    list[Any]
        Original values with native ``AsyncTool`` classes transparently
        replaced by behavior-preserving subclasses.
    """

    return [
        _wrap_async_tool(tool, binding=binding, broker=broker)
        if isinstance(tool, type) and issubclass(tool, AsyncTool)
        else tool
        for tool in tools
    ]


def _wrap_async_tool(
    original: type[AsyncTool],
    *,
    binding: ToolUIBinding,
    broker: ToolUIBroker,
) -> type[AsyncTool]:
    tool_name = original.name or original.__name__

    async def aemit_initial(cls, tool_call_id, tool_input, tool_state, global_state):
        """Delegate the initial emit and publish its read-only observation."""

        del cls
        result = await original.aemit_initial(
            tool_call_id,
            tool_input,
            tool_state,
            global_state,
        )
        status = await _safe_status(
            original,
            tool_input,
            tool_state,
            global_state,
        )
        await broker.publish_status(
            binding=binding,
            tool_name=tool_name,
            call_id=tool_call_id,
            status=status,
            running=True,
        )
        await broker.publish_emit(
            binding=binding,
            tool_name=tool_name,
            call_id=tool_call_id,
            message=result.content,
            status=status,
            running=True,
        )
        return result

    async def aemit_updates(
        cls,
        tool_input,
        tool_state,
        global_state,
    ) -> AsyncIterator[Any]:
        """Delegate updates while observing periodic status and every emit."""

        del cls
        call_id = tool_state.call_id
        updates = original.aemit_updates(
            tool_input,
            tool_state,
            global_state,
        ).__aiter__()
        try:
            while True:
                next_update = asyncio.create_task(anext(updates))
                try:
                    result = await _wait_for_update_with_status(
                        next_update,
                        original=original,
                        binding=binding,
                        broker=broker,
                        tool_name=tool_name,
                        call_id=call_id,
                        tool_input=tool_input,
                        tool_state=tool_state,
                        global_state=global_state,
                    )
                except StopAsyncIteration:
                    status = await _safe_status(
                        original,
                        tool_input,
                        tool_state,
                        global_state,
                    )
                    await broker.publish_status(
                        binding=binding,
                        tool_name=tool_name,
                        call_id=call_id,
                        status=status,
                        running=False,
                    )
                    broker.finish_call(call_id)
                    return

                running = isinstance(result, Running)
                status = await _safe_status(
                    original,
                    tool_input,
                    tool_state,
                    global_state,
                )
                if binding.update_every_s != -1.0 or not running:
                    await broker.publish_status(
                        binding=binding,
                        tool_name=tool_name,
                        call_id=call_id,
                        status=status,
                        running=running,
                    )
                message = (
                    result.content
                    if isinstance(result, Running)
                    else result.content.to_content()
                )
                await broker.publish_emit(
                    binding=binding,
                    tool_name=tool_name,
                    call_id=call_id,
                    message=message,
                    status=status,
                    running=running,
                )
                yield result
                if isinstance(result, Finished):
                    return
        except asyncio.CancelledError:
            status = await _safe_status(
                original,
                tool_input,
                tool_state,
                global_state,
            )
            await broker.publish_status(
                binding=binding,
                tool_name=tool_name,
                call_id=call_id,
                status=status,
                running=False,
            )
            broker.finish_call(call_id)
            raise
        except BaseException as exc:
            if not isinstance(exc, StopAsyncIteration):
                await broker.publish_status(
                    binding=binding,
                    tool_name=tool_name,
                    call_id=call_id,
                    status=f"Tool failed ({type(exc).__name__})",
                    running=False,
                )
                broker.finish_call(call_id)
            raise

    attributes = {
        "__module__": original.__module__,
        "__doc__": original.__doc__,
        "aemit_initial": classmethod(aemit_initial),
        "aemit_updates": classmethod(aemit_updates),
    }
    return type(original.__name__, (original,), attributes)


async def _wait_for_update_with_status(
    next_update: asyncio.Task[Any],
    *,
    original: type[AsyncTool],
    binding: ToolUIBinding,
    broker: ToolUIBroker,
    tool_name: str,
    call_id: str,
    tool_input: Any,
    tool_state: Any,
    global_state: Any,
) -> Any:
    if binding.update_every_s == -1.0:
        return await next_update

    try:
        while True:
            done, _ = await asyncio.wait(
                {next_update},
                timeout=binding.update_every_s,
            )
            if next_update in done:
                return next_update.result()
            status = await _safe_status(
                original,
                tool_input,
                tool_state,
                global_state,
            )
            await broker.publish_status(
                binding=binding,
                tool_name=tool_name,
                call_id=call_id,
                status=status,
                running=True,
            )
    except BaseException:
        next_update.cancel()
        raise


async def _safe_status(
    original: type[AsyncTool],
    tool_input: Any,
    tool_state: Any,
    global_state: Any,
) -> str:
    try:
        return str(
            await original.astatus(
                tool_input,
                tool_state,
                global_state,
            )
        )
    except Exception as exc:
        return f"Status unavailable ({type(exc).__name__})"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
