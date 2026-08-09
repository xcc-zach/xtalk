"""Read-only UI observation for developer-installed asynchronous tools."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import WebSocket, WebSocketDisconnect
from xtalk.models.agents.tools import AsyncTool, Finished, Running, ToolState

from .desktop_tool_bridge import DesktopToolCallBridge


MAX_TOOL_UI_FRAME_BYTES = 2 * 1024 * 1024
MAX_TOOL_UI_FRAME_TICKETS = 400
MAX_REPLAYED_TOOL_UI_HISTORY_ITEMS = 200
MAX_TOOL_UI_EMIT_PAYLOAD_BYTES = 256 * 1024


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

    def __init__(
        self,
        bridge: DesktopToolCallBridge | None = None,
    ) -> None:
        """Initialize empty connection and tool-call state.

        Parameters
        ----------
        bridge : DesktopToolCallBridge | None, optional
            Shared bridge that maps tool calls to assistant-text offsets.
        """

        self._clients: list[WebSocket] = []
        self._bridge = bridge
        self._lock = asyncio.Lock()
        self._current_session_id: str | None = None
        self._call_sessions: dict[str, str | None] = {}
        self._call_offsets: dict[str, int] = {}
        self._status_sequences: dict[str, int] = {}
        self._emit_sequences: dict[str, int] = {}
        self._last_status: dict[str, tuple[str, bool]] = {}
        self._live_status_payloads: dict[str, dict[str, Any]] = {}
        self._history_payloads: dict[str | None, list[dict[str, Any]]] = {}
        self._frame_tickets: dict[str, _ToolUIFrameTicket] = {}

    def register_ui_tool(self, tool_name: str) -> None:
        """Declare one tool whose timeline rows should carry text offsets.

        Parameters
        ----------
        tool_name : str
            Exported tool name used by the LLM tool-call marker.
        """

        if self._bridge is not None:
            self._bridge.register_ui_tool(tool_name)

    def current_session_id(self) -> str | None:
        """Return the session most recently bound by the App UI.

        Returns
        -------
        str | None
            Active persisted session identifier, or ``None`` before binding.
        """

        return self._current_session_id

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
                    for status in self._live_status_payloads.values():
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
            for status in self._live_status_payloads.values():
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
        session_id: str | None = None,
    ) -> None:
        """Publish one deduplicated live status observation.

        Parameters
        ----------
        binding : ToolUIBinding
            Installed UI metadata for the observed tool.
        tool_name : str
            Exported tool name.
        call_id : str
            Stable source tool-call identifier.
        status : str
            Human-readable current status.
        running : bool
            Whether the call remains active.
        session_id : str | None, optional
            Backend-owned session identifier. App display bindings never
            determine tool ownership.
        """

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
                session_id=session_id,
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
        outcome: Literal["running", "complete", "cancelled"] | None = None,
        payload: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Publish one immutable tool emit observation.

        Parameters
        ----------
        binding : ToolUIBinding
            Installed UI metadata for the observed tool.
        tool_name : str
            Exported tool name.
        call_id : str
            Stable source tool-call identifier.
        message : str
            Human-readable content emitted by the tool.
        status : str
            Latest human-readable status observation.
        running : bool
            Whether the tool call remains active.
        outcome : {"running", "complete", "cancelled"} | None, optional
            Explicit lifecycle outcome. When omitted, it is derived from
            ``running`` for compatibility with existing callers.
        payload : dict[str, Any] | None, optional
            Optional structured content emitted by the tool. Payloads that
            exceed the emit size limit are dropped while ``message`` is kept.
        session_id : str | None, optional
            Backend-owned session identifier. The UI may filter by this value
            but cannot assign it.

        Raises
        ------
        ValueError
            If the explicit outcome contradicts ``running``.
        """

        resolved_outcome = outcome or ("running" if running else "complete")
        if running != (resolved_outcome == "running"):
            raise ValueError("tool UI outcome contradicts running state")

        sequence = self._emit_sequences.get(call_id, 0) + 1
        self._emit_sequences[call_id] = sequence
        payload_base = (
            self._event_base(
                event_type="tool_ui.emit",
                binding=binding,
                tool_name=tool_name,
                call_id=call_id,
                session_id=session_id,
            )
        )
        if self._bridge is not None and sequence == 1:
            session_id = payload_base.get("sessionId")
            if isinstance(session_id, str) and session_id:
                offset = self._bridge.consume_tool_offset(
                    session_id=session_id
                )
                if offset is not None:
                    self._call_offsets[call_id] = offset
        event: dict[str, Any] = payload_base | {
            "sequence": sequence,
            "message": str(message),
            "status": str(status),
            "running": running,
            "outcome": resolved_outcome,
            "emittedAt": _utc_now(),
        }
        if payload is not None:
            bounded_payload = _bounded_payload(payload)
            if bounded_payload is not None:
                event["payload"] = bounded_payload
        if self._call_offsets.get(call_id) is not None:
            event["textOffset"] = self._call_offsets[call_id]
        self._retain_history_payload(event)
        await self._broadcast(event)
        if not running:
            self.finish_call(call_id)

    def finish_call(self, call_id: str) -> None:
        """Release broker bookkeeping after a terminal call observation."""

        self._call_sessions.pop(call_id, None)
        self._call_offsets.pop(call_id, None)
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
        """Return replay copies already owned by one backend session."""

        if session_id is None:
            return []
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
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if session_id is not None:
            self._call_sessions[call_id] = session_id
        return {
            "type": event_type,
            "toolId": binding.tool_id,
            "toolName": tool_name,
            "callId": call_id,
            "sessionId": self._call_sessions.get(call_id),
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

    wrapped: list[Any] = []
    for tool in tools:
        if isinstance(tool, type) and issubclass(tool, AsyncTool):
            broker.register_ui_tool(tool.name or tool.__name__)
            wrapped.append(
                _wrap_async_tool(
                    tool,
                    binding=binding,
                    broker=broker,
                )
            )
        else:
            wrapped.append(tool)
    return wrapped


def _bind_tool_session(
    tool_state: ToolState,
    global_state: Any,
) -> str | None:
    """Bind backend-owned session context to one asynchronous tool call.

    Parameters
    ----------
    tool_state : ToolState
        Mutable state shared by one asynchronous tool call. Per-call metadata
        takes precedence when provided by a custom engine.
    global_state : Any
        Shared tool-engine state carrying App session context.

    Returns
    -------
    str | None
        Non-empty backend session identifier bound to the tool call, when one
        was injected by the desktop tool engine.
    """

    session_id = tool_state.metadata.get("session_id")
    if not session_id and isinstance(global_state, dict):
        session_id = global_state.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    tool_state.metadata["session_id"] = session_id
    return session_id


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
        session_id = _bind_tool_session(tool_state, global_state)
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
            session_id=session_id,
        )
        await broker.publish_emit(
            binding=binding,
            tool_name=tool_name,
            call_id=tool_call_id,
            message=result.content,
            status=status,
            running=True,
            payload=_structured_payload(original, result.content),
            session_id=session_id,
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
        session_id = _bind_tool_session(tool_state, global_state)
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
                        session_id=session_id,
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
                        session_id=session_id,
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
                        session_id=session_id,
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
                    payload=_structured_payload(original, message),
                    session_id=session_id,
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
            cancellation_message = status or "Tool cancelled"
            await broker.publish_emit(
                binding=binding,
                tool_name=tool_name,
                call_id=call_id,
                message=cancellation_message,
                status=status,
                running=False,
                outcome="cancelled",
                session_id=session_id,
            )
            raise
        except BaseException as exc:
            if not isinstance(exc, StopAsyncIteration):
                await broker.publish_status(
                    binding=binding,
                    tool_name=tool_name,
                    call_id=call_id,
                    status=f"Tool failed ({type(exc).__name__})",
                    running=False,
                    session_id=session_id,
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
    session_id: str | None,
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
                session_id=session_id,
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


def _structured_payload(
    original: type[AsyncTool],
    message: str,
) -> dict[str, Any] | None:
    """Return one tool's structured emit payload when it declares one.

    Parameters
    ----------
    original : type[AsyncTool]
        Unwrapped tool class whose structured-content declaration is read.
    message : str
        Emit message produced by the tool.

    Returns
    -------
    dict[str, Any] | None
        Parsed JSON object payload, or ``None`` when the tool does not
        declare structured content or the message is not a JSON object.
    """

    if not getattr(original, "structured_payload", False):
        return None
    try:
        decoded = json.loads(message)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _bounded_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a payload that fits the emit limit, otherwise drop it.

    Parameters
    ----------
    payload : dict[str, Any]
        Structured tool emit payload.

    Returns
    -------
    dict[str, Any] | None
        The bounded payload, or ``None`` when it exceeds the size limit or
        cannot be serialized.
    """

    try:
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    if len(encoded) > MAX_TOOL_UI_EMIT_PAYLOAD_BYTES:
        return None
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
