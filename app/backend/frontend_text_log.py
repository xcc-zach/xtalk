"""Log every text frame the sidecar pushes to desktop frontends.

The desktop WebView connects directly to the loopback sidecar over
WebSocket.  The App owns only the ASGI boundary in this package, so the
text push is captured here with a pure ASGI middleware that records each
``websocket.send`` text frame into an append-only JSONL file.  This is a
diagnostic aid for comparing what was streamed to the frontend against
what eventually lands in the persisted chat history.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable


ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]

_WRITE_LOCK = threading.Lock()


def _extract_text(payload: dict[str, Any]) -> str:
    """Return the human-facing text carried by one frontend payload.

    Parameters
    ----------
    payload : dict[str, Any]
        Decoded ``{action, data}`` WebSocket frame.

    Returns
    -------
    str
        Text field when the payload carries one, otherwise an empty
        string for binary or purely structural signals.
    """

    data = payload.get("data")
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return ""
    for key in ("text", "display_text", "content"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    message = data.get("message")
    if isinstance(message, dict):
        inner = message.get("content")
        if isinstance(inner, str):
            return inner
    return ""


class FrontendTextLogMiddleware:
    """Record every JSON text frame sent to a desktop WebSocket client.

    Parameters
    ----------
    app : ASGIApp
        Wrapped ASGI application.
    log_path : str | pathlib.Path
        Append-only JSONL log receiving one line per pushed text frame.
    """

    def __init__(self, app: ASGIApp, *, log_path: str | Path) -> None:
        self._app = app
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        """Wrap websocket sends and dispatch everything else untouched."""

        if scope.get("type") != "websocket":
            await self._app(scope, receive, send)
            return

        session_id = "unknown"

        async def _logged_send(message: dict[str, Any]) -> None:
            nonlocal session_id
            if message.get("type") == "websocket.send" and "text" in message:
                session_id = self._record(message["text"], session_id)
            await send(message)

        await self._app(scope, receive, _logged_send)

    def _record(self, raw_text: str, session_id: str) -> str:
        """Write one log line and return the connection's resolved session."""

        try:
            payload = json.loads(raw_text)
        except (TypeError, ValueError):
            payload = {"raw": raw_text}

        action = payload.get("action") if isinstance(payload, dict) else ""
        if action == "session_attached":
            data = payload.get("data")
            if isinstance(data, dict) and isinstance(data.get("session_id"), str):
                session_id = data["session_id"]

        line = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "session": session_id,
            "action": action,
            "text": (
                _extract_text(payload)
                if isinstance(payload, dict)
                else raw_text
            ),
            "payload": payload,
        }
        try:
            with _WRITE_LOCK, self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        line,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
        except Exception:
            # Diagnostics must never break the live push path.
            pass
        return session_id
