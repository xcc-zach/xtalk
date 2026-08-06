"""Per-session whiteboard stores shared by the tools and the sidecar API.

Every conversation owns an independent whiteboard document. The store keeps
one process-wide registry keyed by session id and persists each document as
JSON under the tool data directory, so boards survive sidecar restarts and
never mix between conversations.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WHITEBOARD_STORE_VERSION = 1
WHITEBOARD_STORE_FILENAME = "whiteboards"

_SAFE_SESSION_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_SESSION_ID = "default"
_STORES_LOCK = threading.RLock()
_STORES: dict[str, WhiteboardStore] = {}
_DATA_DIRECTORY: Path | None = None


def _utc_now() -> str:
    """Return one sortable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_session_id(session_id: str) -> str:
    """Normalize a session id into a safe file-name component."""

    normalized = _SAFE_SESSION_PATTERN.sub("-", session_id).strip("-")
    return normalized or _DEFAULT_SESSION_ID


class WhiteboardStore:
    """One conversation's whiteboard text document persisted to a JSON file.

    Parameters
    ----------
    path : pathlib.Path | None
        Optional JSON file used for persistence. ``None`` keeps the document
        in memory for the lifetime of the process.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._text = ""
        self._revision = 0
        self._updated_at = _utc_now()

    def snapshot(self) -> dict[str, Any]:
        """Return the normalized snapshot of the current document.

        Returns
        -------
        dict[str, Any]
            Serializable ``{version, text, revision, updated_at}`` snapshot
            consumed by the tools and the read-only API.
        """

        with self._lock:
            self._ensure_loaded()
            return {
                "version": WHITEBOARD_STORE_VERSION,
                "text": self._text,
                "revision": self._revision,
                "updated_at": self._updated_at,
            }

    def add_text(self, text: str) -> dict[str, Any]:
        """Append one text block to the end of the document.

        The store joins blocks with a single newline so Markdown sections stay
        separable without altering the caller's content.

        Parameters
        ----------
        text : str
            Non-empty Markdown text to append.

        Returns
        -------
        dict[str, Any]
            Snapshot after the mutation.
        """

        with self._lock:
            self._ensure_loaded()
            if self._text and text:
                self._text = self._text.rstrip("\n") + "\n" + text
            elif text:
                self._text = text
            self._touch()
            return self.snapshot()

    def delete_text(self, text: str) -> dict[str, Any]:
        """Remove every occurrence of one exact text block.

        Parameters
        ----------
        text : str
            Non-empty text that must exist somewhere in the document.

        Returns
        -------
        dict[str, Any]
            Snapshot after the mutation.

        Raises
        ------
        ValueError
            When the requested text is not present in the document.
        """

        with self._lock:
            self._ensure_loaded()
            if not text or text not in self._text:
                raise ValueError("whiteboard text to delete was not found")
            self._text = _remove_text_blocks(self._text, text)
            self._touch()
            return self.snapshot()

    def update_text(self, from_text: str, to_text: str) -> dict[str, Any]:
        """Replace every occurrence of one text block with another.

        Parameters
        ----------
        from_text : str
            Non-empty text that must exist somewhere in the document.
        to_text : str
            Replacement text; may be empty to remove the block.

        Returns
        -------
        dict[str, Any]
            Snapshot after the mutation.

        Raises
        ------
        ValueError
            When ``from_text`` is not present in the document.
        """

        with self._lock:
            self._ensure_loaded()
            if not from_text or from_text not in self._text:
                raise ValueError("whiteboard text to update was not found")
            if to_text:
                self._text = _collapse_blank_lines(
                    self._text.replace(from_text, to_text).strip("\n")
                )
            else:
                self._text = _remove_text_blocks(self._text, from_text)
            self._touch()
            return self.snapshot()

    def reset(self) -> None:
        """Clear the document for tests and manual resets."""

        with self._lock:
            self._text = ""
            self._revision = 0
            self._updated_at = _utc_now()
            self._persist()

    def _ensure_loaded(self) -> None:
        """Load persisted state once when a store is first used."""

        if self._revision > 0 or self._text or self._path is None:
            return
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        text = payload.get("text")
        revision = payload.get("revision")
        updated_at = payload.get("updated_at")
        if isinstance(text, str):
            self._text = text
        if isinstance(revision, int) and revision > 0:
            self._revision = revision
        if isinstance(updated_at, str):
            self._updated_at = updated_at

    def _touch(self) -> None:
        """Advance the revision and persist the mutated document."""

        self._revision += 1
        self._updated_at = _utc_now()
        self._persist()

    def _persist(self) -> None:
        """Write the current document to the configured JSON file."""

        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "version": WHITEBOARD_STORE_VERSION,
                    "text": self._text,
                    "revision": self._revision,
                    "updated_at": self._updated_at,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def get_whiteboard_store(session_id: str | None = None) -> WhiteboardStore:
    """Return one conversation's whiteboard store singleton.

    Parameters
    ----------
    session_id : str | None, optional
        Persisted chat session owning the board. ``None`` uses a shared
        default store for tools that run without session context.

    Returns
    -------
    WhiteboardStore
        The conversation-scoped store backing the tools and the sidecar API.
    """

    key = _safe_session_id(session_id or _DEFAULT_SESSION_ID)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            path = (
                None
                if _DATA_DIRECTORY is None
                else _DATA_DIRECTORY / f"{key}.json"
            )
            store = WhiteboardStore(path)
            _STORES[key] = store
        return store


def configure_whiteboard_data_directory(
    directory: Path | str | None,
) -> None:
    """Point the session store registry at a persistent directory.

    Parameters
    ----------
    directory : pathlib.Path | str | None
        Directory holding one JSON file per conversation, or ``None`` for
        memory-only stores.
    """

    global _DATA_DIRECTORY
    with _STORES_LOCK:
        _DATA_DIRECTORY = (
            Path(directory).expanduser() if directory is not None else None
        )
        _STORES.clear()


def reset_whiteboard_stores() -> None:
    """Reset the registry to fresh in-memory stores for tests."""

    configure_whiteboard_data_directory(None)


def whiteboard_store_path(session_id: str | None = None) -> Path | None:
    """Return the persistence path used for one conversation's board.

    Parameters
    ----------
    session_id : str | None, optional
        Session whose board path is requested.

    Returns
    -------
    pathlib.Path | None
        JSON path, or ``None`` when no data directory is configured.
    """

    if _DATA_DIRECTORY is None:
        return None
    return _DATA_DIRECTORY / f"{_safe_session_id(session_id)}.json"


def _collapse_blank_lines(value: str) -> str:
    """Replace runs of three or more newlines with a single blank line."""

    while "\n\n\n" in value:
        value = value.replace("\n\n\n", "\n\n")
    return value


def _remove_text_blocks(value: str, text: str) -> str:
    """Remove one text block together with its following separator newline."""

    pattern = re.compile(re.escape(text) + r"\n?")
    return _collapse_blank_lines(pattern.sub("", value).strip("\n"))
