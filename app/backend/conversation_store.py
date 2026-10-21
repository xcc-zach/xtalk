from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any


def _build_title(text: str, *, max_chars: int = 32) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 1].rstrip()}…"


class PersistenceStore:
    """SQLite-backed storage for users, sessions, and final text messages."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                    ON chat_sessions(user_id);

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages(session_id, id ASC);
                """
            )

            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)")
            }
            if "item_id" not in columns:
                conn.execute("ALTER TABLE chat_messages ADD COLUMN item_id TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_messages_item ON chat_messages(session_id, item_id)"
            )

    def upsert_assistant(
        self, *, user_id: str, session_id: str, item_id: str, content: str
    ) -> None:
        """Apply a playback snapshot to one assistant message without duplication.

        Parameters
        ----------
        user_id : str
            Authenticated conversation owner.
        session_id : str
            Application conversation identifier.
        item_id : str
            Stable external item identifier.
        content : str
            Full text confirmed as delivered to the user.
        """
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        if not content.strip():
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, item_id) VALUES (?, 'assistant', ?, ?) ON CONFLICT(session_id, item_id) DO UPDATE SET content = excluded.content",
                (session_id, content, item_id),
            )

    def ensure_user(self, user_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (id) VALUES (?)",
                (user_id,),
            )
        return {"id": user_id}

    def create_session(
        self, user_id: str, *, session_id: str | None = None
    ) -> dict[str, Any]:
        self.ensure_user(user_id)
        resolved_session_id = session_id or str(uuid.uuid4())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, user_id, title)
                VALUES (?, ?, NULL)
                """,
                (resolved_session_id, user_id),
            )
        return {
            "session_id": resolved_session_id,
            "title": None,
        }

    def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, title
                FROM chat_sessions
                WHERE id = ? AND user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "title": row["title"],
        }

    def user_owns_session(self, user_id: str, session_id: str) -> bool:
        return self.get_session(user_id, session_id) is not None

    def get_or_create_session(
        self, user_id: str, session_id: str | None = None
    ) -> dict[str, Any]:
        if session_id is None:
            return self.create_session(user_id)
        session = self.get_session(user_id, session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.title, COALESCE(MAX(m.id), 0) AS activity_rank
                FROM chat_sessions AS s
                LEFT JOIN chat_messages AS m ON m.session_id = s.id
                WHERE s.user_id = ?
                GROUP BY s.id, s.title, s.rowid
                ORDER BY activity_rank DESC, s.rowid DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            {
                "session_id": str(row["id"]),
                "title": row["title"],
            }
            for row in rows
        ]

    def list_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "role": str(row["role"]),
                "content": str(row["content"]),
            }
            for row in rows
        ]

    def get_session_detail(self, user_id: str, session_id: str) -> dict[str, Any]:
        session = self.get_session(user_id, session_id)
        if session is None:
            raise KeyError(session_id)
        return {
            "session_id": session["session_id"],
            "title": session["title"],
            "messages": self.list_messages(user_id, session_id),
        }

    def append_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        normalized = content.strip()
        if not normalized:
            return
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, normalized),
            )

            if role == "user":
                row = conn.execute(
                    "SELECT title FROM chat_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                title = row["title"] if row is not None else None
                if not title:
                    conn.execute(
                        "UPDATE chat_sessions SET title = ? WHERE id = ?",
                        (_build_title(normalized), session_id),
                    )
