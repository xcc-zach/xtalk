"""Built-in Codex tools with a persistent App-local session index."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field
from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)


SESSION_RESOLVER_INSTRUCTIONS = """Select Codex sessions only from the JSON candidates supplied by the user.
Use the natural-language query, project path, topic, timestamps, summary, and recent task.
Return matched for exactly one confident result, ambiguous for multiple plausible results,
or not_found when none is plausible. Never inspect local Codex session storage or modify files."""

SESSION_RESOLVER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["matched", "ambiguous", "not_found"],
        },
        "selected_session_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "reason": {"type": "string"},
    },
    "required": ["status", "selected_session_ids", "reason"],
    "additionalProperties": False,
}

CODEX_INSTALL_MESSAGE = (
    "Codex CLI is not installed or could not be started. Install it with "
    "`npm install -g @openai/codex`, then fully restart XTalk."
)
_CODEX_CACHE_LOCK = threading.RLock()
_CODEX_CACHE: tuple[Path, dict[str, str], str] | None = None


def _utc_now() -> str:
    """Return one sortable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _compact_text(value: str, limit: int) -> str:
    """Collapse whitespace and truncate text for session metadata."""

    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _markdown_table_cell(value: str) -> str:
    """Escape one plain-text value for a compact Markdown table cell."""

    return " ".join(value.split()).replace("\\", "\\\\").replace("|", "\\|")


def _resolve_working_directory(value: str) -> Path:
    """Resolve and validate an unrestricted local working directory."""

    directory = Path(value).expanduser().resolve()
    if not directory.exists():
        raise ValueError(f"Working directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Working directory is not a directory: {directory}")
    return directory


def _absolute_path_without_resolving_links(path: Path) -> Path:
    """Expand a path while preserving an npm-managed executable symlink."""

    return Path(os.path.abspath(path.expanduser()))


def _candidate_codex_paths() -> Iterator[Path]:
    """Yield user-installed Codex executables in priority order."""

    candidates: list[Path] = []
    discovered = shutil.which("codex")
    if discovered:
        candidates.append(Path(discovered))

    home = Path.home()
    candidates.extend(
        (
            home / ".volta" / "bin" / "codex",
            home / ".local" / "bin" / "codex",
            home / ".npm-global" / "bin" / "codex",
            home / ".bun" / "bin" / "codex",
            Path("/opt/homebrew/bin/codex"),
            Path("/usr/local/bin/codex"),
            Path("/usr/bin/codex"),
            Path("/snap/bin/codex"),
        )
    )
    candidates.extend(
        sorted(
            (home / ".nvm" / "versions" / "node").glob("*/bin/codex"),
            reverse=True,
        )
    )
    for fnm_root in (
        home / ".local" / "share" / "fnm" / "node-versions",
        home / "Library" / "Application Support" / "fnm" / "node-versions",
    ):
        candidates.extend(
            sorted(
                fnm_root.glob("*/installation/bin/codex"),
                reverse=True,
            )
        )
    app_data = os.environ.get("APPDATA", "").strip()
    if app_data:
        candidates.extend(
            (
                Path(app_data) / "npm" / "codex.exe",
                Path(app_data) / "npm" / "codex.cmd",
            )
        )

    seen: set[str] = set()
    for candidate in candidates:
        absolute = _absolute_path_without_resolving_links(candidate)
        key = os.path.normcase(str(absolute))
        if key in seen:
            continue
        seen.add(key)
        yield absolute


def _codex_environment(executable: Path) -> dict[str, str]:
    """Create a child environment that can run npm's Node.js shim."""

    environment = os.environ.copy()
    path_key = "PATH"
    if os.name == "nt":
        path_key = next(
            (key for key in environment if key.upper() == "PATH"),
            "Path",
        )
    existing = environment.get(path_key, "")
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    executable_directory = str(executable.parent)
    remaining_entries = [
        entry for entry in entries if entry != executable_directory
    ]
    environment[path_key] = os.pathsep.join(
        [executable_directory, *remaining_entries]
    )
    return environment


def _resolve_user_codex() -> tuple[Path, dict[str, str], str]:
    """Find, validate, and cache a user-installed Codex CLI executable.

    Returns
    -------
    tuple[pathlib.Path, dict[str, str], str]
        Executable, complete child environment, and reported CLI version.

    Raises
    ------
    RuntimeError
        If no candidate can execute ``codex --version`` successfully.
    """

    global _CODEX_CACHE

    with _CODEX_CACHE_LOCK:
        if _CODEX_CACHE is not None:
            cached_path, cached_environment, cached_version = _CODEX_CACHE
            if cached_path.is_file() and (
                os.name == "nt" or os.access(cached_path, os.X_OK)
            ):
                return cached_path, cached_environment.copy(), cached_version
            _CODEX_CACHE = None

        failures: list[str] = []
        for candidate in _candidate_codex_paths():
            if not candidate.is_file():
                continue
            if os.name != "nt" and not os.access(candidate, os.X_OK):
                failures.append(f"not executable: {candidate}")
                continue
            environment = _codex_environment(candidate)
            try:
                result = subprocess.run(
                    [str(candidate), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                    env=environment,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f"{candidate}: {exc}")
                continue
            version = (result.stdout or result.stderr).strip().splitlines()
            if result.returncode == 0 and version:
                _CODEX_CACHE = (candidate, environment.copy(), version[0])
                return candidate, environment, version[0]
            failures.append(
                f"{candidate}: version check exited with {result.returncode}"
            )

        detail = f" Checked: {'; '.join(failures)}" if failures else ""
        raise RuntimeError(CODEX_INSTALL_MESSAGE + detail)


@dataclass(frozen=True)
class SessionRecord:
    """Metadata for one persistent Codex thread in the active App pool."""

    session_id: str
    title: str
    summary: str
    cwd: str
    model: str | None
    effort: str | None
    status: str
    created_at: str
    updated_at: str
    last_task: str
    recent_tasks: tuple[str, ...]


class SessionStore:
    """SQLite-backed index for Codex threads owned by the local App."""

    def __init__(self, path: Path) -> None:
        """Create the database and its session table when needed."""

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    model TEXT,
                    effort TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_task TEXT NOT NULL,
                    recent_tasks TEXT NOT NULL DEFAULT '[]',
                    archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(sessions)"
                ).fetchall()
            }
            if "recent_tasks" not in columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN "
                    "recent_tasks TEXT NOT NULL DEFAULT '[]'"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            title=row["title"],
            summary=row["summary"],
            cwd=row["cwd"],
            model=row["model"],
            effort=row["effort"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_task=row["last_task"],
            recent_tasks=tuple(json.loads(row["recent_tasks"])),
        )

    def add(
        self,
        *,
        session_id: str,
        cwd: Path,
        task: str,
        model: str | None,
        effort: str | None,
    ) -> SessionRecord:
        """Add a newly created SDK thread to the active pool."""

        now = _utc_now()
        title = _compact_text(task, 80) or "Codex session"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, title, summary, cwd, model, effort, status,
                    created_at, updated_at, last_task, recent_tasks, archived
                ) VALUES (?, ?, '', ?, ?, ?, 'running', ?, ?, ?, ?, 0)
                """,
                (
                    session_id,
                    title,
                    str(cwd),
                    model,
                    effort,
                    now,
                    now,
                    _compact_text(task, 240),
                    json.dumps([_compact_text(task, 240)]),
                ),
            )
        return self.require(session_id)

    def require(self, session_id: str) -> SessionRecord:
        """Return one active session or raise a user-facing lookup error."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND archived = 0",
                (session_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Active Codex session not found: {session_id}")
        return self._record(row)

    def list_active(self) -> list[SessionRecord]:
        """Return active sessions from newest to oldest."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sessions
                WHERE archived = 0
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def candidates(self, query: str, limit: int = 30) -> list[SessionRecord]:
        """Mechanically bound resolver candidates without selecting a result."""

        records = self.list_active()
        tokens = {token.casefold() for token in query.split() if len(token) > 1}

        def score(record: SessionRecord) -> int:
            haystack = " ".join(
                (
                    record.session_id,
                    record.title,
                    record.summary,
                    record.cwd,
                    record.last_task,
                    " ".join(record.recent_tasks),
                    record.created_at,
                    record.updated_at,
                )
            ).casefold()
            return sum(1 for token in tokens if token in haystack)

        if tokens:
            records.sort(key=lambda record: score(record), reverse=True)
        return records[:limit]

    def mark_running(self, session_id: str, *, task: str, cwd: Path) -> None:
        """Record the start of a new turn."""

        record = self.require(session_id)
        compact_task = _compact_text(task, 240)
        recent_tasks = [*record.recent_tasks, compact_task][-5:]
        self._update(
            session_id,
            status="running",
            cwd=str(cwd),
            last_task=compact_task,
            recent_tasks=json.dumps(recent_tasks),
        )

    def mark_finished(self, session_id: str, response: str) -> None:
        """Record a successful turn and compact response summary."""

        self._update(
            session_id,
            status="idle",
            summary=_compact_text(response, 360),
        )

    def mark_failed(self, session_id: str, error: str) -> None:
        """Record a failed turn while keeping its thread resumable."""

        self._update(
            session_id,
            status="failed",
            summary=_compact_text(error, 360),
        )

    def set_model(
        self,
        session_id: str,
        *,
        model: str,
        effort: str | None,
    ) -> SessionRecord:
        """Persist model settings used for all future turns."""

        values: dict[str, Any] = {"model": model, "status": "idle"}
        if effort is not None:
            values["effort"] = effort
        self._update(session_id, **values)
        return self.require(session_id)

    def archive(self, session_id: str) -> None:
        """Remove a session from the active pool after SDK archival."""

        self.require(session_id)
        self._update(session_id, archived=1, status="archived")

    def _update(self, session_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = _utc_now()
        assignments = ", ".join(f"{column} = ?" for column in values)
        parameters = [*values.values(), session_id]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE sessions SET {assignments} WHERE session_id = ?",
                parameters,
            )
        if cursor.rowcount != 1:
            raise ValueError(f"Active Codex session not found: {session_id}")


class OpenAICodexAdapter:
    """Narrow async adapter around the official ``openai-codex`` SDK."""

    def __init__(self) -> None:
        """Defer importing and starting the optional SDK until first use."""

        self._client: Any = None
        self._approval_mode: Any = None
        self._sandbox: Any = None
        self._reasoning_effort: Any = None

    async def __aenter__(self) -> OpenAICodexAdapter:
        """Start one SDK client and reuse existing Codex authentication."""

        try:
            from openai_codex import (
                ApprovalMode,
                AsyncCodex,
                CodexConfig,
                Sandbox,
            )
            from openai_codex.types import ReasoningEffort
        except ImportError as exc:
            raise RuntimeError(
                "The Codex SDK is unavailable. Rebuild the App backend."
            ) from exc
        codex_bin, codex_environment, _version = _resolve_user_codex()
        self._approval_mode = ApprovalMode.deny_all
        self._sandbox = Sandbox.full_access
        self._reasoning_effort = ReasoningEffort
        self._client = AsyncCodex(
            CodexConfig(
                codex_bin=str(codex_bin),
                env=codex_environment,
            )
        )
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Close the SDK client and its user-installed local runtime."""

        await self._client.__aexit__(exc_type, exc, traceback)

    async def start_thread(
        self,
        *,
        cwd: Path,
        model: str | None,
        effort: str | None,
    ) -> Any:
        """Create one persistent full-access Codex thread."""

        kwargs: dict[str, Any] = {
            "approval_mode": self._approval_mode,
            "cwd": str(cwd),
            "sandbox": self._sandbox,
        }
        if model is not None:
            kwargs["model"] = model
        if effort is not None:
            kwargs["config"] = {"model_reasoning_effort": effort}
        return await self._client.thread_start(**kwargs)

    async def resume_thread(
        self,
        session_id: str,
        *,
        cwd: Path,
        model: str | None,
    ) -> Any:
        """Resume one persistent full-access Codex thread."""

        kwargs: dict[str, Any] = {
            "approval_mode": self._approval_mode,
            "cwd": str(cwd),
            "sandbox": self._sandbox,
        }
        if model is not None:
            kwargs["model"] = model
        return await self._client.thread_resume(session_id, **kwargs)

    async def run_turn(
        self,
        thread: Any,
        task: str,
        *,
        cwd: Path,
        model: str | None,
        effort: str | None,
        state: CodexToolState,
    ) -> str:
        """Run one interruptible turn with explicit persisted settings."""

        kwargs: dict[str, Any] = {
            "approval_mode": self._approval_mode,
            "cwd": str(cwd),
            "sandbox": self._sandbox,
        }
        if model is not None:
            kwargs["model"] = model
        if effort is not None:
            kwargs["effort"] = self._reasoning_effort(effort)
        handle = await thread.turn(task, **kwargs)
        state.active_handle = handle
        result = await handle.run()
        state.active_handle = None
        if result.error is not None:
            raise RuntimeError(str(result.error))
        if result.final_response is None:
            raise RuntimeError("Codex turn completed without a final response")
        return result.final_response

    async def resolve_sessions(
        self,
        *,
        prompt: str,
        cwd: Path,
        state: CodexToolState,
    ) -> dict[str, Any]:
        """Use an ephemeral SDK thread to select from supplied candidates."""

        thread = await self._client.thread_start(
            approval_mode=self._approval_mode,
            cwd=str(cwd),
            sandbox=self._sandbox,
            ephemeral=True,
            developer_instructions=SESSION_RESOLVER_INSTRUCTIONS,
        )
        handle = await thread.turn(
            prompt,
            approval_mode=self._approval_mode,
            cwd=str(cwd),
            sandbox=self._sandbox,
            output_schema=SESSION_RESOLVER_SCHEMA,
        )
        state.active_handle = handle
        result = await handle.run()
        state.active_handle = None
        if result.error is not None:
            raise RuntimeError(str(result.error))
        if result.final_response is None:
            raise RuntimeError("Session resolver returned no structured result")
        payload = json.loads(result.final_response)
        if not isinstance(payload, dict):
            raise ValueError("Session resolver result must be an object")
        return payload

    async def validate_model(
        self,
        *,
        model: str,
        effort: str | None,
    ) -> str:
        """Validate settings and return the canonical SDK model name."""

        response = await self._client.models(include_hidden=False)
        selected = next(
            (
                candidate
                for candidate in response.data
                if model in {candidate.id, candidate.model}
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"Codex model is not available: {model}")
        if effort is None:
            return selected.model
        supported = {
            option.reasoning_effort.value
            for option in selected.supported_reasoning_efforts
        }
        if effort not in supported:
            raise ValueError(
                f"Reasoning effort {effort!r} is not supported by {model}"
            )
        return selected.model

    async def list_models(self) -> list[CodexModelInfo]:
        """Return the current account's visible Codex model catalog."""

        response = await self._client.models(include_hidden=False)
        return [
            CodexModelInfo(
                id=candidate.id,
                model=candidate.model,
                display_name=candidate.display_name,
                description=candidate.description,
                is_default=candidate.is_default,
                default_reasoning_effort=(
                    candidate.default_reasoning_effort.value
                ),
                supported_reasoning_efforts=[
                    option.reasoning_effort.value
                    for option in candidate.supported_reasoning_efforts
                ],
            )
            for candidate in response.data
        ]

    async def archive_thread(self, session_id: str) -> None:
        """Archive one SDK thread."""

        await self._client.thread_archive(session_id)


class CodexCreateInput(ToolInput):
    """Input for creating and immediately running a Codex session."""

    task: str = Field(min_length=1, description="Concrete local task for Codex to execute.")
    cwd: str = Field(
        min_length=1,
        description="Actual local working directory; arbitrary paths are allowed.",
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        description="Codex model ID for future turns.",
    )
    effort: str | None = Field(
        default=None,
        min_length=1,
        description="Reasoning effort for future turns.",
    )


class CodexSearchInput(ToolInput):
    """Input for resolving a persistent session from natural language."""

    query: str = Field(
        min_length=1,
        description="Natural-language description of the target Codex session.",
    )


class CodexContinueInput(ToolInput):
    """Input for continuing an exact persistent Codex session."""

    session_id: str = Field(
        min_length=1,
        description="Exact persistent Codex session ID.",
    )
    task: str = Field(min_length=1, description="Concrete local task for Codex to execute.")
    cwd: str | None = Field(
        default=None,
        min_length=1,
        description="Actual local working directory; arbitrary paths are allowed.",
    )


class CodexModelsInput(ToolInput):
    """Empty input for querying the current Codex model catalog."""


class CodexSetModelInput(ToolInput):
    """Input for changing settings used by future session turns."""

    session_id: str = Field(
        min_length=1,
        description="Exact persistent Codex session ID.",
    )
    model: str = Field(
        min_length=1,
        description="Exact Codex model ID returned by codex_models_list.",
    )
    effort: str | None = Field(
        default=None,
        min_length=1,
        description="Reasoning effort for future turns.",
    )


class CodexDeleteInput(ToolInput):
    """Input for archiving an exact persistent Codex session."""

    session_id: str = Field(
        min_length=1,
        description="Exact persistent Codex session ID.",
    )


class CodexModelInfo(ToolOutput):
    """One currently visible model returned by the Codex SDK catalog."""

    id: str
    model: str
    display_name: str
    description: str
    is_default: bool
    default_reasoning_effort: str
    supported_reasoning_efforts: list[str] = Field(default_factory=list)


@dataclass
class CodexToolState(ToolState):
    """Mutable state for one asynchronous Codex tool call."""

    started_at: float = 0.0
    status_text: str = "Preparing Codex"
    stopped: bool = False
    active_handle: Any = None


class CodexOutput(ToolOutput):
    """Structured final output shared by the atomic Codex tool bundle."""

    action: str
    success: bool
    message: str
    session_id: str | None = None
    sessions: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    models: list[CodexModelInfo] = Field(default_factory=list)


_TOOL_DATA_DIRECTORY = Path(
    os.environ.get("XTALK_TOOL_DATA_DIR", Path.cwd() / ".xtalk-tool-data")
).expanduser().resolve()
_STORE = SessionStore(_TOOL_DATA_DIRECTORY / "codex_sessions.sqlite3")
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_adapter_factory = OpenAICodexAdapter


def _session_lock(session_id: str) -> asyncio.Lock:
    """Return the process-local serialization lock for one session."""

    lock = _SESSION_LOCKS.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _SESSION_LOCKS[session_id] = lock
    return lock


class _CodexTool(AsyncTool):
    """Shared interruptible lifecycle for the five Codex operations."""

    subscribe_by_default = True
    input_type = ToolInput
    state_type = CodexToolState
    output_type = CodexOutput

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: ToolInput,
        tool_state: CodexToolState,
        global_state: ToolEngineState,
    ) -> Running:
        """Start one asynchronous operation and return immediately."""

        del tool_input, global_state
        tool_state.started_at = time.monotonic()
        tool_state.status_text = cls.initial_status
        return Running(
            f"{cls.initial_status}. Tool call ID: {tool_call_id}. "
            "The final result will be delivered asynchronously."
        )

    @classmethod
    def emit_updates(
        cls,
        tool_input: ToolInput,
        tool_state: CodexToolState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[CodexOutput]]:
        """Yield no synchronous updates because Codex uses an async SDK."""

        del tool_input, tool_state, global_state
        return iter(())

    @classmethod
    async def aemit_updates(
        cls,
        tool_input: ToolInput,
        tool_state: CodexToolState,
        global_state: ToolEngineState,
    ) -> AsyncIterator[ToolResult[CodexOutput]]:
        """Run the concrete SDK operation and emit one final result."""

        del global_state
        try:
            output = await cls.execute(tool_input, tool_state)
        except asyncio.CancelledError:
            tool_state.stopped = True
            tool_state.status_text = "Codex stopped"
            raise
        except Exception as exc:
            tool_state.status_text = f"Codex failed ({type(exc).__name__})"
            output = CodexOutput(
                action=cls.name,
                success=False,
                message=str(exc),
            )
        if tool_state.stopped:
            return
        yield Finished(output)

    @classmethod
    async def execute(
        cls,
        tool_input: ToolInput,
        tool_state: CodexToolState,
    ) -> CodexOutput:
        """Execute the concrete operation implemented by a subclass."""

        raise NotImplementedError

    @classmethod
    def status(
        cls,
        tool_input: ToolInput,
        tool_state: CodexToolState,
        global_state: ToolEngineState,
    ) -> str:
        """Return current phase and elapsed time for read-only live UI."""

        del tool_input, global_state
        if tool_state.started_at <= 0:
            return tool_state.status_text
        elapsed = max(0, int(time.monotonic() - tool_state.started_at))
        return f"{tool_state.status_text} · {elapsed}s"

    @classmethod
    def stop(
        cls,
        tool_input: ToolInput,
        tool_state: CodexToolState,
        global_state: ToolEngineState,
    ) -> None:
        """Interrupt the active SDK turn when the XTalk call is stopped."""

        del tool_input, global_state
        tool_state.stopped = True
        tool_state.status_text = "Stopping Codex"

    @classmethod
    async def astop(
        cls,
        tool_input: ToolInput,
        tool_state: CodexToolState,
        global_state: ToolEngineState,
    ) -> None:
        """Interrupt the active SDK turn from the engine event loop."""

        cls.stop(tool_input, tool_state, global_state)
        if tool_state.active_handle is not None:
            await tool_state.active_handle.interrupt()


class CodexSessionCreateTool(_CodexTool):
    """Default tool for new requests that require accessing real local files or running local commands. Create a persistent full-access Codex session and execute the task. Do not use for conceptual questions or when the user refers to an existing Codex session."""

    name = "codex_session_create"
    initial_status = "Creating Codex session"
    input_type = CodexCreateInput

    @classmethod
    async def execute(
        cls,
        tool_input: CodexCreateInput,
        tool_state: CodexToolState,
    ) -> CodexOutput:
        """Create a persistent thread, run its first turn, and index it."""

        cwd = _resolve_working_directory(tool_input.cwd)
        model = tool_input.model
        async with _adapter_factory() as adapter:
            if model is not None:
                tool_state.status_text = "Validating Codex model"
                model = await adapter.validate_model(
                    model=model,
                    effort=tool_input.effort,
                )
            tool_state.status_text = "Creating Codex session"
            thread = await adapter.start_thread(
                cwd=cwd,
                model=model,
                effort=tool_input.effort,
            )
            session_id = thread.id
            _STORE.add(
                session_id=session_id,
                cwd=cwd,
                task=tool_input.task,
                model=model,
                effort=tool_input.effort,
            )
            tool_state.status_text = "Codex is working"
            try:
                response = await adapter.run_turn(
                    thread,
                    tool_input.task,
                    cwd=cwd,
                    model=model,
                    effort=tool_input.effort,
                    state=tool_state,
                )
            except Exception as exc:
                _STORE.mark_failed(session_id, str(exc))
                raise
        _STORE.mark_finished(session_id, response)
        tool_state.status_text = "Codex completed"
        return CodexOutput(
            action=cls.name,
            success=True,
            message=response,
            session_id=session_id,
            model=model,
            effort=tool_input.effort,
        )


class CodexSessionSearchTool(_CodexTool):
    """Find an existing Codex session from a natural-language description such as its project, topic, path, or time. Use before continuing, changing, or deleting a session when its exact ID is unknown."""

    name = "codex_session_search"
    initial_status = "Searching Codex sessions"
    input_type = CodexSearchInput

    @classmethod
    async def execute(
        cls,
        tool_input: CodexSearchInput,
        tool_state: CodexToolState,
    ) -> CodexOutput:
        """Resolve a natural-language query through an ephemeral thread."""

        candidates = _STORE.candidates(tool_input.query)
        if not candidates:
            tool_state.status_text = "No Codex sessions found"
            return CodexOutput(
                action=cls.name,
                success=True,
                message="No active Codex sessions are available.",
            )
        candidate_payload = [asdict(candidate) for candidate in candidates]
        prompt = json.dumps(
            {"query": tool_input.query, "candidates": candidate_payload},
            ensure_ascii=False,
        )
        tool_state.status_text = "Resolving target session"
        async with _adapter_factory() as adapter:
            result = await adapter.resolve_sessions(
                prompt=prompt,
                cwd=_TOOL_DATA_DIRECTORY,
                state=tool_state,
            )
        status = result.get("status")
        selected_ids = result.get("selected_session_ids")
        reason = result.get("reason")
        if (
            status not in {"matched", "ambiguous", "not_found"}
            or not isinstance(selected_ids, list)
            or not all(isinstance(value, str) for value in selected_ids)
            or not isinstance(reason, str)
        ):
            raise ValueError("Session resolver returned an invalid result")
        candidate_ids = {candidate.session_id for candidate in candidates}
        if not set(selected_ids).issubset(candidate_ids):
            raise ValueError("Session resolver selected an unknown session")
        if status == "matched" and len(selected_ids) != 1:
            raise ValueError("Matched session result must contain exactly one ID")
        if status == "ambiguous" and len(selected_ids) < 2:
            raise ValueError("Ambiguous session result must contain multiple IDs")
        if status == "not_found" and selected_ids:
            raise ValueError("Not-found session result must contain no IDs")
        selected = [
            asdict(_STORE.require(session_id)) for session_id in selected_ids
        ]
        tool_state.status_text = "Session search completed"
        return CodexOutput(
            action=cls.name,
            success=True,
            message=reason,
            session_id=(selected_ids[0] if status == "matched" else None),
            sessions=selected,
        )


class CodexSessionContinueTool(_CodexTool):
    """Continue a local file or command task in an existing Codex session. Requires an exact session ID supplied by the user or returned by codex_session_search. Use create instead for unrelated new work."""

    name = "codex_session_continue"
    initial_status = "Continuing Codex session"
    input_type = CodexContinueInput

    @classmethod
    async def execute(
        cls,
        tool_input: CodexContinueInput,
        tool_state: CodexToolState,
    ) -> CodexOutput:
        """Resume and run one serialized turn with persisted settings."""

        async with _session_lock(tool_input.session_id):
            record = _STORE.require(tool_input.session_id)
            cwd = _resolve_working_directory(tool_input.cwd or record.cwd)
            _STORE.mark_running(
                tool_input.session_id,
                task=tool_input.task,
                cwd=cwd,
            )
            tool_state.status_text = "Codex is working"
            try:
                async with _adapter_factory() as adapter:
                    thread = await adapter.resume_thread(
                        tool_input.session_id,
                        cwd=cwd,
                        model=record.model,
                    )
                    response = await adapter.run_turn(
                        thread,
                        tool_input.task,
                        cwd=cwd,
                        model=record.model,
                        effort=record.effort,
                        state=tool_state,
                    )
            except Exception as exc:
                _STORE.mark_failed(tool_input.session_id, str(exc))
                raise
            _STORE.mark_finished(tool_input.session_id, response)
        tool_state.status_text = "Codex completed"
        return CodexOutput(
            action=cls.name,
            success=True,
            message=response,
            session_id=tool_input.session_id,
            model=record.model,
            effort=record.effort,
        )


class CodexModelsListTool(_CodexTool):
    """Fetch the currently available Codex models and their supported reasoning efforts. Call this immediately before codex_session_set_model so model IDs are never guessed or taken from a stale list."""

    name = "codex_models_list"
    initial_status = "Loading Codex models"
    input_type = CodexModelsInput

    @classmethod
    async def execute(
        cls,
        tool_input: CodexModelsInput,
        tool_state: CodexToolState,
    ) -> CodexOutput:
        """Query the authenticated SDK model catalog without hidden entries."""

        del tool_input
        async with _adapter_factory() as adapter:
            models = await adapter.list_models()
        tool_state.status_text = "Codex models loaded"
        rows = [
            "| Model ID | Display name | Default effort | Supported efforts |",
            "|---|---|---|---|",
        ]
        rows.extend(
            "| "
            + " | ".join(
                (
                    _markdown_table_cell(model.id),
                    _markdown_table_cell(model.display_name),
                    _markdown_table_cell(model.default_reasoning_effort),
                    _markdown_table_cell(
                        ", ".join(model.supported_reasoning_efforts) or "—"
                    ),
                )
            )
            + " |"
            for model in models
        )
        message = (
            "No visible Codex models are available for the current account."
            if not models
            else "Current Codex models:\n\n" + "\n".join(rows)
        )
        return CodexOutput(
            action=cls.name,
            success=True,
            message=message,
            models=models,
        )


class CodexSessionSetModelTool(_CodexTool):
    """Change the model and reasoning effort used by future turns in an existing Codex session. Call codex_models_list immediately before this tool and use an exact returned model ID; never guess or rely on a stale model list. Use only when the user explicitly requests a model or effort change."""

    name = "codex_session_set_model"
    initial_status = "Updating Codex model"
    input_type = CodexSetModelInput

    @classmethod
    async def execute(
        cls,
        tool_input: CodexSetModelInput,
        tool_state: CodexToolState,
    ) -> CodexOutput:
        """Validate and persist settings for future turns on one thread."""

        async with _session_lock(tool_input.session_id):
            current = _STORE.require(tool_input.session_id)
            effective_effort = tool_input.effort or current.effort
            tool_state.status_text = "Validating Codex model"
            async with _adapter_factory() as adapter:
                model = await adapter.validate_model(
                    model=tool_input.model,
                    effort=effective_effort,
                )
            record = _STORE.set_model(
                tool_input.session_id,
                model=model,
                effort=tool_input.effort,
            )
        tool_state.status_text = "Codex model updated"
        effort_text = f" ({record.effort})" if record.effort else ""
        return CodexOutput(
            action=cls.name,
            success=True,
            message=(
                f"Future turns in {record.session_id} will use "
                f"{record.model}{effort_text}."
            ),
            session_id=record.session_id,
            model=record.model,
            effort=record.effort,
        )


class CodexSessionDeleteTool(_CodexTool):
    """Remove an exact Codex session from the active pool and archive its thread. Use only when the user explicitly requests session deletion. This does not delete project files."""

    name = "codex_session_delete"
    initial_status = "Archiving Codex session"
    input_type = CodexDeleteInput

    @classmethod
    async def execute(
        cls,
        tool_input: CodexDeleteInput,
        tool_state: CodexToolState,
    ) -> CodexOutput:
        """Archive the SDK thread before removing it from the active pool."""

        async with _session_lock(tool_input.session_id):
            _STORE.require(tool_input.session_id)
            async with _adapter_factory() as adapter:
                await adapter.archive_thread(tool_input.session_id)
            _STORE.archive(tool_input.session_id)
        tool_state.status_text = "Codex session archived"
        return CodexOutput(
            action=cls.name,
            success=True,
            message=f"Archived Codex session {tool_input.session_id}.",
            session_id=tool_input.session_id,
        )


def create_tools() -> list[type[AsyncTool]]:
    """Return the indivisible built-in Codex tool bundle.

    Returns
    -------
    list[type[AsyncTool]]
        All session operations controlled by the single built-in toggle.
    """

    return [
        CodexSessionSearchTool,
        CodexSessionCreateTool,
        CodexSessionContinueTool,
        CodexModelsListTool,
        CodexSessionSetModelTool,
        CodexSessionDeleteTool,
    ]
