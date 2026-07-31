"""Contract tests for the atomic built-in Codex tool bundle."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from xtalk.models.agents.tools import Finished, Running

from backend.tool_registry import load_enabled_tools


class _FakeThread:
    """Minimal official-SDK thread shape used by the adapter fake."""

    def __init__(self, session_id: str) -> None:
        """Record the public thread ID."""

        self.id = session_id


class _FakeAdapter:
    """Deterministic async adapter that never starts the real Codex runtime."""

    operations: list[tuple[str, Any]] = []
    resolver_result: dict[str, Any] = {
        "status": "matched",
        "selected_session_ids": ["thr-test"],
        "reason": "The project path and task match.",
    }

    async def __aenter__(self) -> _FakeAdapter:
        """Return this fake client."""

        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Keep deterministic fake state after the operation."""

    async def validate_model(self, *, model: str, effort: str | None) -> str:
        """Record model validation."""

        self.operations.append(("validate", (model, effort)))
        return model

    async def start_thread(
        self,
        *,
        cwd: Path,
        model: str | None,
        effort: str | None,
    ) -> _FakeThread:
        """Return one persistent fake thread."""

        self.operations.append(("start", (str(cwd), model, effort)))
        return _FakeThread("thr-test")

    async def resume_thread(
        self,
        session_id: str,
        *,
        cwd: Path,
        model: str | None,
    ) -> _FakeThread:
        """Return the requested fake thread."""

        self.operations.append(("resume", (session_id, str(cwd), model)))
        return _FakeThread(session_id)

    async def run_turn(
        self,
        thread: _FakeThread,
        task: str,
        *,
        cwd: Path,
        model: str | None,
        effort: str | None,
        state: Any,
    ) -> str:
        """Return a deterministic response and record explicit settings."""

        del state
        self.operations.append(
            ("turn", (thread.id, task, str(cwd), model, effort))
        )
        return f"Completed: {task}"

    async def resolve_sessions(
        self,
        *,
        prompt: str,
        cwd: Path,
        state: Any,
    ) -> dict[str, Any]:
        """Return the configured structured resolver output."""

        del state
        self.operations.append(("resolve", (json.loads(prompt), str(cwd))))
        return self.resolver_result

    async def archive_thread(self, session_id: str) -> None:
        """Record SDK archival."""

        self.operations.append(("archive", session_id))


def _load_codex_bundle(tmp_path: Path) -> tuple[list[type[Any]], Any]:
    """Enable and load the repository Codex built-in into isolated AppData."""

    app_root = Path(__file__).resolve().parents[2]
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    (data_root / "tool_preferences.json").write_text(
        json.dumps(
            {
                "version": 1,
                "builtin": {
                    "timer": {"enabled": False},
                    "codex": {"enabled": True},
                },
            }
        ),
        encoding="utf-8",
    )
    tools = load_enabled_tools(
        data_root / "tools",
        builtin_tools_root=app_root / "resources" / "tools",
    )
    module = sys.modules[tools[0].__module__]
    module._adapter_factory = _FakeAdapter
    _FakeAdapter.operations = []
    return tools, module


async def _invoke(tool: type[Any], tool_input: Any, module: Any) -> Any:
    """Run one native asynchronous tool lifecycle to completion."""

    state = module.CodexToolState()
    initial = tool.emit_initial("codex-call", tool_input, state, object())
    assert isinstance(initial, Running)
    updates = [
        update
        async for update in tool.aemit_updates(tool_input, state, object())
    ]
    assert len(updates) == 1
    assert isinstance(updates[0], Finished)
    return updates[0].content


def test_codex_catalog_toggle_loads_all_operations_atomically(
    tmp_path: Path,
) -> None:
    """Expose five operations from one disabled-by-default built-in entry."""

    tools, module = _load_codex_bundle(tmp_path)

    assert [tool.name for tool in tools] == [
        "codex_session_search",
        "codex_session_create",
        "codex_session_continue",
        "codex_session_set_model",
        "codex_session_delete",
    ]
    assert "Default tool for new requests" in tools[1].__doc__
    assert "exact session ID" in tools[2].__doc__
    assert module._STORE.path == (
        tmp_path / "data" / "tool-data" / "codex" / "codex_sessions.sqlite3"
    )


def test_codex_session_lifecycle_persists_settings_and_uses_resolver(
    tmp_path: Path,
) -> None:
    """Create, continue, search, reconfigure, and archive one SDK thread."""

    tools, module = _load_codex_bundle(tmp_path)
    by_name = {tool.name: tool for tool in tools}

    async def scenario() -> None:
        create = await _invoke(
            by_name["codex_session_create"],
            module.CodexCreateInput(
                task="Update the local README",
                cwd=str(tmp_path),
                model="gpt-5.6-sol",
                effort="high",
            ),
            module,
        )
        assert create.success is True
        assert create.session_id == "thr-test"

        configured = await _invoke(
            by_name["codex_session_set_model"],
            module.CodexSetModelInput(
                session_id="thr-test",
                model="gpt-5.6-sol",
                effort="high",
            ),
            module,
        )
        assert configured.model == "gpt-5.6-sol"
        assert configured.effort == "high"

        continued = await _invoke(
            by_name["codex_session_continue"],
            module.CodexContinueInput(
                session_id="thr-test",
                task="Now run the documentation checks",
            ),
            module,
        )
        assert continued.message == "Completed: Now run the documentation checks"

        found = await _invoke(
            by_name["codex_session_search"],
            module.CodexSearchInput(query="README documentation project"),
            module,
        )
        assert found.session_id == "thr-test"
        assert found.sessions[0]["model"] == "gpt-5.6-sol"

        deleted = await _invoke(
            by_name["codex_session_delete"],
            module.CodexDeleteInput(session_id="thr-test"),
            module,
        )
        assert deleted.success is True

    asyncio.run(scenario())

    turn_operations = [
        payload for operation, payload in _FakeAdapter.operations if operation == "turn"
    ]
    assert turn_operations[-1][-2:] == ("gpt-5.6-sol", "high")
    resolver_payload = next(
        payload
        for operation, payload in _FakeAdapter.operations
        if operation == "resolve"
    )
    assert resolver_payload[0]["query"] == "README documentation project"
    assert resolver_payload[0]["candidates"][0]["session_id"] == "thr-test"
    assert ("archive", "thr-test") in _FakeAdapter.operations
    with pytest.raises(ValueError, match="not found"):
        module._STORE.require("thr-test")


def test_codex_search_rejects_session_ids_outside_candidate_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent an ephemeral resolver from inventing or escaping pool IDs."""

    tools, module = _load_codex_bundle(tmp_path)
    by_name = {tool.name: tool for tool in tools}
    monkeypatch.setattr(_FakeAdapter, "resolver_result", {
        "status": "matched",
        "selected_session_ids": ["thr-unknown"],
        "reason": "Invented result",
    })

    async def scenario() -> Any:
        await _invoke(
            by_name["codex_session_create"],
            module.CodexCreateInput(task="Inspect files", cwd=str(tmp_path)),
            module,
        )
        return await _invoke(
            by_name["codex_session_search"],
            module.CodexSearchInput(query="Inspect files"),
            module,
        )

    output = asyncio.run(scenario())

    assert output.success is False
    assert "unknown session" in output.message


def test_codex_stop_interrupts_the_active_sdk_turn(tmp_path: Path) -> None:
    """Forward XTalk cancellation to the SDK turn handle."""

    tools, module = _load_codex_bundle(tmp_path)
    tool = next(tool for tool in tools if tool.name == "codex_session_create")

    class _Handle:
        """Record whether the asynchronous interrupt hook ran."""

        interrupted = False

        async def interrupt(self) -> None:
            """Record one SDK interruption."""

            self.interrupted = True

    handle = _Handle()
    state = module.CodexToolState(active_handle=handle)
    tool_input = module.CodexCreateInput(task="Inspect files", cwd=str(tmp_path))

    asyncio.run(tool.astop(tool_input, state, object()))

    assert state.stopped is True
    assert handle.interrupted is True
