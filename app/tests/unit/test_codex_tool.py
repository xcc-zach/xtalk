"""Contract tests for the atomic built-in Codex tool bundle."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
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
    model_catalog: list[Any] = []

    async def __aenter__(self) -> _FakeAdapter:
        """Return this fake client."""

        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Keep deterministic fake state after the operation."""

    async def validate_model(self, *, model: str, effort: str | None) -> str:
        """Record model validation."""

        self.operations.append(("validate", (model, effort)))
        return model

    async def list_models(self) -> list[Any]:
        """Return the current deterministic model catalog."""

        self.operations.append(("models", None))
        return list(self.model_catalog)

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
    codex_tool = next(
        tool for tool in tools if tool.name == "codex_session_search"
    )
    module = sys.modules[codex_tool.__module__]
    tools = [tool for tool in tools if tool.__module__ == module.__name__]
    module._adapter_factory = _FakeAdapter
    module._CODEX_CACHE = None
    _FakeAdapter.operations = []
    _FakeAdapter.model_catalog = [
        module.CodexModelInfo(
            id="gpt-5.6-sol",
            model="gpt-5.6-sol",
            display_name="GPT-5.6 Sol",
            description="Frontier coding model.",
            is_default=True,
            default_reasoning_effort="high",
            supported_reasoning_efforts=["medium", "high", "xhigh"],
        )
    ]
    return tools, module


def test_codex_resolves_and_caches_user_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass a validated external CLI and its Node.js directory to the SDK."""

    _tools, module = _load_codex_bundle(tmp_path)
    executable = tmp_path / "node-bin" / "codex"
    executable.parent.mkdir()
    executable.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(
        module,
        "_candidate_codex_paths",
        lambda: iter((executable,)),
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run_version(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="codex-cli 0.144.4\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run_version)

    first = module._resolve_user_codex()
    second = module._resolve_user_codex()
    path, environment, version = first

    assert second == first
    assert path == executable
    assert version == "codex-cli 0.144.4"
    assert environment["PATH"].split(os.pathsep)[0] == str(executable.parent)
    assert calls[0][0] == [str(executable), "--version"]
    assert len(calls) == 1


def test_codex_replaces_cache_after_executable_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search again only after the cached executable path becomes invalid."""

    _tools, module = _load_codex_bundle(tmp_path)
    first = tmp_path / "node-v1" / "codex"
    second = tmp_path / "node-v2" / "codex"
    for executable in (first, second):
        executable.parent.mkdir()
        executable.write_text("#!/usr/bin/env node\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setattr(
        module,
        "_candidate_codex_paths",
        lambda: iter((first, second)),
    )
    calls: list[list[str]] = []

    def run_version(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"codex-cli {Path(command[0]).parent.name}\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run_version)

    assert module._resolve_user_codex()[0] == first
    first.unlink()
    assert module._resolve_user_codex()[0] == second
    assert len(calls) == 2


def test_codex_prompts_for_install_when_user_cli_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable install command instead of using a bundled CLI."""

    _tools, module = _load_codex_bundle(tmp_path)
    monkeypatch.setattr(
        module,
        "_candidate_codex_paths",
        lambda: iter((tmp_path / "missing-codex",)),
    )

    with pytest.raises(RuntimeError, match="npm install -g @openai/codex"):
        module._resolve_user_codex()


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
    """Expose six operations from one disabled-by-default built-in entry."""

    tools, module = _load_codex_bundle(tmp_path)

    assert [tool.name for tool in tools] == [
        "codex_session_search",
        "codex_session_create",
        "codex_session_continue",
        "codex_models_list",
        "codex_session_set_model",
        "codex_session_delete",
    ]
    assert "Default tool for new requests" in tools[1].__doc__
    assert "exact session ID" in tools[2].__doc__
    assert "never guessed" in tools[3].__doc__
    assert "codex_models_list" in tools[4].__doc__
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
        catalog = await _invoke(
            by_name["codex_models_list"],
            module.CodexModelsInput(),
            module,
        )
        assert catalog.models[0].id == "gpt-5.6-sol"
        assert catalog.models[0].supported_reasoning_efforts == [
            "medium",
            "high",
            "xhigh",
        ]
        assert "| gpt-5.6-sol | GPT-5.6 Sol | high |" in catalog.message

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
    assert ("models", None) in _FakeAdapter.operations
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
