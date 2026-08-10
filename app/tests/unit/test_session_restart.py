"""Static regression tests for conversation recovery after sidecar restarts."""

from pathlib import Path


def test_backend_restart_restores_the_active_conversation() -> None:
    """Reload the selected persisted session after replacing the adapter."""

    logic = (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")

    discovery = logic.split(
        "async function discoverBackend(", maxsplit=1
    )[1].split("async function applyModelConfigPath", maxsplit=1)[0]
    assert "sessionIdToRestore?: string | null" in discovery
    assert "latestSnapshot.sessionId ?? readActiveSessionId()" in discovery
    assert "const sessions = await nextAdapter.getSessions()" in discovery
    assert "restoredSessionId = sessions[0]?.id ?? null" in discovery
    assert "await nextAdapter.switchSession(restoredSessionId)" in discovery
    assert "sessionRestoreError" in discovery
    assert 'showError("sidebar.switchFailed"' in discovery


def test_all_settings_restarts_forward_the_active_conversation() -> None:
    """Preserve the session through both success and rollback reconnections."""

    logic = (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")

    model_restart = logic.split(
        "async function applyModelConfigPath", maxsplit=1
    )[1].split("async function chooseAndApplyModelConfig", maxsplit=1)[0]
    tool_restart = logic.split(
        "async function applyToolChanges", maxsplit=1
    )[1].split("async function initializeApplication", maxsplit=1)[0]

    for restart in (model_restart, tool_restart):
        assert "const sessionIdToRestore = latestSnapshot.sessionId" in restart
        assert restart.count("await discoverBackend(sessionIdToRestore)") == 2


def test_full_app_restart_uses_a_port_independent_active_session_key() -> None:
    """Do not key the selected conversation by the sidecar's random port."""

    logic = (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")

    assert (
        'const ACTIVE_SESSION_STORAGE_KEY = "xtalk.desktop.active-session.v1";'
        in logic
    )
    assert "function readActiveSessionId(): string | null" in logic
    assert "function persistActiveSessionId(sessionId: string | null): void" in logic
    assert "persistActiveSessionId(snapshot.sessionId)" in logic
