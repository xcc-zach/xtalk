"""Static regression tests for lazy desktop credential loading."""

from pathlib import Path


def _main_logic() -> str:
    """Read the trusted desktop UI source."""

    return (
        Path(__file__).parents[2] / "ui" / "main.ts"
    ).read_text(encoding="utf-8")


def test_credentials_refresh_lazily_when_diagnostics_open() -> None:
    """Load credential status only when the settings drawer is opened."""

    logic = _main_logic()
    diagnostics = logic.split(
        "function setDiagnosticsOpen(open: boolean): void {",
        maxsplit=1,
    )[1].split("function setToolsDialogOpen", maxsplit=1)[0]

    assert "void refreshCredentials().catch(() => undefined);" in diagnostics


def test_startup_does_not_refresh_credentials() -> None:
    """Avoid a keychain read on launch; open settings to load credentials."""

    logic = _main_logic()
    init = logic.split(
        "async function initializeApplication(): Promise<void> {",
        maxsplit=1,
    )[1].split("async function ensureNativeBackendStarted", maxsplit=1)[0]

    assert "await refreshCredentials();" not in init
