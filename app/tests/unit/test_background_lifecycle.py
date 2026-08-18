"""Static regressions for acknowledged desktop backgrounding."""

from pathlib import Path


APP_ROOT = Path(__file__).parents[2]


def test_native_background_request_does_not_hide_the_window() -> None:
    """Keep the close request separate from the final native hide command."""

    source = (APP_ROOT / "src-tauri" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    request = source.split(
        "fn request_main_window_backgrounding", maxsplit=1
    )[1].split("\n}\n\nfn hide_main_window", maxsplit=1)[0]
    hide = source.split("fn hide_main_window", maxsplit=1)[1].split(
        "\n}\n\nfn handle_window_event", maxsplit=1
    )[0]
    command = source.split("async fn background_main_window", maxsplit=1)[1].split(
        "\n}\n\nfn request_main_window_backgrounding", maxsplit=1
    )[0]

    assert 'app.emit("app-backgrounding", ())' in request
    assert ".hide()" not in request
    assert ".hide()" in hide
    assert "WakeWordState::Listening" in command
    hide_index = command.index("hide_main_window(&app)?")
    second_state_check = command.index("wake_word.settings().await", hide_index)
    restore_index = command.index("tray::show_main_window(&app)", second_state_check)

    assert hide_index < second_state_check < restore_index


def test_webview_prepares_background_state_before_hiding() -> None:
    """Close the Session and restore wake listening before hiding the window."""

    source = (APP_ROOT / "ui" / "main.ts").read_text(encoding="utf-8")
    flow = source.split("async function enterSleepMode", maxsplit=1)[1].split(
        "\n}\n\nasync function sendTextMessage", maxsplit=1
    )[0]
    disconnect_index = flow.index("await activeAdapter.disconnect()")
    resume_index = flow.index("await resumeNativeWakeWord()")
    listening_index = flow.index('settings.state !== "listening"')
    hide_index = flow.index("await backgroundNativeMainWindow()")

    assert disconnect_index < resume_index < listening_index < hide_index
    assert "backgroundingRequested = false" in flow
    listener = source.split(
        "await listenNativeAppBackgrounding", maxsplit=1
    )[1].split("await refreshWakeWordSettings()", maxsplit=1)[0]
    assert "void enterSleepMode()" in listener
    assert "void disconnectSession()" not in listener


def test_wake_word_failure_restores_the_main_window() -> None:
    """Keep a background App reachable when its detector exits unexpectedly."""

    source = (APP_ROOT / "src-tauri" / "src" / "wake_word.rs").read_text(
        encoding="utf-8"
    )
    error_handler = source.split("async fn set_error", maxsplit=1)[1].split(
        "\n    }\n}", maxsplit=1
    )[0]

    assert "WakeWordState::Error" in error_handler
    assert "self.emit_status(app).await" in error_handler
    assert "crate::tray::show_main_window(app)" in error_handler


def test_whiteboard_creation_commands_are_async() -> None:
    """Keep Windows webview creation away from synchronous Tauri commands."""

    source = (APP_ROOT / "src-tauri" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )

    assert "async fn show_whiteboard_window(" in source
    assert "async fn set_whiteboard_window_visible(" in source
