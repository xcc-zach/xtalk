"""Static regression tests for the desktop message composer."""

from pathlib import Path


def test_ime_confirmation_does_not_consume_the_next_enter() -> None:
    """Track the held confirmation key instead of guarding by elapsed time."""

    app_root = Path(__file__).parents[2]
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    assert "messageInputCompositionEnterHeld" in logic
    assert 'addEventListener("keyup"' in logic
    assert "messageInputCompositionEnterHeld = false" in logic
    assert "IME_COMPOSITION_COMMIT_GUARD_MS" not in logic
    assert "messageInputCompositionCommitPending" not in logic


def test_only_unmodified_enter_sends_a_message() -> None:
    """Reserve Shift+Enter for native textarea line insertion."""

    app_root = Path(__file__).parents[2]
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    assert 'event.key === "Enter" && !event.shiftKey' in logic
    enter_branch = logic.split(
        'if (event.key === "Enter" && !event.shiftKey) {',
        maxsplit=1,
    )[1].split("}", maxsplit=1)[0]
    assert "event.preventDefault()" in enter_branch
    assert "sendTextMessage()" in enter_branch
