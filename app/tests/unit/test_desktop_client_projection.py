"""Static regression tests for the desktop SDK message mapping."""

from pathlib import Path


def _adapter_logic() -> str:
    """Read the trusted desktop adapter source."""

    return (
        Path(__file__).parents[2]
        / "ui"
        / "adapters"
        / "xtalk-client-adapter.ts"
    ).read_text(encoding="utf-8")


def test_desktop_maps_one_display_entry_per_sdk_message() -> None:
    """Render the SDK message array verbatim like the sample application."""

    logic = _adapter_logic()
    mapping = logic.split(
        "function mapDesktopMessages(", maxsplit=1
    )[1].split("export class XtalkClientAdapter", maxsplit=1)[0]

    assert "return messages.map((message, index) =>" in mapping
    assert 'id: `${sessionId ?? "pending"}:${index}`' in mapping
    assert "role: message.role" in mapping
    assert "content: message.content" in mapping
    assert "final: message.final === true" in mapping


def test_desktop_does_not_coalesce_assistant_messages() -> None:
    """Do not merge consecutive assistant entries across turns."""

    logic = _adapter_logic()
    mapping = logic.split(
        "function mapDesktopMessages(", maxsplit=1
    )[1].split("export class XtalkClientAdapter", maxsplit=1)[0]

    assert 'previous?.role === "assistant"' not in mapping
    assert "message.content.length >= previous.content.length" not in mapping
    assert 'streamState === "idle"' not in mapping


def test_desktop_forces_microphone_echo_cancellation() -> None:
    """Prevent speaker output from being recognized as another user turn."""

    logic = _adapter_logic()
    constraints = logic.split(
        "function installDesktopAudioConstraints", maxsplit=1
    )[1].split("export class XtalkClientAdapter", maxsplit=1)[0]

    assert "echoCancellation: true" in constraints
    assert "noiseSuppression: true" in constraints
