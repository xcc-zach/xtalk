"""Static regression tests for desktop voice-orb state colors."""

from pathlib import Path


def test_orb_stream_states_have_distinct_semantic_colors() -> None:
    """Map user speech, processing, and model speech to green, yellow, and blue."""

    app_root = Path(__file__).parents[2]
    styles = (app_root / "ui" / "styles.css").read_text(encoding="utf-8")

    expected_colors = {
        "listening": "#2fa66e",
        "processing": "#e0a52b",
        "speaking": "#3f73e8",
    }
    for state, color in expected_colors.items():
        selector = (
            f'.orb-button[data-stream-state="{state}"],\n'
            f'.chat-mini-orb[data-stream-state="{state}"]'
        )
        rule = styles.split(selector, maxsplit=1)[1].split("}", maxsplit=1)[0]
        assert f"--orb-gradient-start: {color}" in rule


def test_idle_orb_keeps_the_existing_default_palette() -> None:
    """Leave idle without a state override so it retains the lavender palette."""

    app_root = Path(__file__).parents[2]
    styles = (app_root / "ui" / "styles.css").read_text(encoding="utf-8")
    logic = (app_root / "ui" / "main.ts").read_text(encoding="utf-8")

    assert "--orb-gradient-start: var(--orb-blue)" in styles
    assert "--orb-gradient-middle: var(--orb-lavender)" in styles
    assert '[data-stream-state="idle"]' not in styles
    assert "orb.dataset.streamState = snapshot.streamState" in logic
