"""Static regression tests for desktop conversation switching."""

from pathlib import Path


def _adapter_logic() -> str:
    """Read the trusted desktop adapter source."""

    return (
        Path(__file__).parents[2]
        / "ui"
        / "adapters"
        / "xtalk-client-adapter.ts"
    ).read_text(encoding="utf-8")


def test_switch_session_leaves_the_realtime_session_closed() -> None:
    """Stay stopped after switching until the user presses the start button."""

    logic = _adapter_logic()
    switch = logic.split(
        "async switchSession(sessionId: string | null): Promise<void> {",
        maxsplit=1,
    )[1].split("#notify(): void {", maxsplit=1)[0]

    assert "await this.#session.switchSession(sessionId);" in switch
    assert "await this.#session.close();" in switch
    assert "await this.#session.open();" not in switch
