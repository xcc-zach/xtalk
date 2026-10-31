"""Provider-neutral contracts for continuous speech sessions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..registry import model_type


@dataclass(frozen=True)
class AudioProfile:
    """Describe the raw audio shared by capture and playback.

    Parameters
    ----------
    encoding : str
        PCM encoding used on the application transport.
    sample_rate : int
        Samples per second in each direction.
    channels : int
        Number of interleaved channels.
    """

    encoding: str = "pcm_s16le"
    sample_rate: int = 24000
    channels: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible media description."""
        return asdict(self)


@dataclass(frozen=True)
class EndToEndModelEvent:
    """Carry a normalized provider event without serving dependencies.

    Parameters
    ----------
    kind : str
        Normalized audio, transcript, control, backend, usage, error, or closed kind.
    data : dict
        Validated event fields, including upstream correlation identifiers.
    audio : bytes
        Decoded PCM bytes for audio events.
    """

    kind: Literal[
        "audio",
        "transcript",
        "control_ack",
        "backend_activity",
        "usage",
        "error",
        "closed",
    ]
    data: dict[str, Any] = field(default_factory=dict)
    audio: bytes = b""


@dataclass(frozen=True)
class EndToEndCloseResult:
    """Report finalization independently of application playback.

    Parameters
    ----------
    finalization : str
        Complete, incomplete, or not_started session finalization.
    reason : str
        Sanitized terminal reason.
    voice_usage : dict
        Latest cumulative voice usage snapshot.
    backend_usage : list
        Separately identified delegated backend usage records.
    """

    finalization: str = "not_started"
    reason: str = ""
    voice_usage: dict[str, Any] = field(default_factory=dict)
    backend_usage: list[dict[str, Any]] = field(default_factory=list)


@model_type
class EndToEnd(ABC):
    """Adapt one independently runnable continuous speech session."""

    @property
    def audio_profile(self) -> AudioProfile:
        """Return the immutable input/output media profile."""
        return AudioProfile()

    @property
    def metadata(self) -> dict[str, str]:
        """Return secret-free provider information for display."""
        return {"provider": type(self).__name__}

    @abstractmethod
    def clone(self) -> EndToEnd:
        """Create a separate unopened runtime with the same configuration."""

    @abstractmethod
    async def open(self) -> None:
        """Connect and return after the provider confirms startup."""

    @abstractmethod
    async def append_audio(self, pcm: bytes) -> None:
        """Admit ordered PCM into the bounded upstream writer."""

    @abstractmethod
    def events(self) -> AsyncIterator[EndToEndModelEvent]:
        """Return the session's single normalized event consumer."""

    @abstractmethod
    async def set_input_muted(self, muted: bool) -> None:
        """Enqueue input control and report application asynchronously."""

    @abstractmethod
    async def close(self) -> EndToEndCloseResult:
        """Release resources after bounded graceful finalization."""
