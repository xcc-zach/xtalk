"""Shared configuration helpers for speaker-aware ASR history filtering."""

from __future__ import annotations

from typing import Any, Mapping

from ..models import Models, SpeakerDiarization


def exclude_non_focus_from_history(config: Mapping[str, Any] | None) -> bool:
    """Return whether non-focus speech should be excluded from history.

    Parameters
    ----------
    config : Mapping[str, Any] | None
        Service configuration containing the optional ``multi_speaker`` block.

    Returns
    -------
    bool
        Configured value, defaulting to ``True``.

    Raises
    ------
    ValueError
        Raised when the configured value is not a boolean.
    """

    multi_config = dict((config or {}).get("multi_speaker") or {})
    value = multi_config.get("exclude_non_focus_from_history", True)
    if not isinstance(value, bool):
        raise ValueError(
            "multi_speaker.exclude_non_focus_from_history must be a boolean"
        )
    return value


def speaker_history_gate_enabled(
    models: Models,
    config: Mapping[str, Any] | None,
) -> bool:
    """Return whether the speaker-aware ASR history gate is effective.

    The gate is active only for ``focus_only`` sessions that have an actual
    speaker-diarization model and explicitly or implicitly enable history
    exclusion. Sessions without a diarization model preserve the existing ASR
    event flow.

    Parameters
    ----------
    models : Models
        Session model container.
    config : Mapping[str, Any] | None
        Service configuration containing the optional ``multi_speaker`` block.

    Returns
    -------
    bool
        Whether ASR partial/final events require stoppable dispatch and
        speaker-aware filtering.
    """

    multi_config = dict((config or {}).get("multi_speaker") or {})
    response_policy = str(multi_config.get("response_policy", "focus_only"))
    return (
        models.get(SpeakerDiarization) is not None
        and response_policy == "focus_only"
        and exclude_non_focus_from_history(config)
    )


__all__ = [
    "exclude_non_focus_from_history",
    "speaker_history_gate_enabled",
]
