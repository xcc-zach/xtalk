"""Tests for desktop-only conversation completion semantics."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from backend.desktop_service import (
    DesktopService,
    DesktopTTSPlaybackManager,
)
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import TTSPlaybackFinished
from xtalk.serving.modules.tts_playback_manager import TTSPlaybackManager
from xtalk.serving.service import DefaultService


def _manager() -> DesktopTTSPlaybackManager:
    """Create a desktop playback manager with no external models."""

    return DesktopTTSPlaybackManager(
        event_bus=EventBus(),
        session_id="desktop-test-session",
        config={},
    )


def _capture_commits(
    manager: DesktopTTSPlaybackManager,
) -> tuple[list[str], Callable[[str], Awaitable[None]]]:
    """Replace response publication with an in-memory recorder.

    Parameters
    ----------
    manager : DesktopTTSPlaybackManager
        Manager whose commit method is replaced by the caller.

    Returns
    -------
    tuple[list[str], Callable[[str], Awaitable[None]]]
        Mutable commit log and compatible asynchronous recorder.
    """

    committed: list[str] = []

    async def record(text: str) -> None:
        """Record one response text without publishing an event."""

        committed.append(text)

    manager._commit_playback_text = record  # type: ignore[method-assign]
    return committed, record


def test_normal_completion_commits_full_generated_text() -> None:
    """Promote the complete LLM turn after all desktop audio has played."""

    manager = _manager()
    manager._received_audio = True
    manager._reported_text = "正在为您查 今天是2026年8月4日，国内方"
    manager._pending_text = (
        "正在为您查。今天是2026年8月4日，主要有这些新闻。"
        "国际方面有新的进展，国内方面也有多项更新。"
    )
    committed, _ = _capture_commits(manager)

    asyncio.run(
        manager._publish_response_finish(
            TTSPlaybackFinished(session_id=manager.session_id)
        )
    )

    assert committed == [
        "正在为您查。今天是2026年8月4日，主要有这些新闻。"
        "国际方面有新的进展，国内方面也有多项更新。"
    ]


def test_interruption_keeps_playback_confirmed_prefix() -> None:
    """Keep inherited stop handling from exposing unheard generated text."""

    manager = _manager()
    manager._reported_text = "已经播放的前缀"
    manager._pending_text = "已经播放的前缀以及没有播放的后半段"
    committed, _ = _capture_commits(manager)

    asyncio.run(manager._commit_stopped_playback())

    assert committed == ["已经播放的前缀"]


def test_desktop_service_replaces_only_playback_manager() -> None:
    """Retain the default manager stack with one desktop specialization."""

    assert DesktopTTSPlaybackManager in DesktopService.MANAGER_CLASSES
    assert TTSPlaybackManager not in DesktopService.MANAGER_CLASSES
    assert len(DesktopService.MANAGER_CLASSES) == len(
        DefaultService.MANAGER_CLASSES
    )


def test_desktop_playback_finish_has_one_inherited_subscription() -> None:
    """Avoid publishing the final response twice through duplicate handlers."""

    event_bus = EventBus()
    DesktopTTSPlaybackManager(
        event_bus=event_bus,
        session_id="desktop-test-session",
        config={},
    )

    handlers: list[Any] = event_bus._handlers[TTSPlaybackFinished.TYPE]
    assert len(handlers) == 1
