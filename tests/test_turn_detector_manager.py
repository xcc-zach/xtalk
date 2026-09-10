"""Tests for turn-detector event routing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from xtalk.models.turn_detector.interfaces import (
    TurnDetectionAction,
    TurnDetectionResult,
    TurnDetectionSemantic,
)
from xtalk.serving.events import TTSPlaybackFinished, TTSStopped
from xtalk.serving.modules.turn_detector_manager import TurnDetectorManager


class TurnDetectorManagerPlaybackTests(unittest.IsolatedAsyncioTestCase):
    """Verify playback events forwarded to the turn detector."""

    async def test_only_early_stop_reports_assistant_interruption(self) -> None:
        """Distinguish interrupted playback from natural completion."""
        idle_result = TurnDetectionResult(
            action=TurnDetectionAction.DO_NOTHING,
            semantic=TurnDetectionSemantic.IDLE,
        )
        async_detect = AsyncMock(return_value=idle_result)
        detector = SimpleNamespace(
            listening=False,
            async_detect=async_detect,
        )
        manager = object.__new__(TurnDetectorManager)
        manager.turn_detector = detector
        manager._handle_detection_result = AsyncMock()

        await manager._on_tts_stopped(
            TTSStopped(session_id="session", response_id="response"),
        )

        self.assertTrue(detector.listening)
        async_detect.assert_awaited_once_with(assistant_interrupted=True)
        manager._handle_detection_result.assert_awaited_once_with(idle_result)

        detector.listening = False
        async_detect.reset_mock()
        manager._handle_detection_result.reset_mock()

        await manager._on_tts_playback_finished(
            TTSPlaybackFinished(session_id="session", response_id="response"),
        )

        self.assertTrue(detector.listening)
        async_detect.assert_not_awaited()
        manager._handle_detection_result.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
