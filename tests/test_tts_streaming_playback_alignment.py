"""Tests for StreamingTextTTS playback-alignment timing."""

from __future__ import annotations

import unittest
from collections.abc import Callable

from xtalk.models import Models
from xtalk.serving.event_bus import EventBus, EventDispatchMode
from xtalk.serving.events import (
    ResponseUpdate,
    TTSStarted,
    TTSStreamingTextAccepted,
)
from xtalk.serving.modules.tts_manager import TTSManager
from xtalk.serving.modules.tts_playback_manager import (
    TTSPlaybackManager,
    _PlaybackSegment,
)


class _FakeStreamingTTS:
    """Minimal streaming TTS double accepting incremental text."""

    def __init__(self, on_append: Callable[[], None] | None = None) -> None:
        """Initialize the double with an optional append callback."""

        self.on_append = on_append

    async def append_text(self, text: str) -> None:
        """Record one accepted text increment."""

        self.text = text
        if self.on_append is not None:
            self.on_append()


class StreamingPlaybackAlignmentTests(unittest.IsolatedAsyncioTestCase):
    """Verify immutable StreamingTextTTS L1 timing."""

    async def test_publishes_prepared_audio_watermark(self) -> None:
        """Attach the current prepared PCM duration to accepted text."""

        event_bus = EventBus(enable_history=True)
        manager = TTSManager(event_bus, "session", Models())
        manager._response_id = "response-1"
        manager._delivery_started = True
        manager._streaming_audio_duration_ms = 725.0
        manager._streaming_tts = _FakeStreamingTTS(
            lambda: setattr(manager, "_streaming_audio_duration_ms", 950.0)
        )

        await manager._append_streaming_text("新增")

        events = event_bus.get_history(event_type=TTSStreamingTextAccepted.TYPE)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].prepared_audio_ms, 725.0)

    async def test_new_text_does_not_reuse_played_audio(self) -> None:
        """Keep newly accepted text hidden until its anchored audio is played."""

        event_bus = EventBus(enable_history=True)
        manager = TTSPlaybackManager(event_bus, "session")
        await event_bus.publish(
            TTSStarted(session_id="session", response_id="response-1"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        await event_bus.publish(
            TTSStreamingTextAccepted(
                session_id="session",
                response_id="response-1",
                text="你",
                prepared_audio_ms=0.0,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        segment = manager._streaming_segment
        self.assertIsNotNone(segment)
        assert segment is not None

        segment.played_audio_ms = 580.0
        await manager._publish_progress_if_grew()
        self.assertEqual(manager._reported_text, "你")
        update_count = len(
            event_bus.get_history(event_type=ResponseUpdate.TYPE)
        )

        await event_bus.publish(
            TTSStreamingTextAccepted(
                session_id="session",
                response_id="response-1",
                text="好",
                prepared_audio_ms=1200.0,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

        self.assertEqual(manager._reported_text, "你")
        self.assertEqual(
            len(event_bus.get_history(event_type=ResponseUpdate.TYPE)),
            update_count,
        )
        self.assertEqual(segment.streaming_rough_units[-1].end_ms, 1480.0)

        segment.played_audio_ms = 1779.0
        await manager._publish_progress_if_grew()
        self.assertEqual(manager._reported_text, "你")

        segment.played_audio_ms = 1780.0
        await manager._publish_progress_if_grew()
        self.assertEqual(manager._reported_text, "你好")

    async def test_applies_conservative_streaming_l1_defaults(self) -> None:
        """Use fixed conservative timing only for StreamingTextTTS L1."""

        event_bus = EventBus(enable_history=True)
        manager = TTSPlaybackManager(event_bus, "session")
        await event_bus.publish(
            TTSStarted(session_id="session", response_id="response-1"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        await event_bus.publish(
            TTSStreamingTextAccepted(
                session_id="session",
                response_id="response-1",
                text="你，hello",
                prepared_audio_ms=0.0,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        segment = manager._streaming_segment
        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertEqual(
            [unit.end_ms for unit in segment.streaming_rough_units],
            [440.0, 820.0],
        )

        segment.played_audio_ms = 1119.0
        await manager._publish_progress_if_grew()
        self.assertEqual(manager._reported_text, "你，")

        segment.played_audio_ms = 1120.0
        await manager._publish_progress_if_grew()
        self.assertEqual(manager._reported_text, "你，hello")

    def test_regular_l1_and_streaming_l2_remain_unchanged(self) -> None:
        """Keep non-streaming-L1 rough-ratio behavior unchanged."""

        manager = TTSPlaybackManager(EventBus(), "session")
        regular = _PlaybackSegment(
            text="你好",
            turn_id=0,
            tts_mode="regular",
            generated_audio_ms=500.0,
            played_audio_ms=600.0,
        )
        streaming_l2 = _PlaybackSegment(
            text="你好",
            turn_id=0,
            tts_mode="streaming",
            total_audio_ms=800.0,
            played_audio_ms=400.0,
        )

        self.assertEqual(manager._build_rough_segment_prefix(regular), "你")
        self.assertEqual(
            manager._build_rough_segment_prefix(streaming_l2),
            "你",
        )


if __name__ == "__main__":
    unittest.main()
