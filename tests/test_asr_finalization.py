"""Regression tests for ASR turn finalization."""

from __future__ import annotations

import unittest

from xtalk.models import ASR, Models
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import ASRResultFinal, ASRResultPartial
from xtalk.serving.modules.asr_manager import AudioConsumer


class _BlankFinalASR(ASR):
    """Return a useful partial transcript followed by a blank final result."""

    PARTIAL_TEXT = "已经识别出的文本"

    def recognize(self, audio: bytes) -> str:
        """Return an empty one-shot recognition result."""

        del audio
        return ""

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Return a non-empty partial and whitespace for the final flush."""

        del audio
        del chat_history
        return "   " if is_final else self.PARTIAL_TEXT

    def reset(self) -> None:
        """Reset the stateless test recognizer."""

        return None

    def clone(self) -> "_BlankFinalASR":
        """Return another stateless test recognizer."""

        return _BlankFinalASR()


class ASRFinalizationTests(unittest.IsolatedAsyncioTestCase):
    """Verify that finalization cannot discard a useful partial transcript."""

    async def test_blank_final_uses_cached_partial_for_final_event(self) -> None:
        """Publish the cached partial when the final ASR flush is blank."""

        event_bus = EventBus(enable_history=True)
        consumer = AudioConsumer(
            event_bus=event_bus,
            session_id="session",
            models=Models({ASR: _BlankFinalASR()}),
        )
        self.addAsyncCleanup(consumer.shutdown)

        await consumer._recognize_and_publish(b"partial-audio")
        await consumer._recognize_and_publish(
            b"",
            is_asr_end=True,
            is_final_chunk=True,
        )

        events = event_bus.get_history()
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], ASRResultPartial)
        self.assertIsInstance(events[1], ASRResultFinal)
        self.assertEqual(events[0].text, _BlankFinalASR.PARTIAL_TEXT)
        self.assertEqual(events[1].text, _BlankFinalASR.PARTIAL_TEXT)


if __name__ == "__main__":
    unittest.main()
