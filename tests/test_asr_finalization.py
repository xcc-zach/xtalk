"""Regression tests for ASR turn finalization."""

from __future__ import annotations

import asyncio
import unittest

from xtalk.models import ASR, Models
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    ASRGateState,
    ASRResultFinal,
    ASRResultPartial,
    AudioFrameReceived,
)
from xtalk.serving.modules.asr_manager import ASRManager, AudioConsumer


class ASRManagerAudioSourceTests(unittest.TestCase):
    """Verify ASR consumes the original microphone audio stream."""

    def test_audio_handler_subscribes_to_raw_frames(self) -> None:
        """Subscribe the ASR audio handler only to raw audio frames."""

        event_types = {
            metadata["event_type"]
            for method_name, handlers in ASRManager.__event_handlers_meta__
            if method_name == "_handle_audio_frame"
            for metadata in handlers
        }

        self.assertEqual(event_types, {AudioFrameReceived})

    def test_asr_events_default_to_unchecked_gate_state(self) -> None:
        """Leave existing ASR events unaccepted until a future gate runs."""

        partial = ASRResultPartial(session_id="session")
        final = ASRResultFinal(session_id="session")

        self.assertIs(partial.gate_state, ASRGateState.UNCHECKED)
        self.assertIs(final.gate_state, ASRGateState.UNCHECKED)


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


class _BlockingASR(ASR):
    """Hold a streaming recognition result until the test releases it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def recognize(self, audio: bytes) -> str:
        """Return a fixed one-shot recognition result."""

        del audio
        return "stale"

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Wait before returning a result from the superseded audio turn."""

        del audio
        del is_final
        del chat_history
        self.started.set()
        await self.release.wait()
        return "stale"

    def reset(self) -> None:
        """Reset the stateless test recognizer."""

        return None

    def clone(self) -> "_BlockingASR":
        """Return another independently controlled recognizer."""

        return _BlockingASR()


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

    async def test_abort_drops_in_flight_recognition_result(self) -> None:
        """Do not publish stale speech text after a text turn aborts audio."""

        event_bus = EventBus(enable_history=True)
        asr = _BlockingASR()
        consumer = AudioConsumer(
            event_bus=event_bus,
            session_id="session",
            models=Models({ASR: asr}),
        )
        self.addAsyncCleanup(consumer.shutdown)

        await asyncio.sleep(0)
        await consumer.start()
        end_task = asyncio.create_task(consumer.end())
        await asyncio.wait_for(asr.started.wait(), timeout=1.0)

        abort_task = asyncio.create_task(consumer.abort())
        await asyncio.sleep(0)
        self.assertGreaterEqual(consumer._recognition_generation, 1)
        asr.release.set()
        await asyncio.wait_for(
            asyncio.gather(end_task, abort_task),
            timeout=1.0,
        )

        self.assertEqual(event_bus.get_history(), [])


if __name__ == "__main__":
    unittest.main()
