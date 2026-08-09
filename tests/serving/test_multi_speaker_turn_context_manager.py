"""Tests for generic multi-speaker diarization orchestration."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from xtalk.models import Models, SpeakerDiarization
from xtalk.models.speaker_diarization.interfaces import DiarizationResult
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    ASRResultFinal,
    EnhancedAudioFrameReceived,
    MultiSpeakerTurnReady,
    SpeakerDiarizationTurnFinal,
    TurnASREndRequested,
    TurnASRStartRequested,
    VADSpeechEnd,
)
from xtalk.serving.modules.multi_speaker_turn_context_manager import (
    MultiSpeakerTurnContextManager,
)


class _FakeDiarization:
    """Record generic snapshot calls and return one speaker segment."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def decode_snapshot(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        is_final: bool,
    ) -> DiarizationResult:
        """Return one deterministic snapshot-local result."""

        self.calls.append(
            {
                "request_id": request_id,
                "pcm16": pcm16,
                "sample_rate": sample_rate,
                "is_final": is_final,
            }
        )
        return DiarizationResult(
            segments=[
                {
                    "start_s": 0.0,
                    "end_s": len(pcm16) / (sample_rate * 2),
                    "speaker_id": "S01",
                    "text": "测试",
                }
            ],
            metrics={"backend": "fake"},
        )

    async def cancel(self, request_id: str) -> None:
        """Accept generic cancellation requests."""

        del request_id

    async def close(self) -> None:
        """Record model shutdown."""

        self.closed = True


class MultiSpeakerTurnContextManagerTest(unittest.IsolatedAsyncioTestCase):
    """Verify generic scheduling and ASR/diarization joining."""

    async def test_final_snapshot_uses_generic_model_contract(self) -> None:
        """Buffer one segment and publish its turn-level diarization result."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        model = _FakeDiarization()
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: model}),
            config={
                "multi_speaker": {
                    "enabled": True,
                    "diarization": {
                        "sample_rate": 16000,
                        "pre_buffer_s": 0.1,
                    },
                }
            },
        )
        self.addAsyncCleanup(manager.shutdown)

        await manager._on_audio_frame(
            EnhancedAudioFrameReceived(
                session_id="session",
                audio_data=b"\0\0" * 160,
                sample_rate=16000,
            )
        )
        await manager._on_segment_start(
            TurnASRStartRequested(
                session_id="session",
                turn_id=1,
                segment_id=1,
            )
        )
        await manager._on_audio_frame(
            EnhancedAudioFrameReceived(
                session_id="session",
                audio_data=b"\0\0" * 320,
                sample_rate=16000,
            )
        )
        await manager._on_vad_end(VADSpeechEnd(session_id="session"))
        await manager._on_turn_end(
            TurnASREndRequested(
                session_id="session",
                turn_id=1,
                segment_id=1,
            )
        )

        for _ in range(20):
            turn_finals = event_bus.get_history(
                event_type=SpeakerDiarizationTurnFinal.TYPE
            )
            if turn_finals:
                break
            await asyncio.sleep(0)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            set(model.calls[0]), {"request_id", "pcm16", "sample_rate", "is_final"}
        )
        self.assertTrue(model.calls[0]["is_final"])
        self.assertEqual(len(model.calls[0]["pcm16"]), 2 * (160 + 320))
        self.assertEqual(len(turn_finals), 1)
        self.assertEqual(turn_finals[0].active_speaker_id, "S01")

    async def test_join_publishes_multi_speaker_turn_ready(self) -> None:
        """Join independent ASR and diarization finals in the same manager."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
            config={"multi_speaker": {"enabled": True}},
        )
        self.addAsyncCleanup(manager.shutdown)

        await manager._on_asr_final(
            ASRResultFinal(
                session_id="session",
                turn_id=2,
                text="你好",
            )
        )
        await manager._on_diarization_final(
            SpeakerDiarizationTurnFinal(
                session_id="session",
                turn_id=2,
                diarization_text="[0.00][S02]你好[0.50]",
                active_speaker_id="S02",
            )
        )

        ready = event_bus.get_history(event_type=MultiSpeakerTurnReady.TYPE)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].asr_text, "你好")
        self.assertEqual(ready[0].active_speaker_id, "S02")


if __name__ == "__main__":
    unittest.main()
