"""Tests for generic multi-speaker diarization orchestration."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from xtalk.models import Models, SpeakerDiarization
from xtalk.models.speaker_diarization.interfaces import DiarizationResult
from xtalk.serving.event_bus import EventBus, EventDispatchMode
from xtalk.serving.events import (
    ASRGateState,
    ASRResultFinal,
    ASRResultPartial,
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
from xtalk.serving.modules.output_gateway import OutputGateway


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


class _RecordingWebSocket:
    """Record frontend text messages without a live network connection."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        """Record one JSON WebSocket message."""

        self.messages.append(message)


class MultiSpeakerTurnContextManagerTest(unittest.IsolatedAsyncioTestCase):
    """Verify generic scheduling and ASR/diarization joining."""

    async def test_model_presence_controls_enablement(self) -> None:
        """Enable diarization only when its model is registered."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        without_model = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="without-model",
            models=Models(),
            config={"multi_speaker": {}},
        )
        self.addAsyncCleanup(without_model.shutdown)
        with_model = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="with-model",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
            config={"multi_speaker": {}},
        )
        self.addAsyncCleanup(with_model.shutdown)

        self.assertFalse(without_model.enabled)
        self.assertFalse(without_model.history_gate_enabled)
        self.assertTrue(with_model.enabled)
        self.assertTrue(with_model.history_gate_enabled)

    async def test_history_gate_rejects_non_boolean_config(self) -> None:
        """Reject ambiguous string or numeric values for the boolean option."""

        with self.assertRaisesRegex(
            ValueError,
            "exclude_non_focus_from_history must be a boolean",
        ):
            MultiSpeakerTurnContextManager(
                event_bus=EventBus(),
                session_id="session",
                models=Models({SpeakerDiarization: _FakeDiarization()}),
                config={
                    "multi_speaker": {
                        "exclude_non_focus_from_history": "false",
                    }
                },
            )

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
                    "diarization": {
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
            config={
                "multi_speaker": {
                    "exclude_non_focus_from_history": False,
                }
            },
        )
        self.addAsyncCleanup(manager.shutdown)
        self.assertFalse(manager.history_gate_enabled)

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
        self.assertFalse(ready[0].should_respond)

    async def test_default_focuses_s01(self) -> None:
        """Respond to S01 and suppress other identified speakers by default."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
        )
        self.addAsyncCleanup(manager.shutdown)

        self.assertEqual(manager.response_policy, "focus_only")
        self.assertEqual(manager.focus_speaker_ids, {"S01"})
        self.assertTrue(manager.history_gate_enabled)
        self.assertTrue(manager._should_respond("S01"))
        self.assertFalse(manager._should_respond("S02"))
        unknown = manager._filter_focus_history(
            ASRResultFinal(session_id="session", text="unknown accepted"),
            SpeakerDiarizationTurnFinal(session_id="session"),
        )
        self.assertIsNotNone(unknown)
        assert unknown is not None
        self.assertEqual(unknown[0], "unknown accepted")

    async def test_default_gate_keeps_partial_out_of_history_and_agent(self) -> None:
        """Show previews until non-focus speech, while always shielding history."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
        )
        self.addAsyncCleanup(manager.shutdown)
        previews: list[str] = []
        agent_partials: list[str] = []

        async def capture_preview(event: ASRResultPartial) -> None:
            previews.append(event.text)

        async def capture_agent_partial(event: ASRResultPartial) -> None:
            agent_partials.append(event.text)

        event_bus.subscribe(ASRResultPartial, capture_preview, priority=40)
        event_bus.subscribe(ASRResultPartial, capture_agent_partial, priority=20)

        await event_bus.publish(
            ASRResultPartial(
                session_id="session",
                turn_id=3,
                text="focus preview",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )
        manager._observe_diarization_segments(
            3,
            [{"speaker_id": "S02", "text": "other"}],
        )
        await event_bus.publish(
            ASRResultPartial(
                session_id="session",
                turn_id=3,
                text="must not update",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )

        self.assertEqual(previews, ["focus preview"])
        self.assertEqual(agent_partials, [])
        self.assertEqual(
            event_bus.get_history(event_type=ASRResultPartial.TYPE),
            [],
        )

    async def test_output_gateway_previews_before_the_history_barrier(self) -> None:
        """Use the production output priority for the last allowed preview."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        models = Models({SpeakerDiarization: _FakeDiarization()})
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=models,
        )
        self.addAsyncCleanup(manager.shutdown)
        websocket = _RecordingWebSocket()
        OutputGateway(
            event_bus=event_bus,
            session_id="session",
            websocket=websocket,  # type: ignore[arg-type]
            models=models,
        )

        await event_bus.publish(
            ASRResultPartial(
                session_id="session",
                turn_id=8,
                text="visible",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )
        manager._observe_diarization_segments(
            8,
            [{"speaker_id": "S02", "text": "other"}],
        )
        await event_bus.publish(
            ASRResultPartial(
                session_id="session",
                turn_id=8,
                text="hidden",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(
            json.loads(websocket.messages[0]),
            {
                "action": "update_asr",
                "data": {
                    "text": "visible",
                    "origin": "asr",
                },
            },
        )

    async def test_mixed_turn_republishes_only_focus_content(self) -> None:
        """Replace a stopped mixed final with focus-only history content."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
        )
        self.addAsyncCleanup(manager.shutdown)
        accepted_finals: list[ASRResultFinal] = []

        async def capture_final(event: ASRResultFinal) -> None:
            accepted_finals.append(event)

        event_bus.subscribe(ASRResultFinal, capture_final, priority=5)
        await event_bus.publish(
            ASRResultFinal(
                session_id="session",
                turn_id=4,
                text="你好 不要记录",
                display_text="你好 不要记录",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )
        await manager._on_diarization_final(
            SpeakerDiarizationTurnFinal(
                session_id="session",
                turn_id=4,
                segments=[
                    {
                        "start_s": 0.0,
                        "end_s": 0.5,
                        "speaker_id": "S01",
                        "text": "你好",
                    },
                    {
                        "start_s": 0.5,
                        "end_s": 1.0,
                        "speaker_id": "S02",
                        "text": "不要记录",
                    },
                ],
                active_speaker_id="S02",
            )
        )

        self.assertEqual(len(accepted_finals), 1)
        self.assertEqual(accepted_finals[0].text, "你好")
        self.assertIs(accepted_finals[0].gate_state, ASRGateState.ACCEPTED)
        final_history = event_bus.get_history(event_type=ASRResultFinal.TYPE)
        self.assertEqual(final_history, accepted_finals)
        ready = event_bus.get_history(event_type=MultiSpeakerTurnReady.TYPE)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].asr_text, "你好")
        self.assertEqual(
            [segment["speaker_id"] for segment in ready[0].diarization_segments],
            ["S01"],
        )
        self.assertEqual(ready[0].active_speaker_id, "S01")
        self.assertFalse(ready[0].should_respond)

    async def test_pure_non_focus_turn_is_not_republished(self) -> None:
        """Drop pure non-focus finals before output, persistence, and Agent."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
        )
        self.addAsyncCleanup(manager.shutdown)
        downstream_finals: list[ASRResultFinal] = []

        async def capture_final(event: ASRResultFinal) -> None:
            downstream_finals.append(event)

        event_bus.subscribe(ASRResultFinal, capture_final, priority=5)
        await event_bus.publish(
            ASRResultFinal(
                session_id="session",
                turn_id=5,
                text="不要保留",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )
        await manager._on_diarization_final(
            SpeakerDiarizationTurnFinal(
                session_id="session",
                turn_id=5,
                segments=[
                    {
                        "start_s": 0.0,
                        "end_s": 0.5,
                        "speaker_id": "S02",
                        "text": "不要保留",
                    }
                ],
                active_speaker_id="S02",
            )
        )

        self.assertEqual(downstream_finals, [])
        self.assertEqual(event_bus.get_history(event_type=ASRResultFinal.TYPE), [])
        self.assertEqual(
            event_bus.get_history(event_type=MultiSpeakerTurnReady.TYPE),
            [],
        )

    async def test_pure_focus_turn_keeps_accurate_asr_text(self) -> None:
        """Prefer the full ASR transcript when every speaker is in focus."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
        )
        self.addAsyncCleanup(manager.shutdown)
        accepted_finals: list[ASRResultFinal] = []

        async def capture_final(event: ASRResultFinal) -> None:
            accepted_finals.append(event)

        event_bus.subscribe(ASRResultFinal, capture_final, priority=5)
        await event_bus.publish(
            ASRResultFinal(
                session_id="session",
                turn_id=6,
                text="准确的 ASR 文本！",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )
        await manager._on_diarization_final(
            SpeakerDiarizationTurnFinal(
                session_id="session",
                turn_id=6,
                segments=[
                    {
                        "start_s": 0.0,
                        "end_s": 0.5,
                        "speaker_id": "S01",
                        "text": "近似文本",
                    }
                ],
                active_speaker_id="S01",
            )
        )

        self.assertEqual(accepted_finals[0].text, "准确的 ASR 文本！")
        ready = event_bus.get_history(event_type=MultiSpeakerTurnReady.TYPE)
        self.assertEqual(ready[0].asr_text, "准确的 ASR 文本！")
        self.assertTrue(ready[0].should_respond)

    async def test_missing_speaker_can_be_suppressed(self) -> None:
        """Drop an unknown-speaker turn when the missing-speaker policy asks."""

        event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(event_bus.shutdown)
        manager = MultiSpeakerTurnContextManager(
            event_bus=event_bus,
            session_id="session",
            models=Models({SpeakerDiarization: _FakeDiarization()}),
            config={
                "multi_speaker": {
                    "suppress_when_speaker_missing": True,
                }
            },
        )
        self.addAsyncCleanup(manager.shutdown)

        await event_bus.publish(
            ASRResultFinal(
                session_id="session",
                turn_id=7,
                text="speaker unknown",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
        )
        await manager._on_diarization_final(
            SpeakerDiarizationTurnFinal(
                session_id="session",
                turn_id=7,
            )
        )

        self.assertEqual(event_bus.get_history(event_type=ASRResultFinal.TYPE), [])
        self.assertEqual(
            event_bus.get_history(event_type=MultiSpeakerTurnReady.TYPE),
            [],
        )


if __name__ == "__main__":
    unittest.main()
