"""Tests for XTurnix request construction."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from xtalk.models.turn_detector.xturnix import XTurnix
from xtalk.models.turn_detector.interfaces import (
    TurnDetectionAction,
    TurnDetectionSemantic,
)


class XTurnixRequestTests(unittest.IsolatedAsyncioTestCase):
    """Verify dialogue normalization before XTurnix inference requests."""

    async def test_infer_action_removes_only_a_trailing_assistant_message(
        self,
    ) -> None:
        """Remove a trailing assistant turn without mutating stored dialogue."""
        cases = [
            {
                "dialogue": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                "expected": [{"role": "user", "content": "hello"}],
            },
            {
                "dialogue": [
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "hello"},
                ],
                "expected": [
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "hello"},
                ],
            },
        ]
        for case in cases:
            dialogue = case["dialogue"]
            with self.subTest(dialogue=dialogue):
                await self._assert_request_dialogue(dialogue, case["expected"])

    async def _assert_request_dialogue(
        self,
        dialogue: list[dict[str, str]],
        expected_dialogue: list[dict[str, str]],
    ) -> None:
        """Run one mocked inference and verify its normalized messages."""
        detector = XTurnix()
        original_dialogue = [dict(message) for message in dialogue]
        action_token_ids = {
            "<|start|>": 1,
            "<|keep|>": 2,
            "<|stop|>": 3,
        }
        response = {
            "choices": [
                {
                    "message": {
                        "content": "<|start|>",
                    }
                }
            ]
        }

        with (
            patch.object(
                detector,
                "_ensure_action_token_ids",
                new=AsyncMock(return_value=action_token_ids),
            ),
            patch.object(
                detector,
                "_truncate_messages",
                new=AsyncMock(side_effect=lambda _session, messages: messages),
            ),
            patch.object(
                detector,
                "_post_json",
                new=AsyncMock(return_value=response),
            ) as post_json,
        ):
            action = await detector._infer_action(dialogue, "<|listening|>")

        self.assertEqual(action, "<|start|>")
        request_body = post_json.await_args.args[2]
        self.assertEqual(request_body["messages"][1:], expected_dialogue)
        self.assertEqual(dialogue, original_dialogue)


class XTurnixActionGatingTests(unittest.IsolatedAsyncioTestCase):
    """Verify listening actions are gated by the current VAD segment."""

    async def test_new_speech_segment_closes_previous_pause_gate(self) -> None:
        """Ignore start predictions until the newest VAD segment has paused."""
        detector = XTurnix()
        infer_action = AsyncMock(side_effect=["<|keep|>", "<|start|>"])

        with patch.object(detector, "_infer_action", new=infer_action):
            await detector.async_detect(speech_start=True)
            speaking_result = await detector.async_detect(text="hello")
            paused_result = await detector.async_detect(
                text="hello",
                speech_pause=True,
            )
            await detector.async_detect(speech_start=True)
            resumed_result = await detector.async_detect(text="hello again")
            resumed_pause_result = await detector.async_detect(
                text="hello again",
                speech_pause=True,
            )

        self.assertEqual(
            speaking_result.action,
            TurnDetectionAction.DO_NOTHING,
        )
        self.assertEqual(
            paused_result.action,
            TurnDetectionAction.DO_NOTHING,
        )
        self.assertEqual(
            resumed_result.action,
            TurnDetectionAction.DO_NOTHING,
        )
        self.assertEqual(
            resumed_pause_result.action,
            TurnDetectionAction.START_GENERATION,
        )
        self.assertEqual(infer_action.await_count, 2)

    async def test_pause_reuses_matching_partial_start_once(self) -> None:
        """Apply a matching pre-pause start without requesting XTurnix again."""
        detector = XTurnix()
        infer_action = AsyncMock(return_value="<|start|>")

        with patch.object(detector, "_infer_action", new=infer_action):
            await detector.async_detect(speech_start=True)
            partial_result = await detector.async_detect(text="hello")
            paused_result = await detector.async_detect(
                text="hello",
                speech_pause=True,
            )
            duplicate_pause_result = await detector.async_detect(
                text="hello",
                speech_pause=True,
            )

        self.assertEqual(
            partial_result.action,
            TurnDetectionAction.DO_NOTHING,
        )
        self.assertEqual(
            paused_result.action,
            TurnDetectionAction.START_GENERATION,
        )
        self.assertEqual(
            duplicate_pause_result.action,
            TurnDetectionAction.DO_NOTHING,
        )
        self.assertEqual(infer_action.await_count, 1)

    async def test_in_flight_partial_start_is_reused_when_pause_arrives(self) -> None:
        """Apply a pre-pause result that completes after the matching pause."""
        detector = XTurnix()
        inference_started = asyncio.Event()
        release_inference = asyncio.Event()
        inference_count = 0

        async def infer_start(
            _dialogue: list[dict[str, str]],
            _state: str,
        ) -> str:
            nonlocal inference_count
            inference_count += 1
            inference_started.set()
            await release_inference.wait()
            return "<|start|>"

        with patch.object(detector, "_infer_action", new=infer_start):
            await detector.async_detect(speech_start=True)
            partial_task = asyncio.create_task(detector.async_detect(text="hello"))
            await asyncio.wait_for(inference_started.wait(), timeout=1.0)
            pause_task = asyncio.create_task(
                detector.async_detect(text="hello", speech_pause=True)
            )
            for _ in range(10):
                if detector._user_stream.speech_pause_seen:
                    break
                await asyncio.sleep(0)
            self.assertTrue(detector._user_stream.speech_pause_seen)
            release_inference.set()
            partial_result, pause_result = await asyncio.gather(
                partial_task,
                pause_task,
            )

        self.assertEqual(
            {partial_result.action, pause_result.action},
            {
                TurnDetectionAction.START_GENERATION,
                TurnDetectionAction.DO_NOTHING,
            },
        )
        self.assertEqual(inference_count, 1)

    async def test_pause_does_not_reuse_result_for_changed_dialogue(self) -> None:
        """Request a fresh action when text changes before the pause arrives."""
        detector = XTurnix()
        infer_action = AsyncMock(side_effect=["<|keep|>", "<|start|>"])

        with patch.object(detector, "_infer_action", new=infer_action):
            await detector.async_detect(speech_start=True)
            await detector.async_detect(text="hello")
            result = await detector.async_detect(
                text="hello world",
                speech_pause=True,
            )

        self.assertEqual(result.action, TurnDetectionAction.START_GENERATION)
        self.assertEqual(infer_action.await_count, 2)

    async def test_pause_does_not_reuse_result_from_another_state(self) -> None:
        """Request a fresh action after the assistant state changes."""
        detector = XTurnix()
        detector.listening = False
        infer_action = AsyncMock(side_effect=["<|keep|>", "<|start|>"])

        with patch.object(detector, "_infer_action", new=infer_action):
            await detector.async_detect(speech_start=True)
            await detector.async_detect(text="hello")
            detector.listening = True
            result = await detector.async_detect(
                text="hello",
                speech_pause=True,
            )

        self.assertEqual(result.action, TurnDetectionAction.START_GENERATION)
        self.assertEqual(infer_action.await_count, 2)

    async def test_speech_start_invalidates_in_flight_paused_decision(self) -> None:
        """Discard a paused-segment result that returns after speech resumes."""
        detector = XTurnix()
        with patch.object(
            detector,
            "_infer_action",
            new=AsyncMock(return_value="<|keep|>"),
        ):
            await detector.async_detect(speech_start=True)
            await detector.async_detect(text="hello", speech_pause=True)

        inference_started = asyncio.Event()
        release_inference = asyncio.Event()
        inference_count = 0

        async def infer_start(
            _dialogue: list[dict[str, str]],
            _state: str,
        ) -> str:
            nonlocal inference_count
            inference_count += 1
            if inference_count == 1:
                inference_started.set()
                await release_inference.wait()
            return "<|start|>"

        with patch.object(detector, "_infer_action", new=infer_start):
            pending_result = asyncio.create_task(
                detector.async_detect(text="hello again")
            )
            await asyncio.wait_for(inference_started.wait(), timeout=1.0)
            await detector.async_detect(speech_start=True)
            release_inference.set()
            result = await asyncio.wait_for(pending_result, timeout=1.0)

        self.assertEqual(result.action, TurnDetectionAction.DO_NOTHING)
        self.assertEqual(inference_count, 2)

    async def test_assistant_interruption_rechecks_paused_user_input(self) -> None:
        """Re-evaluate the latest paused input after interrupted playback."""
        detector = XTurnix()
        detector.listening = False
        infer_action = AsyncMock(side_effect=["<|stop|>", "<|start|>"])

        with patch.object(detector, "_infer_action", new=infer_action):
            await detector.async_detect(speech_start=True)
            stop_result = await detector.async_detect(
                text="filter",
                speech_pause=True,
            )
            detector.listening = True
            start_result = await detector.async_detect(
                assistant_interrupted=True,
            )
            duplicate_result = await detector.async_detect(
                assistant_interrupted=True,
            )

        self.assertEqual(
            stop_result.action,
            TurnDetectionAction.STOP_SPEAKING,
        )
        self.assertEqual(
            start_result.action,
            TurnDetectionAction.START_GENERATION,
        )
        self.assertEqual(
            duplicate_result.action,
            TurnDetectionAction.DO_NOTHING,
        )
        self.assertEqual(infer_action.await_count, 2)
        self.assertEqual(
            [call.args[1] for call in infer_action.await_args_list],
            ["<|speaking|>", "<|listening|>"],
        )

    async def test_assistant_interruption_keeps_vad_pause_gate(self) -> None:
        """Do not start after interruption until the user has paused."""
        detector = XTurnix()
        detector.listening = False
        infer_action = AsyncMock(side_effect=["<|stop|>", "<|start|>"])

        with patch.object(detector, "_infer_action", new=infer_action):
            await detector.async_detect(speech_start=True)
            await detector.async_detect(text="filter")
            detector.listening = True
            result = await detector.async_detect(assistant_interrupted=True)

        self.assertEqual(result.action, TurnDetectionAction.DO_NOTHING)
        self.assertEqual(result.semantic, TurnDetectionSemantic.INCOMPLETE)
        self.assertEqual(infer_action.await_count, 2)

    async def test_in_flight_stop_survives_new_asr_revision(self) -> None:
        """Return stop immediately when a newer speaking partial arrives."""
        detector = XTurnix()
        detector.listening = False
        inference_started = asyncio.Event()
        release_inference = asyncio.Event()
        inference_count = 0

        async def infer_stop(
            _dialogue: list[dict[str, str]],
            _state: str,
        ) -> str:
            nonlocal inference_count
            inference_count += 1
            inference_started.set()
            await release_inference.wait()
            return "<|stop|>"

        with patch.object(detector, "_infer_action", new=infer_stop):
            first_task = asyncio.create_task(detector.async_detect(text="hello"))
            await asyncio.wait_for(inference_started.wait(), timeout=1.0)
            latest_task = asyncio.create_task(
                detector.async_detect(text="hello world")
            )
            for _ in range(10):
                if detector._user_revision == 2:
                    break
                await asyncio.sleep(0)
            self.assertEqual(detector._user_revision, 2)
            release_inference.set()
            first_result, latest_result = await asyncio.gather(
                first_task,
                latest_task,
            )

        self.assertEqual(
            first_result.action,
            TurnDetectionAction.STOP_SPEAKING,
        )
        self.assertEqual(
            latest_result.action,
            TurnDetectionAction.DO_NOTHING,
        )
        self.assertEqual(inference_count, 1)

    async def test_in_flight_stop_expires_after_speaking_state_changes(self) -> None:
        """Discard stop when playback leaves speaking state during inference."""
        detector = XTurnix()
        detector.listening = False
        inference_started = asyncio.Event()
        release_inference = asyncio.Event()
        inference_count = 0

        async def infer_stop(
            _dialogue: list[dict[str, str]],
            _state: str,
        ) -> str:
            nonlocal inference_count
            inference_count += 1
            if inference_count == 1:
                inference_started.set()
                await release_inference.wait()
            return "<|stop|>"

        with patch.object(detector, "_infer_action", new=infer_stop):
            pending_result = asyncio.create_task(
                detector.async_detect(text="hello")
            )
            await asyncio.wait_for(inference_started.wait(), timeout=1.0)
            detector.listening = True
            release_inference.set()
            result = await asyncio.wait_for(pending_result, timeout=1.0)

        self.assertEqual(result.action, TurnDetectionAction.DO_NOTHING)
        self.assertEqual(inference_count, 2)
