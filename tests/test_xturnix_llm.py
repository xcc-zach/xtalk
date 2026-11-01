"""Tests for XTurnixLLM waiting windows and LLM judgements."""

from __future__ import annotations

import asyncio
import types
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, BaseMessage

from xtalk.model_loader import init_registered_model
from xtalk.models.turn_detector import xturnix_llm
from xtalk.models.turn_detector.interfaces import (
    TurnDetectionAction,
    TurnDetectionSemantic,
)
from xtalk.models.turn_detector.xturnix_llm import XTurnixLLM

RESPOND = AIMessage(content='{"reason": "done", "judgement": "respond"}')
WAIT = AIMessage(content='{"reason": "not done", "judgement": "wait"}')


class _FakeLLM:
    """Chat model stand-in whose replies are resolved by the test."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[BaseMessage], asyncio.Future]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Record the request and wait for the test to resolve it."""
        future = asyncio.get_running_loop().create_future()
        self.calls.append((messages, future))
        return await future


class XTurnixLLMTests(unittest.IsolatedAsyncioTestCase):
    """Verify window scheduling, judgement delivery, and prompt rendering."""

    def setUp(self) -> None:
        self.now = 0.0
        clock = types.SimpleNamespace(monotonic=lambda: self.now)
        patcher = patch.object(xturnix_llm, "time", clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.llm = _FakeLLM()

    def _detector(self, **kwargs) -> XTurnixLLM:
        """Build a detector whose XTurnix actions are mocked."""
        detector = XTurnixLLM(llm=self.llm, **kwargs)
        detector._infer_action = AsyncMock(return_value="<|keep|>")
        return detector

    async def _audio_at(self, detector: XTurnixLLM, now: float):
        """Advance the clock and feed one audio frame."""
        self.now = now
        await self._settle()
        result = await detector.async_detect(audio=b"\x00\x00")
        await self._settle()
        return result

    @staticmethod
    async def _settle() -> None:
        """Let background request tasks run."""
        for _ in range(5):
            await asyncio.sleep(0)

    def _assert_start(self, result) -> None:
        self.assertEqual(result.action, TurnDetectionAction.START_GENERATION)
        self.assertEqual(result.semantic, TurnDetectionSemantic.COMPLETE)

    def _assert_idle(self, result) -> None:
        self.assertEqual(result.action, TurnDetectionAction.DO_NOTHING)

    async def test_keep_opens_window_and_respond_starts_once(self) -> None:
        """Send the first request after one interval and start on respond."""
        detector = self._detector()
        await detector.async_detect(text="明天上海的天气")

        self._assert_idle(await self._audio_at(detector, 0.5))
        self.assertEqual(len(self.llm.calls), 0)
        self._assert_idle(await self._audio_at(detector, 1.0))
        self.assertEqual(len(self.llm.calls), 1)

        self.llm.calls[0][1].set_result(RESPOND)
        self._assert_start(await self._audio_at(detector, 1.1))
        self._assert_idle(await self._audio_at(detector, 5.0))
        self.assertEqual(len(self.llm.calls), 1)

    async def test_start_without_pause_opens_window(self) -> None:
        """Open a window when XTurnix start is held back by a missing pause."""
        detector = self._detector()
        detector._infer_action.return_value = "<|start|>"
        result = await detector.async_detect(text="明天上海的天气")
        self.assertEqual(result.semantic, TurnDetectionSemantic.INCOMPLETE)

        await self._audio_at(detector, 1.0)
        self.assertEqual(len(self.llm.calls), 1)

    async def test_xturnix_start_does_not_open_window(self) -> None:
        """Leave the window closed when XTurnix itself starts the reply."""
        detector = self._detector()
        detector._infer_action.return_value = "<|start|>"
        result = await detector.async_detect(text="明天上海的天气", speech_pause=True)
        self._assert_start(result)

        await self._audio_at(detector, 1.0)
        self.assertEqual(len(self.llm.calls), 0)

    async def test_vad_events_do_not_open_or_close_window(self) -> None:
        """Keep the window schedule independent of VAD start and pause events."""
        detector = self._detector()
        await detector.async_detect(speech_pause=True)
        await self._audio_at(detector, 1.0)
        self.assertEqual(len(self.llm.calls), 0)

        await detector.async_detect(text="明天上海的天气")
        await detector.async_detect(speech_start=True)
        await detector.async_detect(speech_pause=True)
        await self._audio_at(detector, 2.0)
        self.assertEqual(len(self.llm.calls), 1)
        self.assertFalse(self.llm.calls[0][1].cancelled())

    async def test_requests_follow_interval_multiples(self) -> None:
        """Send one request per due multiple without backfill or after timeout."""
        cases = [
            {"times": [1.0, 1.5, 2.9, 3.0, 4.0, 5.0], "waits": ["1.0", "2.9", "3.0"]},
            {"times": [3.0, 3.5, 4.0], "waits": ["3.0"]},
        ]
        for case in cases:
            with self.subTest(times=case["times"]):
                self.llm = _FakeLLM()
                self.now = 0.0
                detector = self._detector(llm_timeout=3.0)
                await detector.async_detect(text="明天上海的天气")
                for now in case["times"]:
                    await self._audio_at(detector, now)

                waits = [
                    messages[1].content.rsplit(": ", 1)[1]
                    for messages, _ in self.llm.calls
                ]
                self.assertEqual(waits, case["waits"])
                self.assertFalse(
                    any(future.done() for _, future in self.llm.calls)
                )

    async def test_later_respond_wins_and_cancels_other_requests(self) -> None:
        """Start on whichever request responds first and cancel the rest."""
        detector = self._detector()
        await detector.async_detect(text="明天上海的天气")
        for now in (1.0, 2.0, 3.0):
            await self._audio_at(detector, now)
        self.assertEqual(len(self.llm.calls), 3)

        self.llm.calls[1][1].set_result(RESPOND)
        self._assert_start(await self._audio_at(detector, 3.1))
        await self._settle()
        self.assertTrue(self.llm.calls[0][1].cancelled())
        self.assertTrue(self.llm.calls[2][1].cancelled())

    async def test_wait_unparseable_and_failed_requests_do_not_start(self) -> None:
        """End only the affected request when it waits, is unparseable, or fails."""
        detector = self._detector()
        await detector.async_detect(text="明天上海的天气")
        for now in (1.0, 2.0, 3.0, 4.0):
            await self._audio_at(detector, now)

        self.llm.calls[0][1].set_result(WAIT)
        self.llm.calls[1][1].set_result(AIMessage(content="respond"))
        with self.assertLogs(xturnix_llm.logger, level="ERROR"):
            self.llm.calls[2][1].set_exception(RuntimeError("llm down"))
            self._assert_idle(await self._audio_at(detector, 4.5))
        self.assertFalse(self.llm.calls[3][1].done())

        self.llm.calls[3][1].set_result(RESPOND)
        self._assert_start(await self._audio_at(detector, 4.6))

    async def test_respond_after_llm_timeout_is_accepted(self) -> None:
        """Accept a respond judgement that arrives after new requests stopped."""
        detector = self._detector(llm_timeout=1.0)
        await detector.async_detect(text="明天上海的天气")
        await self._audio_at(detector, 1.0)
        await self._audio_at(detector, 5.0)
        self.assertEqual(len(self.llm.calls), 1)

        self.llm.calls[0][1].set_result(RESPOND)
        self._assert_start(await self._audio_at(detector, 5.1))

    async def test_window_closure_cancels_and_discards_requests(self) -> None:
        """Cancel requests and ignore their replies once the window closes."""

        async def change_user_text(detector: XTurnixLLM) -> None:
            await detector.async_detect(text="明天上海的天气怎么样")

        async def start_speaking(detector: XTurnixLLM) -> None:
            detector.listening = False

        async def xturnix_starts(detector: XTurnixLLM) -> None:
            detector._infer_action.return_value = "<|start|>"
            await detector.async_detect(text="明天上海的天气呢", speech_pause=True)

        for close in (change_user_text, start_speaking, xturnix_starts):
            with self.subTest(close=close.__name__):
                self.llm = _FakeLLM()
                self.now = 0.0
                detector = self._detector()
                await detector.async_detect(text="明天上海的天气")
                await self._audio_at(detector, 1.0)
                self.llm.calls[0][1].set_result(RESPOND)
                await close(detector)

                self._assert_idle(await self._audio_at(detector, 1.5))
                self.assertTrue(self.llm.calls[0][1].done())

    async def test_new_xturnix_keep_restarts_window(self) -> None:
        """Restart the window timer from the newest XTurnix keep decision."""
        detector = self._detector()
        await detector.async_detect(text="明天上海的天气")
        await self._audio_at(detector, 1.0)
        self.now = 1.5
        await detector.async_detect(text="明天上海的天气怎么样")
        await self._settle()
        self.assertTrue(self.llm.calls[0][1].cancelled())

        await self._audio_at(detector, 2.0)
        self.assertEqual(len(self.llm.calls), 1)
        await self._audio_at(detector, 2.5)
        self.assertEqual(len(self.llm.calls), 2)
        self.assertTrue(self.llm.calls[1][0][1].content.endswith(": 1.0"))

    async def test_prompt_renders_history_and_output_format(self) -> None:
        """Render the system prompt variant and the full dialogue history."""
        for output_reason in (True, False):
            with self.subTest(output_reason=output_reason):
                self.llm = _FakeLLM()
                self.now = 0.0
                detector = self._detector(llm_output_reason=output_reason)
                await detector.async_detect(text="帮我查一下明天")
                detector.listening = False
                await detector.async_detect(assistant_text="好的，请问您想查明天的什么？")
                detector.listening = True
                await detector.async_detect(text="明天", speech_pause=True)
                await detector.async_detect(text="明天上海的天气")
                await self._audio_at(detector, 2.0)

                system, user = self.llm.calls[0][0]
                expected_format = (
                    '{"reason": "<one short sentence>", "judgement": "respond" | "wait"}'
                    if output_reason
                    else '{"judgement": "respond" | "wait"}'
                )
                self.assertTrue(
                    system.content.startswith(
                        "You are the turn-taking judge of a voice assistant"
                    )
                )
                self.assertTrue(
                    system.content.endswith(
                        "Return exactly one JSON object and nothing else:\n"
                        + expected_format
                    )
                )
                self.assertEqual(
                    user.content,
                    "Conversation so far (oldest first):\n"
                    "User: 帮我查一下明天\n"
                    "Assistant: 好的，请问您想查明天的什么？\n"
                    "User: 明天上海的天气\n"
                    "\n"
                    "Seconds since the user's last words: 2.0",
                )

    async def test_clone_shares_llm_without_session_state(self) -> None:
        """Clone settings and the LLM client but not dialogue or windows."""
        detector = self._detector(llm_interval=0.5, llm_timeout=4.0)
        await detector.async_detect(text="明天上海的天气")
        clone = detector.clone()
        clone._infer_action = AsyncMock(return_value="<|keep|>")

        self.assertIs(clone._llm, detector._llm)
        self.assertEqual(clone._llm_interval, 0.5)
        self.assertEqual(clone._llm_timeout, 4.0)
        self.assertEqual(clone._render_dialogue_locked(), [])
        await self._audio_at(clone, 1.0)
        self.assertEqual(len(self.llm.calls), 0)

    def test_invalid_llm_parameters_are_rejected(self) -> None:
        """Reject non-positive intervals and timeouts and non-bool reason flags."""
        for kwargs in (
            {"llm_interval": 0},
            {"llm_timeout": -1.0},
            {"llm_output_reason": "yes"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    XTurnixLLM(llm=self.llm, **kwargs)


def test_xturnix_llm_loads_through_model_discovery() -> None:
    """Load XTurnixLLM with a ChatOpenAI configuration dict."""
    detector = init_registered_model(
        slot="turn_detector",
        model_config={
            "type": "XTurnixLLM",
            "params": {
                "base_url": "http://127.0.0.1:8000",
                "llm": {
                    "model": "judge",
                    "base_url": "http://127.0.0.1:8001/v1",
                    "api_key": "EMPTY",
                },
                "llm_output_reason": False,
            },
        },
    )

    assert isinstance(detector, XTurnixLLM)
    assert detector._llm_output_reason is False
