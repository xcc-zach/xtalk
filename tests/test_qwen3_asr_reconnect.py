"""Regression tests for preserving Qwen ASR turn state across reconnects."""

from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

try:
    from xtalk.models.asr.qwen3_asr_flash_realtime import (
        Qwen3ASRFlashConfig,
        Qwen3ASRFlashRealtime,
    )
except ImportError:  # pragma: no cover - depends on optional DashScope package
    Qwen3ASRFlashConfig = None  # type: ignore[assignment,misc]
    Qwen3ASRFlashRealtime = None  # type: ignore[assignment,misc]


class _FakeCallback:
    """Return configured partial and final transcript values."""

    def __init__(self, *, partial: str = "", final: str = "") -> None:
        self.partial = partial
        self.final = final

    def get_last_partial(self) -> str:
        """Return the configured partial transcript."""

        return self.partial

    def wait_final(self, timeout: float) -> None:
        """Simulate a final-result timeout."""

        del timeout
        return None

    def finalize_segment(self, final_text: str | None = None) -> str:
        """Return the configured finalized transcript."""

        del final_text
        return self.final


class _FakeConversation:
    """Record decoded audio payloads and optionally fail one operation."""

    def __init__(
        self,
        *,
        fail_append_once: bool = False,
        fail_commit_once: bool = False,
    ) -> None:
        self.fail_append_once = fail_append_once
        self.fail_commit_once = fail_commit_once
        self.audio_payloads: list[bytes] = []
        self.commit_count = 0
        self.closed = False

    def append_audio(self, audio_base64: str) -> None:
        """Record audio or raise the configured transient send failure."""

        if self.fail_append_once:
            self.fail_append_once = False
            raise RuntimeError("transient append failure")
        self.audio_payloads.append(base64.b64decode(audio_base64))

    def commit(self) -> None:
        """Record commit or raise the configured transient commit failure."""

        if self.fail_commit_once:
            self.fail_commit_once = False
            raise RuntimeError("transient commit failure")
        self.commit_count += 1

    def close(self) -> None:
        """Record that the fake connection was closed."""

        self.closed = True


@unittest.skipIf(
    Qwen3ASRFlashRealtime is None,
    "DashScope optional dependency is unavailable",
)
class Qwen3ASRReconnectTests(unittest.TestCase):
    """Verify that reconnects retain transcript and full-turn audio state."""

    def _make_model(self, *, enable_turn_detection: bool = True):
        """Create a Qwen ASR instance configured for immediate local tests."""

        assert Qwen3ASRFlashConfig is not None
        assert Qwen3ASRFlashRealtime is not None
        return Qwen3ASRFlashRealtime(
            api_key="test-key",
            config=Qwen3ASRFlashConfig(
                enable_turn_detection=enable_turn_detection,
                tail_silence_cycles=1,
                tail_silence_bytes_per_cycle=2,
                tail_silence_delay_sec=0.0,
                final_wait_timeout_sec=0.0,
                reconnect_on_send_error_max_attempts=1,
            ),
        )

    @staticmethod
    def _install_session(model, conversation, callback) -> None:
        """Install one fake connected remote session on the model."""

        model._conv = conversation
        model._callback = callback
        model._connected = True

    def test_audio_send_reconnect_replays_complete_turn(self) -> None:
        """Replay prior and current audio when an incremental send reconnects."""

        model = self._make_model()
        old_conversation = _FakeConversation()
        old_callback = _FakeCallback(partial="已有文本")
        self._install_session(model, old_conversation, old_callback)
        self.assertEqual(model.recognize_stream(b"first-"), "已有文本")
        old_conversation.fail_append_once = True

        new_conversation = _FakeConversation()
        new_callback = _FakeCallback()

        def reconnect():
            """Swap in the fake replacement session without touching turn state."""

            self._install_session(model, new_conversation, new_callback)
            return new_conversation

        with patch.object(
            model,
            "_reconnect_preserving_turn",
            side_effect=reconnect,
        ):
            partial_text = model.recognize_stream(b"second")

        self.assertEqual(partial_text, "已有文本")
        self.assertEqual(new_conversation.audio_payloads, [b"first-second"])
        self.assertEqual(bytes(model._turn_audio), b"first-second")
        self.assertEqual(model._turn_text, "已有文本")

    def test_tail_reconnect_replays_audio_and_keeps_partial_text(self) -> None:
        """Replay the turn and return cached text after final flush reconnects."""

        model = self._make_model()
        old_conversation = _FakeConversation()
        old_callback = _FakeCallback(partial="缓存文本")
        self._install_session(model, old_conversation, old_callback)

        self.assertEqual(model.recognize_stream(b"turn-audio"), "缓存文本")
        old_conversation.fail_append_once = True

        new_conversation = _FakeConversation()
        new_callback = _FakeCallback()

        def reconnect():
            """Swap in the fake replacement session without touching turn state."""

            self._install_session(model, new_conversation, new_callback)
            return new_conversation

        with patch.object(
            model,
            "_reconnect_preserving_turn",
            side_effect=reconnect,
        ):
            final_text = model.recognize_stream(b"", is_final=True)

        self.assertEqual(final_text, "缓存文本")
        self.assertEqual(new_conversation.audio_payloads[0], b"turn-audio")
        self.assertEqual(new_conversation.audio_payloads[1], bytes(2))
        self.assertEqual(bytes(model._turn_audio), b"turn-audio")
        self.assertEqual(model._turn_text, "缓存文本")

    def test_commit_reconnect_replays_audio_before_committing(self) -> None:
        """Replay buffered audio when manual final commit requires reconnecting."""

        model = self._make_model(enable_turn_detection=False)
        old_conversation = _FakeConversation(fail_commit_once=True)
        old_callback = _FakeCallback(partial="提交前文本")
        self._install_session(model, old_conversation, old_callback)
        self.assertEqual(model.recognize_stream(b"all-audio"), "提交前文本")

        new_conversation = _FakeConversation()
        new_callback = _FakeCallback()

        def reconnect():
            """Swap in the fake replacement session without touching turn state."""

            self._install_session(model, new_conversation, new_callback)
            return new_conversation

        with patch.object(
            model,
            "_reconnect_preserving_turn",
            side_effect=reconnect,
        ):
            final_text = model.recognize_stream(b"", is_final=True)

        self.assertEqual(final_text, "提交前文本")
        self.assertEqual(new_conversation.audio_payloads[0], b"all-audio")
        self.assertEqual(new_conversation.audio_payloads[1], bytes(2))
        self.assertEqual(new_conversation.commit_count, 1)


if __name__ == "__main__":
    unittest.main()
