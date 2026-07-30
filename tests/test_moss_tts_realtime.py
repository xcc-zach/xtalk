from __future__ import annotations

import unittest
from typing import Any

from aiohttp import web

from xtalk.models.registry import get_model_class
from xtalk.models.tts.interfaces import StreamingTextTTS
from xtalk.models.tts.moss_tts_realtime import MossTTSRealtime


class MossTTSRealtimeTest(unittest.IsolatedAsyncioTestCase):
    """Exercise the MOSS-TTS-Realtime adapter against a local WebSocket."""

    async def asyncSetUp(self) -> None:
        """Start a protocol-compatible local test service."""
        self.events: list[dict[str, Any]] = []
        app = web.Application()
        app.router.add_get("/tts/ws", self._websocket_handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        server = self.site._server
        if server is None or not server.sockets:
            raise RuntimeError("Test WebSocket server failed to start")
        port = server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self) -> None:
        """Stop the local test service."""
        await self.runner.cleanup()

    async def _websocket_handler(
        self, request: web.Request
    ) -> web.WebSocketResponse:
        """Serve one MOSS-TTS-Realtime-compatible streaming session."""
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)

        start_event = await websocket.receive_json()
        self.events.append(start_event)
        await websocket.send_json(
            {
                "type": "started",
                "session_id": start_event["session_id"],
                "audio_codec": "pcm_s16le",
                "sample_rate": 24000,
                "channels": 1,
            }
        )

        async for message in websocket:
            event = message.json()
            self.events.append(event)
            if event["type"] == "close":
                await websocket.send_json({"type": "closing"})
                break
            await websocket.send_json(
                {
                    "type": "accepted",
                    "accepted_text_len": len(event["text"]),
                    "is_final": event["is_final"],
                }
            )
            if event["is_final"]:
                await websocket.send_bytes(b"\x01\x02")
                await websocket.send_bytes(b"\x03\x04")
                await websocket.send_json(
                    {
                        "type": "completed",
                        "audio_bytes": 4,
                        "audio_chunks": 2,
                    }
                )
                break

        await websocket.close()
        return websocket

    async def test_streaming_lifecycle(self) -> None:
        """Map start, append, flush, audio, stop, and clone correctly."""
        client = MossTTSRealtime(self.base_url)
        self.assertIsInstance(client, StreamingTextTTS)

        await client.start()
        self.assertEqual(client.output_sample_rate, 24000)
        await client.append_text("你好")
        await client.append_text("，世界")
        await client.flush()
        await client.stop()

        chunks = [chunk async for chunk in client.audio_stream()]
        self.assertEqual(chunks, [b"\x01\x02", b"\x03\x04"])
        self.assertEqual(self.events[0]["type"], "start")
        self.assertEqual(self.events[0]["text"], "")
        self.assertEqual(
            [(event["text"], event["is_final"]) for event in self.events[1:]],
            [("你好", False), ("，世界", False), ("", True)],
        )

        clone = client.clone()
        self.assertIsInstance(clone, MossTTSRealtime)
        self.assertEqual(clone.base_url, self.base_url)

    async def test_async_synthesize(self) -> None:
        """Collect a complete request into one PCM payload."""
        client = MossTTSRealtime(self.base_url)
        self.assertEqual(await client.async_synthesize("完整文本"), b"\x01\x02\x03\x04")

    async def test_removes_line_breaks_from_incremental_text(self) -> None:
        """Do not forward CR or LF characters to the TTS service."""
        client = MossTTSRealtime(self.base_url)
        await client.start()
        await client.append_text("你\n好\r\n")
        await client.flush()
        await client.stop()
        self.assertEqual(self.events[1]["text"], "你好")

    async def test_emits_race_diagnostic_lifecycle_logs(self) -> None:
        """Expose append, flush, stop, and completion ordering at debug level."""
        client = MossTTSRealtime(self.base_url)
        with self.assertLogs(
            "xtalk.models.tts.moss_tts_realtime",
            level="DEBUG",
        ) as captured:
            await client.start()
            await client.append_text("你好")
            await client.flush()
            await client.stop()

        messages = "\n".join(captured.output)
        for stage in (
            "moss_started",
            "moss_append_sent",
            "moss_flush_sent",
            "moss_stop_begin",
            "moss_completed",
        ):
            self.assertIn(f"stage={stage}", messages)

        abort_client = MossTTSRealtime(self.base_url)
        with self.assertLogs(
            "xtalk.models.tts.moss_tts_realtime",
            level="DEBUG",
        ) as abort_captured:
            await abort_client.start()
            await abort_client.stop()
        self.assertIn("finalized=False", "\n".join(abort_captured.output))

    async def test_uses_first_reference_voice_by_default(self) -> None:
        """Send the first configured voice path in the start event."""
        client = MossTTSRealtime(
            self.base_url,
            voices=[
                {"name": "man", "path": "/server/man.wav"},
                {"name": "woman", "path": "/server/woman.wav"},
            ],
        )
        await client.start()
        await client.stop()
        self.assertEqual(self.events[0]["prompt_audio"], "/server/man.wav")

    async def test_set_voice_changes_reference_for_next_session(self) -> None:
        """Resolve a selected voice name to its configured reference path."""
        client = MossTTSRealtime(
            self.base_url,
            voices=[
                {"name": "man", "path": "/server/man.wav"},
                {"name": "woman", "path": "/server/woman.wav"},
            ],
        )
        client.set_voice(["woman"])
        clone = client.clone()
        await clone.start()
        await clone.stop()
        self.assertEqual(self.events[0]["prompt_audio"], "/server/woman.wav")

    def test_registered_model(self) -> None:
        """Expose the adapter to the standard TTS model loader."""
        self.assertIs(
            get_model_class("tts", "MossTTSRealtime"),
            MossTTSRealtime,
        )

    def test_rejects_relative_base_url(self) -> None:
        """Require an absolute service URL."""
        with self.assertRaises(ValueError):
            MossTTSRealtime("localhost:8000")

    def test_defaults_to_local_service(self) -> None:
        """Use the local port 8000 service by default."""
        client = MossTTSRealtime()
        self.assertEqual(client.base_url, "http://127.0.0.1:8000")

    def test_rejects_invalid_voices(self) -> None:
        """Validate the sample_local voice schema."""
        with self.assertRaises(ValueError):
            MossTTSRealtime(voices=[{"name": "missing-path"}])


if __name__ == "__main__":
    unittest.main()
