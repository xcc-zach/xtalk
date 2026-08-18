from __future__ import annotations

import asyncio
import base64
import inspect
import io
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any

from aiohttp import web

from xtalk.models.registry import get_model_class
from xtalk.models.tts.moss_tts_nano import MossTTSNano, _decode_wav_to_pcm48


def _wav_bytes(
    sample_rate: int,
    samples: list[int],
    channels: int = 1,
) -> bytes:
    """Build a PCM16 WAV payload for protocol tests."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
        )
    return output.getvalue()


class MossTTSNanoTest(unittest.IsolatedAsyncioTestCase):
    """Exercise the shared MOSS-TTS-Nano HTTP protocol."""

    async def asyncSetUp(self) -> None:
        """Start local Python- and Rust-compatible HTTP test services."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        self.first_voice_path = temporary_path / "first.wav"
        self.second_voice_path = temporary_path / "second.wav"
        self.first_voice_path.write_bytes(_wav_bytes(16_000, [1, 2, 3, 4]))
        self.second_voice_path.write_bytes(_wav_bytes(16_000, [5, 6, 7, 8]))

        self.rust_requests: list[dict[str, Any]] = []
        rust_app = web.Application()
        rust_app.router.add_post("/api/generate", self._rust_generate)
        self.rust_runner, self.rust_base_url = await self._start_app(rust_app)

        self.python_requests: list[dict[str, Any]] = []
        python_app = web.Application()
        python_app.router.add_post("/api/generate", self._python_generate)
        self.python_runner, self.python_base_url = await self._start_app(python_app)

        self.retry_requests: dict[str, int] = {}
        retry_app = web.Application()
        retry_app.router.add_post("/api/generate", self._retry_generate)
        self.retry_runner, self.retry_base_url = await self._start_app(retry_app)

    async def asyncTearDown(self) -> None:
        """Stop local HTTP test services and remove temporary voices."""
        await self.rust_runner.cleanup()
        await self.python_runner.cleanup()
        await self.retry_runner.cleanup()
        self.temporary_directory.cleanup()

    async def _start_app(
        self, app: web.Application
    ) -> tuple[web.AppRunner, str]:
        """Start one aiohttp application on an ephemeral loopback port."""
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        if server is None or not server.sockets:
            raise RuntimeError("Test HTTP server failed to start")
        port = server.sockets[0].getsockname()[1]
        return runner, f"http://127.0.0.1:{port}"

    async def _rust_generate(self, request: web.Request) -> web.Response:
        """Capture the shared multipart request on the Rust service."""
        form = await request.post()
        upload = form["prompt_audio"]
        if not isinstance(upload, web.FileField):
            raise RuntimeError("prompt_audio was not uploaded as a file")
        self.rust_requests.append(
            {
                "text": form["text"],
                "filename": upload.filename,
                "content": upload.file.read(),
            }
        )
        return web.json_response(
            {
                "audio_base64": base64.b64encode(
                    _wav_bytes(48_000, [0, 1_000, -1_000, 0])
                ).decode("ascii"),
                "sample_rate": 48_000,
            }
        )

    async def _python_generate(self, request: web.Request) -> web.Response:
        """Capture an official multipart request and return base64 WAV."""
        form = await request.post()
        upload = form["prompt_audio"]
        if not isinstance(upload, web.FileField):
            raise RuntimeError("prompt_audio was not uploaded as a file")
        self.python_requests.append(
            {
                "text": form["text"],
                "filename": upload.filename,
                "content": upload.file.read(),
            }
        )
        return web.json_response(
            {
                "audio_base64": base64.b64encode(
                    _wav_bytes(
                        48_000,
                        [0, 0, 2_000, 1_000, -2_000, -1_000, 0, 0],
                        channels=2,
                    )
                ).decode("ascii"),
                "sample_rate": 48_000,
            }
        )

    async def _retry_generate(self, request: web.Request) -> web.Response:
        """Return empty audio once per text before a usable retry response."""
        form = await request.post()
        text = str(form["text"])
        request_count = self.retry_requests.get(text, 0) + 1
        self.retry_requests[text] = request_count
        samples = (
            []
            if request_count == 1 or text == "永远为空"
            else [0, 1_000, -1_000, 0]
        )
        return web.json_response(
            {
                "audio_base64": base64.b64encode(
                    _wav_bytes(48_000, samples)
                ).decode("ascii"),
                "sample_rate": 48_000,
            }
        )

    def _voices(self) -> list[dict[str, str]]:
        """Return two IndexTTS-compatible voice entries."""
        return [
            {"name": "Junhao", "path": str(self.first_voice_path)},
            {"name": "Xiaoyu", "path": str(self.second_voice_path)},
        ]

    async def test_sync_client_uses_rust_protocol(self) -> None:
        """Send the official multipart shape to the native Rust endpoint."""
        client = MossTTSNano(self.rust_base_url, voices=self._voices())
        client.set_voice(["Xiaoyu"])
        pcm = await asyncio.to_thread(client.synthesize, "你好")

        self.assertEqual(len(pcm), 8)
        self.assertEqual(
            self.rust_requests,
            [
                {
                    "text": "你好",
                    "filename": "second.wav",
                    "content": self.second_voice_path.read_bytes(),
                }
            ],
        )

    async def test_async_client_uses_rust_protocol(self) -> None:
        """Use the same multipart protocol asynchronously with Rust."""
        client = MossTTSNano(self.rust_base_url, voices=self._voices())
        pcm = await client.async_synthesize("测试")

        self.assertEqual(len(pcm), 8)
        self.assertEqual(self.rust_requests[0]["filename"], "first.wav")

    async def test_sync_client_uses_python_protocol(self) -> None:
        """Upload the selected reference path to the Python endpoint."""
        client = MossTTSNano(self.python_base_url, voices=self._voices())
        client.set_voice(["Xiaoyu"])
        pcm = await asyncio.to_thread(client.synthesize, "Python 测试")

        self.assertEqual(len(pcm), 8)
        self.assertEqual(self.python_requests[0]["text"], "Python 测试")
        self.assertEqual(self.python_requests[0]["filename"], "second.wav")
        self.assertEqual(
            self.python_requests[0]["content"],
            self.second_voice_path.read_bytes(),
        )

    async def test_async_client_uses_python_protocol(self) -> None:
        """Support asynchronous multipart synthesis against Python."""
        client = MossTTSNano(self.python_base_url, voices=self._voices())
        pcm = await client.async_synthesize("异步测试")

        self.assertEqual(len(pcm), 8)
        self.assertEqual(self.python_requests[0]["filename"], "first.wav")

    async def test_sync_client_retries_empty_audio(self) -> None:
        """Retry one successful service response containing an empty WAV."""
        client = MossTTSNano(self.retry_base_url, voices=self._voices())

        pcm = await asyncio.to_thread(client.synthesize, "同步重试")

        self.assertEqual(len(pcm), 8)
        self.assertEqual(self.retry_requests["同步重试"], 2)

    async def test_async_client_retries_empty_audio(self) -> None:
        """Apply the same empty-audio retry policy to async synthesis."""
        client = MossTTSNano(self.retry_base_url, voices=self._voices())

        pcm = await client.async_synthesize("异步重试")

        self.assertEqual(len(pcm), 8)
        self.assertEqual(self.retry_requests["异步重试"], 2)

    async def test_client_reports_repeated_empty_audio(self) -> None:
        """Raise a specific failure after the bounded retry is exhausted."""
        client = MossTTSNano(self.retry_base_url, voices=self._voices())

        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            await client.async_synthesize("永远为空")

        self.assertEqual(self.retry_requests["永远为空"], 2)

    async def test_clone_preserves_voice(self) -> None:
        """Copy voice selection without sharing mutable client state."""
        client = MossTTSNano(self.rust_base_url, voices=self._voices())
        client.set_voice(["Xiaoyu"])

        clone = client.clone()
        await clone.async_synthesize("克隆")
        self.assertEqual(self.rust_requests[-1]["filename"], "second.wav")

    def test_init_only_exposes_base_url_and_voices(self) -> None:
        """Keep model configuration limited to the requested two fields."""
        self.assertEqual(
            list(inspect.signature(MossTTSNano.__init__).parameters),
            ["self", "base_url", "voices"],
        )

    def test_registered_model(self) -> None:
        """Expose the client to the standard model loader."""
        self.assertIs(
            get_model_class("tts", "MossTTSNano"),
            MossTTSNano,
        )

    def test_rejects_invalid_voice_configuration(self) -> None:
        """Require the same name/path fields consumed by IndexTTS."""
        with self.assertRaises(ValueError):
            MossTTSNano(voices=[{"name": "missing-path"}])

    def test_rejects_non_48000_hz_service_audio(self) -> None:
        """Keep the client and both service implementations fixed at 48 kHz."""
        with self.assertRaisesRegex(RuntimeError, "48000 Hz"):
            _decode_wav_to_pcm48(_wav_bytes(24_000, [0, 1, -1, 0]))


if __name__ == "__main__":
    unittest.main()
