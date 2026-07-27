"""Tests for the native SGLang-Omni MTD HTTP client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from aiohttp import web

from xtalk.model_loader import init_registered_model
from xtalk.models.speaker_diarization import SglangOmniMtdClient
from xtalk.models.speaker_diarization.sglang_omni_mtd_client import (
    SglangOmniRequestCancelled,
)


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


async def _start_server(handler: Handler) -> tuple[web.AppRunner, str]:
    """Start one local ephemeral aiohttp server."""

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _read_form(request: web.Request) -> dict[str, Any]:
    """Read multipart values while preserving uploaded bytes."""

    reader = await request.multipart()
    result: dict[str, Any] = {}
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.filename:
            result[part.name] = await part.read()
        else:
            result[part.name] = await part.text()
    return result


def test_http_request_wraps_wav_and_maps_swapped_exemplar_labels() -> None:
    """Exemplar overlap restores global labels after local-label permutation."""

    async def scenario() -> None:
        captured: dict[str, Any] = {}

        async def handler(request: web.Request) -> web.Response:
            captured.update(await _read_form(request))
            return web.json_response(
                {
                    "text": (
                        "[0.00][S02]我是甲[1.00] "
                        "[1.50][S01]我是乙[2.50] "
                        "[3.50][S02]继续说话[4.30]"
                    ),
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.0, "text": "[S02]我是甲"},
                        {"id": 1, "start": 1.5, "end": 2.5, "text": "[S01]我是乙"},
                        {"id": 2, "start": 3.5, "end": 4.3, "text": "[S02]继续说话"},
                    ],
                    "duration": 4.5,
                    "usage": {"type": "duration", "seconds": 5},
                },
                headers={"X-Request-Id": "transcription-server-id"},
            )

        runner, base_url = await _start_server(handler)
        client = SglangOmniMtdClient(base_url=base_url, model="mtd-test")
        try:
            result = await client.decode_snapshot(
                request_id="session/1/1/1/final",
                pcm16=b"\0\0" * 72000,
                sample_rate=16000,
                decoder_prefix=(
                    "[0.00][S01]我是甲[1.00] [1.50][S02]我是乙[2.50]"
                ),
                context_seconds=3.5,
                current_audio_seconds=1.0,
                is_final=True,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert captured["file"].startswith(b"RIFF")
        assert captured["file"][8:12] == b"WAVE"
        assert captured["model"] == "mtd-test"
        assert captured["response_format"] == "verbose_json"
        assert result.current_segments == [
            {
                "start_s": 0.0,
                "end_s": 0.8,
                "speaker_id": "S01",
                "text": "继续说话",
            }
        ]
        assert result.metrics["speaker_mapping"] == {"S02": "S01", "S01": "S02"}
        assert result.metrics["remote_request_id"] == "transcription-server-id"

    asyncio.run(scenario())


def test_unmatched_local_label_gets_new_global_id_without_collision() -> None:
    """A new local speaker never reuses a registered global label."""

    async def scenario() -> None:
        async def handler(_request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "text": "[0.00][S01]甲[1.00] [2.00][S01]旧人[2.50] [2.50][S02]新人[3.00]",
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.0, "text": "[S01]甲"},
                        {"id": 1, "start": 2.0, "end": 2.5, "text": "[S01]旧人"},
                        {"id": 2, "start": 2.5, "end": 3.0, "text": "[S02]新人"},
                    ],
                }
            )

        runner, base_url = await _start_server(handler)
        client = SglangOmniMtdClient(base_url=base_url)
        try:
            result = await client.decode_snapshot(
                request_id="new-speaker",
                pcm16=b"\0\0" * 48000,
                sample_rate=16000,
                decoder_prefix="[0.00][S02]甲[1.00]",
                context_seconds=2.0,
                current_audio_seconds=1.0,
                is_final=True,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert [item["speaker_id"] for item in result.current_segments] == ["S02", "S01"]
        assert all(item["speaker_id"] != "UNKNOWN" for item in result.current_segments)

    asyncio.run(scenario())


def test_unmatched_label_avoids_all_registered_and_mapped_ids() -> None:
    """A local-label collision allocates a fresh ID above the registered pool."""

    async def scenario() -> None:
        async def handler(_request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "text": (
                        "[0.00][S01]甲[1.00] [1.50][S03]乙[2.50] "
                        "[3.50][S01]旧人[4.00] [4.00][S02]新人[4.50]"
                    ),
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.0, "text": "[S01]甲"},
                        {"id": 1, "start": 1.5, "end": 2.5, "text": "[S03]乙"},
                        {"id": 2, "start": 3.5, "end": 4.0, "text": "[S01]旧人"},
                        {"id": 3, "start": 4.0, "end": 4.5, "text": "[S02]新人"},
                    ],
                }
            )

        runner, base_url = await _start_server(handler)
        client = SglangOmniMtdClient(base_url=base_url)
        try:
            result = await client.decode_snapshot(
                request_id="label-collision",
                pcm16=b"\0\0" * 72000,
                sample_rate=16000,
                decoder_prefix="[0.00][S02]甲[1.00] [1.50][S01]乙[2.50]",
                context_seconds=3.5,
                current_audio_seconds=1.0,
                is_final=True,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert [item["speaker_id"] for item in result.current_segments] == ["S02", "S03"]
        assert result.metrics["speaker_mapping"] == {
            "S01": "S02",
            "S03": "S01",
            "S02": "S03",
        }

    asyncio.run(scenario())


def test_empty_pool_preserves_compact_local_speaker_order() -> None:
    """Cold-start local labels become compact global IDs without UNKNOWN."""

    async def scenario() -> None:
        async def handler(_request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "text": "[0.10][S01]甲[0.40] [0.45][S02]乙[0.90]",
                    "segments": [
                        {"id": 0, "start": 0.1, "end": 0.4, "text": "[S01]甲"},
                        {"id": 1, "start": 0.45, "end": 0.9, "text": "[S02]乙"},
                    ],
                }
            )

        runner, base_url = await _start_server(handler)
        client = SglangOmniMtdClient(base_url=base_url)
        try:
            result = await client.decode_snapshot(
                request_id="cold-start",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                decoder_prefix="",
                context_seconds=0.0,
                current_audio_seconds=1.0,
                is_final=False,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert [item["speaker_id"] for item in result.current_segments] == ["S01", "S02"]

    asyncio.run(scenario())


def test_cancel_releases_waiting_decode_quickly() -> None:
    """Cancelling a partial unblocks the manager-facing decode coroutine."""

    async def scenario() -> None:
        started = asyncio.Event()
        release_handler = asyncio.Event()

        async def handler(_request: web.Request) -> web.Response:
            started.set()
            await release_handler.wait()
            return web.json_response({"text": "", "segments": []})

        runner, base_url = await _start_server(handler)
        client = SglangOmniMtdClient(base_url=base_url, request_timeout_s=60.0)
        decode_task = asyncio.create_task(
            client.decode_snapshot(
                request_id="cancel-me",
                pcm16=b"\0\0" * 1600,
                sample_rate=16000,
                decoder_prefix="",
                context_seconds=0.0,
                current_audio_seconds=0.1,
                is_final=False,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
            cancelled_at = time.perf_counter()
            await client.cancel("cancel-me")
            try:
                await asyncio.wait_for(decode_task, timeout=1.0)
                raise AssertionError("decode_snapshot unexpectedly completed")
            except SglangOmniRequestCancelled:
                pass
            assert time.perf_counter() - cancelled_at < 1.0
        finally:
            release_handler.set()
            await client.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_model_loader_discovers_sglang_client() -> None:
    """Configuration discovery can instantiate the new client type."""

    client = init_registered_model(
        slot="speaker_diarization",
        model_config={
            "type": "SglangOmniMtdClient",
            "params": {"base_url": "http://127.0.0.1:18714"},
        },
    )
    assert isinstance(client, SglangOmniMtdClient)
