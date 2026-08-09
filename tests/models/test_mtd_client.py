"""Tests for the unified MTD HTTP client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

import numpy as np
from aiohttp import web

from xtalk.model_loader import init_registered_model
from xtalk.models.speaker_diarization import OfficialMtdClient
from xtalk.models.speaker_diarization.mtd import (
    MtdRequestCancelled,
    _SpeakerExemplar,
    _join_decoder_prefix_and_suffix,
    _parse_timestamped_text,
)


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _seed_exemplars(client: OfficialMtdClient) -> None:
    """Register two deterministic one-second exemplars in one model clone."""

    audio = np.zeros(16000, dtype=np.float32)
    client._exemplar_pool.items = {
        "S01": _SpeakerExemplar(
            speaker_id="S01",
            audio=audio.copy(),
            text="我是甲",
            score=1.0,
            quality={"overlap_class": "non_overlap"},
            source_start_s=0.0,
            source_end_s=1.0,
        ),
        "S02": _SpeakerExemplar(
            speaker_id="S02",
            audio=audio.copy(),
            text="我是乙",
            score=1.0,
            quality={"overlap_class": "non_overlap"},
            source_start_s=0.0,
            source_end_s=1.0,
        ),
    }


async def _start_server(
    handler: Handler,
    *,
    route: str = "/v1/audio/transcriptions",
    model_name: str | None = "mtd-test",
) -> tuple[web.AppRunner, str]:
    """Start one local ephemeral aiohttp server."""

    app = web.Application()
    app.router.add_post(route, handler)
    if model_name is not None:

        async def models(_request: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": model_name}]})

        app.router.add_get("/v1/models", models)
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


def test_http_request_uses_forced_decoder_prefix_and_crops_current_audio() -> None:
    """The raw assistant prefix is fixed decoder context, not post-hoc mapping."""

    async def scenario() -> None:
        captured: dict[str, Any] = {}

        async def handler(request: web.Request) -> web.Response:
            captured.update(await _read_form(request))
            return web.json_response(
                {
                    # Native SGLang returns only the newly generated suffix.
                    "text": "[3.50][S02]继续说话[4.30]",
                    "segments": [
                        {"id": 2, "start": 3.5, "end": 4.3, "text": "[S02]继续说话"},
                    ],
                    "duration": 4.5,
                    "usage": {"type": "duration", "seconds": 5},
                },
                headers={"X-Request-Id": "transcription-server-id"},
            )

        runner, base_url = await _start_server(handler)
        client = OfficialMtdClient(base_url=base_url)
        _seed_exemplars(client)
        try:
            result = await client.decode_snapshot(
                request_id="session/1/1/1/final",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert captured["file"].startswith(b"RIFF")
        assert captured["file"][8:12] == b"WAVE"
        assert captured["model"] == "mtd-test"
        assert captured["response_format"] == "verbose_json"
        prefix = "[0.00][S01]我是甲[1.00] [1.50][S02]我是乙[2.50]"
        assert "<|audio_pad|>" in captured["prompt"]
        assert captured["prompt"].count("<|audio_pad|>") == 1
        assert captured["prompt"].endswith("<|im_start|>assistant\n" + prefix)
        assert result.raw_text == prefix + " [3.50][S02]继续说话[4.30]"
        assert result.segments == [
            {
                "start_s": 0.0,
                "end_s": 0.8,
                "speaker_id": "S02",
                "text": "继续说话",
            }
        ]
        assert result.metrics["registration_mode"] == "fixed_decoder_prefix"
        assert result.metrics["generated_suffix"] == "[3.50][S02]继续说话[4.30]"
        assert result.metrics["remote_request_id"] == "transcription-server-id"

    asyncio.run(scenario())


def test_official_runtime_is_selected_when_model_listing_is_unavailable() -> None:
    """A missing model-list route selects the official runtime protocol."""

    async def scenario() -> None:
        captured: dict[str, Any] = {}

        async def handler(request: web.Request) -> web.Response:
            captured.update(await _read_form(request))
            return web.json_response(
                {
                    "raw_text": "[0.00][S01]你好[0.50]",
                    "current_segments": [
                        {
                            "start_s": 0.0,
                            "end_s": 0.5,
                            "speaker_id": "S01",
                            "text": "你好",
                        }
                    ],
                    "latency_ms": 2.5,
                    "metrics": {"engine": "async_llm"},
                }
            )

        runner, base_url = await _start_server(
            handler,
            route="/v1/mtd/decode",
            model_name=None,
        )
        client = OfficialMtdClient(base_url=base_url)
        try:
            result = await client.decode_snapshot(
                request_id="official-final",
                pcm16=b"\0\0" * 8000,
                sample_rate=16000,
                is_final=True,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert captured["request_id"] == "official-final"
        assert captured["is_final"] == "true"
        assert captured["max_tokens"] == "2048"
        assert result.segments[0]["speaker_id"] == "S01"
        assert result.metrics["engine"] == "async_llm"

    asyncio.run(scenario())


def test_fixed_prefix_does_not_apply_exemplar_slot_label_mapping() -> None:
    """A generated S01 is retained even when prefix audio contains S01/S02."""

    async def scenario() -> None:
        async def handler(_request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "text": "[3.50][S01]当前说话人[4.20]",
                    "segments": [
                        {"id": 0, "start": 3.5, "end": 4.2, "text": "[S01]当前说话人"},
                    ],
                }
            )

        runner, base_url = await _start_server(handler)
        client = OfficialMtdClient(base_url=base_url)
        _seed_exemplars(client)
        try:
            result = await client.decode_snapshot(
                request_id="fixed-label",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert [item["speaker_id"] for item in result.segments] == ["S01"]

    asyncio.run(scenario())


def test_empty_pool_preserves_compact_local_speaker_order() -> None:
    """Cold-start local labels become compact global IDs without UNKNOWN."""

    async def scenario() -> None:
        captured: dict[str, Any] = {}

        async def handler(request: web.Request) -> web.Response:
            captured.update(await _read_form(request))
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
        client = OfficialMtdClient(base_url=base_url)
        try:
            result = await client.decode_snapshot(
                request_id="cold-start",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert [item["speaker_id"] for item in result.segments] == [
            "S01",
            "S02",
        ]
        assert captured["prompt"].endswith("<|im_start|>assistant\n")
        assert "<|audio_pad|>" in captured["prompt"]

    asyncio.run(scenario())


def test_timestamp_parser_accepts_nested_speaker_bracket() -> None:
    """Continuation-boundary ``[[Sxx]`` is parsed as the intended speaker."""

    assert _parse_timestamped_text("[0.00][S01]甲[1.00] [1.50][[S02]乙[2.50]") == [
        _parse_timestamped_text("[0.00][S01]甲[1.00]")[0],
        _parse_timestamped_text("[1.50][S02]乙[2.50]")[0],
    ]


def test_suffix_without_start_timestamp_reuses_prefix_boundary() -> None:
    """A speaker-only continuation keeps both sides of the shared boundary."""

    assert _join_decoder_prefix_and_suffix(
        "[0.00][S01]你好，我叫张三。[2.10]",
        "[S02]你好，我叫李四。[5.64]",
    ) == ("[0.00][S01]你好，我叫张三。[2.10] " "[2.10][S02]你好，我叫李四。[5.64]")
    parsed = _parse_timestamped_text(
        _join_decoder_prefix_and_suffix(
            "[0.00][S01]你好，我叫张三。[2.10]",
            "[S02]你好，我叫李四。[5.64]",
        )
    )
    assert [item.speaker_id for item in parsed] == ["S01", "S02"]
    assert [(item.start_s, item.end_s) for item in parsed] == [
        (0.0, 2.1),
        (2.1, 5.64),
    ]


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
        client = OfficialMtdClient(base_url=base_url, request_timeout_s=60.0)
        decode_task = asyncio.create_task(
            client.decode_snapshot(
                request_id="cancel-me",
                pcm16=b"\0\0" * 1600,
                sample_rate=16000,
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
            except MtdRequestCancelled:
                pass
            assert time.perf_counter() - cancelled_at < 1.0
        finally:
            release_handler.set()
            await client.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_model_loader_discovers_unified_client() -> None:
    """Configuration discovery can instantiate the unified client type."""

    client = init_registered_model(
        slot="speaker_diarization",
        model_config={
            "type": "OfficialMtdClient",
            "params": {"base_url": "http://127.0.0.1:18714"},
        },
    )
    assert isinstance(client, OfficialMtdClient)
