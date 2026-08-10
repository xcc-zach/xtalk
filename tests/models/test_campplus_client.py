"""Tests for the session-local CAM++ embedding client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

import numpy as np
from aiohttp import web

from xtalk.model_loader import init_registered_model
from xtalk.models.speaker_diarization import CampPlusDiarization
from xtalk.models.speaker_diarization.campplus import (
    CampPlusRequestCancelled,
    _SpeakerProfile,
)


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _embedding(index: int) -> list[float]:
    """Return one deterministic 192-dimensional unit embedding."""

    value = np.zeros(192, dtype=np.float32)
    value[index] = 1.0
    return value.tolist()


async def _start_server(handler: Handler) -> tuple[web.AppRunner, str]:
    """Start one local ephemeral CAM++ embedding server."""

    app = web.Application()
    app.router.add_post("/v1/speaker/embeddings", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _read_form(request: web.Request) -> dict[str, Any]:
    """Read multipart fields while retaining uploaded PCM bytes."""

    reader = await request.multipart()
    result: dict[str, Any] = {}
    while True:
        part = await reader.next()
        if part is None:
            return result
        if part.filename:
            result[part.name] = await part.read()
        else:
            result[part.name] = await part.text()


def test_request_contract_and_first_speaker_commit_only_on_final() -> None:
    """Send raw PCM and keep the first partial speaker provisional."""

    async def scenario() -> None:
        captured: list[dict[str, Any]] = []

        async def handler(request: web.Request) -> web.Response:
            captured.append(await _read_form(request))
            return web.json_response(
                {
                    "model": "campplus-cn-common",
                    "embedding": _embedding(0),
                    "latency_ms": 2.5,
                    "metrics": {"engine": "onnxruntime"},
                },
                headers={"X-Request-Id": "remote-campplus-id"},
            )

        runner, base_url = await _start_server(handler)
        client = CampPlusDiarization(base_url=base_url)
        try:
            partial = await client.decode_snapshot(
                request_id="session/1/1/1/partial",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
            assert not client._profiles
            final = await client.decode_snapshot(
                request_id="session/1/1/2/final",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=True,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert captured[0] == {
            "request_id": "session/1/1/1/partial",
            "sample_rate": "16000",
            "is_final": "false",
            "audio": b"\0\0" * 16000,
        }
        assert partial.segments == [
            {
                "start_s": 0.0,
                "end_s": 1.0,
                "speaker_id": "S01",
                "text": "",
            }
        ]
        assert partial.metrics["clustering_action"] == "provisional_first_speaker"
        assert final.segments[0]["speaker_id"] == "S01"
        assert final.raw_text == "[0.00][S01][1.00]"
        assert final.metrics["centroid_updated"] is True
        assert final.metrics["committed_speakers"] == 1
        assert final.metrics["remote_request_id"] == "remote-campplus-id"

    asyncio.run(scenario())


def test_new_speaker_partial_requires_confirmation_then_final_commits() -> None:
    """Publish a later speaker only after consecutive partial confirmation."""

    async def scenario() -> None:
        async def handler(request: web.Request) -> web.Response:
            fields = await _read_form(request)
            request_id = str(fields["request_id"])
            embedding = _embedding(0) if "first" in request_id else _embedding(1)
            return web.json_response({"embedding": embedding})

        runner, base_url = await _start_server(handler)
        client = CampPlusDiarization(base_url=base_url)
        try:
            await client.decode_snapshot(
                request_id="session/first/1/1/final",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=True,
            )
            pending = await client.decode_snapshot(
                request_id="session/second/1/1/partial",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
            confirmed = await client.decode_snapshot(
                request_id="session/second/1/2/partial",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
            assert len(client._profiles) == 1
            final = await client.decode_snapshot(
                request_id="session/second/1/3/final",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=True,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert pending.segments == []
        assert pending.metrics["clustering_action"] == "new_speaker_pending"
        assert pending.metrics["confirmations"] == 1
        assert confirmed.segments[0]["speaker_id"] == "S02"
        assert confirmed.metrics["confirmations"] == 2
        assert final.segments[0]["speaker_id"] == "S02"
        assert final.metrics["clustering_action"] == "registered_final"
        assert final.metrics["committed_speakers"] == 2

    asyncio.run(scenario())


def test_close_top_two_scores_still_selects_best_without_margin_rule() -> None:
    """Use only the configured threshold even when two profiles score closely."""

    async def scenario() -> None:
        angle = 0.01
        first = np.zeros(192, dtype=np.float32)
        first[0] = 1.0
        second = np.zeros(192, dtype=np.float32)
        second[0] = np.cos(angle)
        second[1] = np.sin(angle)

        async def handler(_request: web.Request) -> web.Response:
            return web.json_response({"embedding": second.tolist()})

        runner, base_url = await _start_server(handler)
        client = CampPlusDiarization(base_url=base_url)
        client._profiles = [
            _SpeakerProfile("S01", first),
            _SpeakerProfile("S02", second),
        ]
        try:
            result = await client.decode_snapshot(
                request_id="session/close/1/1/partial",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
        finally:
            await client.close()
            await runner.cleanup()

        assert result.segments[0]["speaker_id"] == "S02"
        assert result.metrics["clustering_action"] == "matched_partial"
        assert result.metrics["best_similarity"] > 0.999

    asyncio.run(scenario())


def test_partial_match_does_not_update_centroid_but_final_does() -> None:
    """Mutate a matched centroid only when processing a final snapshot."""

    async def scenario() -> None:
        initial = np.asarray(_embedding(0), dtype=np.float32)
        shifted = initial.copy()
        shifted[0] = 0.9
        shifted[1] = 0.1

        async def handler(request: web.Request) -> web.Response:
            fields = await _read_form(request)
            vector = initial if "register" in fields["request_id"] else shifted
            return web.json_response({"embedding": vector.tolist()})

        runner, base_url = await _start_server(handler)
        client = CampPlusDiarization(
            base_url=base_url,
            centroid_update_alpha=0.5,
        )
        try:
            await client.decode_snapshot(
                request_id="session/register/1/1/final",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=True,
            )
            before = client._profiles[0].embedding.copy()
            await client.decode_snapshot(
                request_id="session/update/1/1/partial",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=False,
            )
            after_partial = client._profiles[0].embedding.copy()
            final = await client.decode_snapshot(
                request_id="session/update/1/2/final",
                pcm16=b"\0\0" * 16000,
                sample_rate=16000,
                is_final=True,
            )
            after_final = client._profiles[0].embedding.copy()
        finally:
            await client.close()
            await runner.cleanup()

        np.testing.assert_array_equal(before, after_partial)
        assert not np.array_equal(before, after_final)
        assert final.metrics["centroid_updated"] is True
        assert client._profiles == []

    asyncio.run(scenario())


def test_too_short_snapshot_returns_unknown_without_network_request() -> None:
    """Fail closed with S00 before contacting the service for short audio."""

    async def scenario() -> None:
        client = CampPlusDiarization(base_url="http://127.0.0.1:1")
        try:
            result = await client.decode_snapshot(
                request_id="short-final",
                pcm16=b"\0\0" * 1600,
                sample_rate=16000,
                is_final=True,
            )
        finally:
            await client.close()

        assert result.segments[0]["speaker_id"] == "S00"
        assert result.metrics["clustering_action"] == "too_short"
        assert result.latency_ms == 0.0

    asyncio.run(scenario())


def test_clone_preserves_configuration_with_fresh_speaker_state() -> None:
    """Keep model settings while isolating all session-local clustering state."""

    client = CampPlusDiarization(
        base_url="http://127.0.0.1:18715/",
        request_timeout_s=7.0,
        similarity_threshold=0.7,
        min_audio_duration_s=0.8,
        new_speaker_confirmations=3,
        centroid_update_alpha=0.2,
        max_speakers=4,
    )
    client._profiles.append(
        _SpeakerProfile("S01", np.asarray(_embedding(0), dtype=np.float32))
    )

    clone = client.clone()

    assert clone.base_url == "http://127.0.0.1:18715"
    assert clone.request_timeout_s == 7.0
    assert clone.similarity_threshold == 0.7
    assert clone.min_audio_duration_s == 0.8
    assert clone.new_speaker_confirmations == 3
    assert clone.centroid_update_alpha == 0.2
    assert clone.max_speakers == 4
    assert clone._profiles == []


def test_cancel_releases_waiting_decode_quickly() -> None:
    """Translate local HTTP task cancellation into a regular model failure."""

    async def scenario() -> None:
        started = asyncio.Event()
        release_handler = asyncio.Event()

        async def handler(_request: web.Request) -> web.Response:
            started.set()
            await release_handler.wait()
            return web.json_response({"embedding": _embedding(0)})

        runner, base_url = await _start_server(handler)
        client = CampPlusDiarization(base_url=base_url, request_timeout_s=60.0)
        decode_task = asyncio.create_task(
            client.decode_snapshot(
                request_id="cancel-me",
                pcm16=b"\0\0" * 16000,
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
            except CampPlusRequestCancelled:
                pass
            assert time.perf_counter() - cancelled_at < 1.0
        finally:
            release_handler.set()
            await client.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_model_loader_discovers_campplus_client() -> None:
    """Instantiate the CAM++ client through configuration discovery."""

    client = init_registered_model(
        slot="speaker_diarization",
        model_config={
            "type": "CampPlusDiarization",
            "params": {"base_url": "http://127.0.0.1:18715"},
        },
    )

    assert isinstance(client, CampPlusDiarization)
