"""Concurrency tests for the headless AsyncLLM MTD runtime."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web


_SERVER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "mtd_runtime" / "server.py"
)
_SPEC = importlib.util.spec_from_file_location("mtd_runtime_server", _SERVER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)


class _FakeEngine:
    """Record AsyncLLM abort calls made by the runtime."""

    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def abort(self, request_id: str) -> None:
        """Record one aborted request ID."""

        self.aborted.append(request_id)

    def shutdown(self) -> None:
        """Match the AsyncLLM lifecycle API used by the runtime."""


def _runtime(fake_engine: _FakeEngine) -> Any:
    """Create a runtime instance without importing or loading vLLM."""

    runtime = object.__new__(server.OfficialMtdRuntime)
    runtime.sample_rate = 16000
    runtime.default_instruction = "test"
    runtime.engine = fake_engine
    runtime.model_arch = "FakeMtd"
    runtime._cancelled = set()
    runtime._active_requests = set()
    runtime._active_tasks = {}
    runtime._max_active_requests = 0
    runtime._closed = False
    return runtime


def _fields(request_id: str) -> dict[str, Any]:
    """Return one minimal valid runtime request body."""

    return {
        "request_id": request_id,
        "sample_rate": "16000",
        "audio": b"\0\0" * 160,
        "current_audio_seconds": "0.01",
        "max_tokens": "8",
    }


def _result(request_id: str) -> dict[str, Any]:
    """Return one deterministic timestamped fake engine result."""

    return {
        "raw_text": f"[0.00][S01]{request_id}[0.01]",
        "latency_ms": 1.0,
        "metrics": {},
    }


def test_decode_requests_overlap_without_global_lock() -> None:
    """Two HTTP-level decodes can enter the engine concurrently."""

    async def scenario() -> None:
        fake_engine = _FakeEngine()
        runtime = _runtime(fake_engine)
        both_started = asyncio.Event()

        async def decode_engine(**kwargs: Any) -> dict[str, Any]:
            if len(runtime._active_requests) >= 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
            return _result(str(kwargs["request_id"]))

        runtime._decode_engine = decode_engine
        first, second = await asyncio.gather(
            runtime.decode(_fields("request-a")),
            runtime.decode(_fields("request-b")),
        )

        assert first["request_id"] == "request-a"
        assert second["request_id"] == "request-b"
        assert runtime._max_active_requests == 2
        assert first["metrics"]["engine"] == "async_llm"
        assert second["metrics"]["max_active_requests"] == 2
        assert not runtime._active_requests

    asyncio.run(scenario())


def test_cancel_aborts_engine_and_unblocks_decode() -> None:
    """Cancel propagates to AsyncLLM and terminates the waiting decode."""

    async def scenario() -> None:
        fake_engine = _FakeEngine()
        runtime = _runtime(fake_engine)
        started = asyncio.Event()

        async def decode_engine(**_kwargs: Any) -> dict[str, Any]:
            started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

        runtime._decode_engine = decode_engine
        decode_task = asyncio.create_task(runtime.decode(_fields("cancel-me")))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await runtime.cancel("cancel-me")

        with pytest.raises(web.HTTPConflict) as raised:
            await asyncio.wait_for(decode_task, timeout=1.0)
        assert raised.value.text == "request cancelled"
        assert fake_engine.aborted == ["cancel-me"]
        assert not runtime._active_requests
        assert "cancel-me" not in runtime._active_tasks

    asyncio.run(scenario())
