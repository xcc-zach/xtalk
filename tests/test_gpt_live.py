"""Single-call protocol, playback backpressure and lifecycle acceptance."""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiohttp import web
from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.sample_app.host.persistence import PersistenceStore
from xtalk import Xtalk
from xtalk.models.end_to_end.gpt_live import GPTLive
from xtalk.serving.modules.end_to_end_manager import _AudioPlaybackBudget

DELEGATION = {
    "type": "responses",
    "responses": {"model": "fixture", "tools": [], "tool_choice": "none"},
}


class LiveFixture(unittest.IsolatedAsyncioTestCase):
    """Exercise the actual adapter and authenticated application over sockets."""

    async def asyncSetUp(self):
        self.commands = []
        self.connections = 0
        self.scenario = "normal"
        app = web.Application()
        app.router.add_get("/v1/live/sessions", self.upstream)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"ws://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1/live/sessions"
        self.addAsyncCleanup(self.runner.cleanup)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    async def upstream(self, request):
        self.assertEqual(request.headers["Authorization"], "Bearer local-test")
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        start = await ws.receive_json()
        self.commands.append(start)
        self.assertEqual(start["type"], "session.start")
        if self.scenario == "start_timeout":
            await ws.receive()
            return ws
        await ws.send_json(
            {"type": "session.started", "session": {"id": str(self.connections)}}
        )
        if self.scenario == "bad_base64":
            await ws.send_json({"type": "session.output_audio.delta", "delta": "!bad!"})
        elif self.scenario == "slow_consumer":
            await ws.send_json(
                {
                    "type": "session.output_audio.delta",
                    "delta": base64.b64encode(bytes(24000 * 2 * 6)).decode(),
                }
            )
        elif self.scenario == "backend_degraded":
            await ws.send_json(
                {"type": "error", "error": {"code": "responses_validation_failed"}}
            )
        async for packet in ws:
            if packet.type != web.WSMsgType.TEXT:
                continue
            event = packet.json()
            self.commands.append(event)
            if event["type"] == "session.input_audio.append":
                delta = {
                    "type": "session.input_transcript.delta",
                    "event_id": "input",
                    "delta": " 你好 你好 ",
                    "start_ms": 0,
                    "end_ms": 20,
                }
                await ws.send_json(delta)
                await ws.send_json(delta)
                for _ in range(6 if self.scenario == "burst" else 1):
                    await ws.send_json(
                        {"type": "session.output_audio.delta", "delta": event["audio"]}
                    )
                if self.scenario == "burst":
                    await ws.send_json(
                        {
                            "type": "session.output_transcript.delta",
                            "event_id": "late",
                            "delta": "late caption",
                            "start_ms": 100,
                            "end_ms": 200,
                        }
                    )
                    await ws.send_json(
                        {"type": "error", "error": {"code": "audio_interrupted"}}
                    )
            elif event["type"] in {
                "session.input_audio.mute",
                "session.input_audio.unmute",
            }:
                await ws.send_json(
                    {
                        "type": "session.input_audio.muted",
                        "client_event_id": event["event_id"],
                    }
                )
            elif event["type"] == "session.close":
                if self.scenario == "close_timeout":
                    continue
                await ws.send_json(
                    {
                        "type": "session.output_transcript.delta",
                        "event_id": "last",
                        "delta": " 结束。 ",
                        "start_ms": 10,
                        "end_ms": 30,
                    }
                )
                await ws.send_json(
                    {
                        "type": "session.closed",
                        "reason": "close_requested",
                        "usage": {"seconds": 1},
                    }
                )
                break
        await ws.close()
        return ws

    def adapter(self, **kwargs):
        return GPTLive(
            api_key="local-test", base_url=self.url, delegation=DELEGATION, **kwargs
        )


class LiveProtocolTests(LiveFixture):
    """Validate provider protocol, bounded resources and session isolation."""

    async def test_adapter_start_audio_controls_close_and_isolation(self):
        first = self.adapter()
        second = first.clone()
        await asyncio.gather(first.open(), second.open())
        self.addAsyncCleanup(second.close)
        self.addAsyncCleanup(first.close)
        await first.append_audio(b"\x01\x02" * 240)
        await first.set_input_muted(True)
        kinds = []
        async for event in first.events():
            kinds.append(event.kind)
            if event.kind == "audio":
                self.assertEqual(event.audio, b"\x01\x02" * 240)
            if event.kind == "control_ack":
                break
        result = await first.close()
        self.assertEqual(result.finalization, "complete")
        self.assertEqual(result.voice_usage, {"seconds": 1})
        self.assertEqual(result.reason, "close_requested")
        await second.append_audio(b"\0\0" * 240)
        self.assertIn("audio", kinds)
        self.assertEqual(self.connections, 2)
        self.assertEqual(await first.close(), result)

    async def test_authenticated_single_call_without_history(self):
        from contextlib import asynccontextmanager

        from asgi_lifespan import LifespanManager

        from examples.sample_app.host.application import ApplicationHost

        config = {
            "end_to_end": {
                "type": "GPTLive",
                "params": {
                    "api_key": "local-test",
                    "base_url": self.url,
                    "delegation": DELEGATION,
                },
            },
            "service_config": {"enable_persistence": False},
        }

        def case():
            instance = Xtalk.from_config(config)

            @asynccontextmanager
            async def lifespan(app):
                async with LifespanManager(instance.create_asgi_app()):
                    yield

            app = FastAPI(lifespan=lifespan)
            host = ApplicationHost(instance, data_dir=self.directory.name)
            host.mount_routes(app)
            with TestClient(app) as client:
                token = client.post("/api/auth/login").json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                for _ in range(2):
                    with client.websocket_connect(
                        f"/v1/realtime?access_token={token}"
                    ) as ws:
                        created = ws.receive_json()
                        self.assertEqual(created["type"], "session.created")
                        profile = created["session"]["xtalk"]["end_to_end"]
                        stream = profile["stream_id"]
                        self.assertEqual(profile["protocol_version"], 2)
                        ws.send_json(
                            {
                                "type": "session.update",
                                "session": {"xtalk": {"end_to_end_protocol": 2}},
                            }
                        )
                        self.assertEqual(ws.receive_json()["type"], "session.updated")
                        ws.send_json(
                            {"type": "xtalk.audio_stream.ready", "stream_id": stream}
                        )
                        self.assertEqual(
                            ws.receive_json()["type"], "xtalk.audio_stream.started"
                        )
                        ws.send_bytes(b"\1\0" * 240)
                        transcript = ws.receive_json()
                        self.assertEqual(
                            transcript["type"], "xtalk.transcript.delta", transcript
                        )
                        self.assertEqual(transcript["delta"], " 你好 你好 ")
                        audio = ws.receive_json()
                        self.assertEqual(audio["type"], "xtalk.audio_stream.delta")
                        self.assertEqual(
                            base64.b64decode(audio["delta"]), b"\1\0" * 240
                        )
                        ws.send_json(
                            {
                                "type": "xtalk.audio_stream.progress",
                                "stream_id": stream,
                                "played_samples": 240,
                            }
                        )
                        ws.send_json(
                            {"type": "xtalk.audio_stream.close", "stream_id": stream}
                        )
                        end = ws.receive_json()
                        self.assertEqual(end["type"], "xtalk.audio_stream.ended")
                        self.assertEqual(end["finalization"], "complete")
                    self.assertEqual(
                        client.get("/api/sessions", headers=headers).json()["sessions"],
                        [],
                    )

        await asyncio.wait_for(asyncio.to_thread(case), 15)

    async def test_realtime_binary_16k_and_capability_errors(self):
        """Negotiate the new envelope and reject unsupported controls without ending audio."""
        config = {
            "end_to_end": {
                "type": "GPTLive",
                "params": {
                    "api_key": "local-test",
                    "base_url": self.url,
                    "delegation": DELEGATION,
                    "sample_rate": 16000,
                },
            }
        }

        def case():
            runtime = Xtalk.from_config(config)
            with (
                TestClient(runtime.create_asgi_app()) as client,
                client.websocket_connect("/v1/realtime?model=xtalk") as ws,
            ):
                created = ws.receive_json()
                profile = created["session"]["xtalk"]["end_to_end"]
                stream = profile["stream_id"]
                self.assertEqual(profile["media"]["sample_rate"], 16000)
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "xtalk": {
                                "end_to_end_protocol": 2,
                                "output_audio_encoding": "binary_pcm",
                            }
                        },
                    }
                )
                self.assertEqual(ws.receive_json()["type"], "session.updated")
                ws.send_json(
                    {"type": "input_audio_buffer.commit", "event_id": "unsupported"}
                )
                error = ws.receive_json()
                self.assertEqual(error["error"]["code"], "unsupported_capability")
                self.assertEqual(error["error"]["event_id"], "unsupported")
                ws.send_json({"type": "xtalk.audio_stream.ready", "stream_id": stream})
                self.assertEqual(
                    ws.receive_json()["type"], "xtalk.audio_stream.started"
                )
                pcm = b"\1\0" * 160
                ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode(),
                    }
                )
                self.assertEqual(ws.receive_json()["type"], "xtalk.transcript.delta")
                payload = ws.receive_bytes()
                length = int.from_bytes(payload[:4], "big")
                header = json.loads(payload[4 : 4 + length])
                self.assertEqual(header["version"], 1)
                self.assertEqual(header["type"], "xtalk.audio_stream.delta")
                self.assertEqual(header["stream_id"], stream)
                self.assertEqual(payload[4 + length :], pcm)
                ws.send_json(
                    {
                        "type": "xtalk.audio_stream.progress",
                        "stream_id": stream,
                        "played_samples": 160,
                    }
                )
                ws.send_json({"type": "xtalk.audio_stream.close", "stream_id": stream})
                self.assertEqual(ws.receive_json()["finalization"], "complete")

        await asyncio.wait_for(asyncio.to_thread(case), 15)
        self.assertTrue(
            all(
                "input" not in event["session"]
                for event in self.commands
                if event["type"] == "session.start"
            )
        )

    async def test_budget_burst_waits_and_ack_before_return(self):
        budget = _AudioPlaybackBudget(stall_timeout_s=1)
        await budget.reset("call")
        sent = []

        async def deliver():
            for i in range(6):
                await budget.reserve(100)
                sent.append(i)

        sender = asyncio.create_task(deliver())
        await asyncio.sleep(0.02)
        self.assertEqual(sent, [0, 1, 2])
        self.assertFalse(sender.done())
        self.assertFalse(await budget.acknowledge_samples("old", 2400, 24000))
        self.assertFalse(await budget.acknowledge_samples("call", 7201, 24000))
        for played in (2400, 4800, 7200):
            self.assertTrue(await budget.acknowledge_samples("call", played, 24000))
            await asyncio.sleep(0)
        await asyncio.wait_for(sender, 1)
        self.assertEqual(sent, list(range(6)))
        await budget.acknowledge_samples("call", 14400, 24000)
        self.assertEqual(budget.outstanding_ms, 0)
        token = await budget.reserve(100)
        await budget.acknowledge_samples("call", 16800, 24000)  # ack during publish
        await budget.rollback(token)
        self.assertEqual(budget.outstanding_ms, 0)

    async def test_budget_reset_stop_and_failed_send(self):
        budget = _AudioPlaybackBudget()
        await budget.reset("old")
        tokens = [await budget.reserve(100) for _ in range(3)]
        blocked = asyncio.create_task(budget.reserve(100))
        await asyncio.sleep(0)
        await budget.reset("new")
        with self.assertRaises(asyncio.CancelledError):
            await blocked
        token = await budget.reserve(100)
        await budget.rollback(tokens[0])
        self.assertEqual(budget.outstanding_ms, 100)
        await budget.rollback(token)
        self.assertEqual(budget.outstanding_ms, 0)
        for _ in range(3):
            await budget.reserve(100)
        blocked = asyncio.create_task(budget.reserve(100))
        await asyncio.sleep(0)
        await budget.stop()
        with self.assertRaises(asyncio.CancelledError):
            await blocked

    async def test_budget_stall_is_bounded(self):
        budget = _AudioPlaybackBudget(stall_timeout_s=0.02)
        for _ in range(3):
            await budget.reserve(100)
        with self.assertRaises(asyncio.TimeoutError):
            await budget.reserve(100)

    async def test_malformed_and_over_budget_output_fail_without_leaked_tasks(self):
        for scenario in ("bad_base64", "slow_consumer"):
            with self.subTest(scenario=scenario):
                self.scenario = scenario
                adapter = self.adapter()
                try:
                    await adapter.open()
                except RuntimeError:
                    pass  # A fast provider may report the error during startup.
                else:
                    event = await asyncio.wait_for(anext(adapter.events()), 1)
                    self.assertEqual(event.kind, "error")
                    self.assertTrue(event.data["fatal"])
                await adapter.close()
                self.assertTrue(adapter._reader.done())
                self.assertTrue(adapter._writer.done())

    async def test_start_timeout_is_not_started_and_close_timeout_is_incomplete(self):
        self.scenario = "start_timeout"
        adapter = self.adapter(connect_timeout_s=0.05)
        with self.assertRaises(asyncio.TimeoutError):
            await adapter.open()
        self.assertEqual((await adapter.close()).finalization, "not_started")
        self.scenario = "close_timeout"
        adapter = self.adapter(close_timeout_s=0.03)
        await adapter.open()
        self.assertEqual((await adapter.close()).finalization, "incomplete")
        self.assertTrue(adapter._reader.done())

    async def test_backend_validation_keeps_voice_session_active(self):
        self.scenario = "backend_degraded"
        adapter = self.adapter()
        await adapter.open()
        event = await asyncio.wait_for(anext(adapter.events()), 1)
        self.assertFalse(event.data["fatal"])
        await adapter.append_audio(bytes(480))
        self.assertEqual((await adapter.close()).finalization, "complete")

    async def test_api_key_comes_from_config_even_when_environment_is_set(self):
        adapter = self.adapter()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "unused-environment-key"}):
            try:
                await adapter.open()
                self.assertEqual((await adapter.close()).finalization, "complete")
            finally:
                await adapter.close()

    async def test_missing_config_key_does_not_fall_back_to_environment(self):
        for key in (None, "", "  "):
            with self.subTest(key=key):
                adapter = GPTLive(api_key=key, base_url=self.url, delegation=DELEGATION)
                with (
                    patch.dict(
                        "os.environ", {"OPENAI_API_KEY": "unused-environment-key"}
                    ),
                    self.assertRaisesRegex(ValueError, "end_to_end.params.api_key"),
                ):
                    await adapter.open()
                self.assertIsNone(adapter._session)
        self.assertEqual(self.connections, 0)

    async def test_registration_graph_and_history_shapes(self):
        from xtalk.models import ASR, EndToEnd, Models
        from xtalk.serving.modules.end_to_end_manager import EndToEndManager
        from xtalk.serving.service import DefaultService

        config = {
            "end_to_end": {
                "type": "GPTLive",
                "params": {
                    "api_key": "local-test",
                    "base_url": self.url,
                    "delegation": DELEGATION,
                },
            },
            "service_config": {"enable_persistence": False},
        }
        models = Xtalk.create_models_from_config(config_path_or_dict=config)
        self.assertIsInstance(models.require(EndToEnd), GPTLive)
        prototype = Xtalk.configure(config).build()._service_manager._service_prototype
        self.assertEqual(prototype._manager_classes, [EndToEndManager])
        custom = DefaultService(models=models, manager_classes=[])
        self.assertEqual(custom._manager_classes, [])
        adapter = models.require(EndToEnd)
        await adapter.open()
        self.assertNotIn("input", self.commands[-1]["session"])
        await adapter.close()
        with self.assertRaises(ValueError):
            DefaultService(models=Models({EndToEnd: self.adapter(), ASR: object()}))

    async def test_odd_and_oversized_input_frames_close_service(self):
        from xtalk.models import EndToEnd, Models
        from xtalk.serving.events import AudioFrameReceived
        from xtalk.serving.service import DefaultService

        manager_service = DefaultService(
            models=Models({EndToEnd: self.adapter()}), _websocket=AsyncMock()
        )
        manager = manager_service._managers[0]
        manager._active = True
        await manager._on_audio(
            AudioFrameReceived(
                session_id=manager_service.session_id,
                audio_data=b"odd",
                sample_rate=24000,
            )
        )
        self.assertTrue(manager._stop.is_set())
        await manager_service.stop()

    async def test_missing_ready_closes_without_provider(self):
        from xtalk.models import EndToEnd, Models
        from xtalk.serving.service import DefaultService

        socket = AsyncMock()
        socket.client_state.value = 1
        service = DefaultService(
            models=Models({EndToEnd: self.adapter()}),
            _websocket=socket,
            service_config={"end_to_end": {"client_ready_timeout_s": 0.02}},
        )
        from xtalk.serving.event_bus import EventDispatchMode
        from xtalk.serving.events import ServiceRuntimeStarted

        await service.event_bus.publish(
            ServiceRuntimeStarted(session_id=service.session_id, mode="end_to_end"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        await asyncio.sleep(0.05)
        await service.stop()
        self.assertEqual(self.connections, 0)

    async def test_close_discards_queued_recording_and_keeps_reader(self):
        model = self.adapter()
        await model.open()
        for _ in range(20):
            await model.append_audio(bytes(4800))
        result = await asyncio.wait_for(model.close(), 0.5)
        self.assertEqual(result.finalization, "complete")
        self.assertEqual(
            [event["type"] for event in self.commands],
            ["session.start", "session.close"],
        )
        self.assertEqual(model._write_audio_bytes, 0)

    async def test_protocol_one_is_rejected_with_rebuild_hint(self):
        from xtalk.models import EndToEnd, Models
        from xtalk.serving.service import DefaultService

        socket = AsyncMock()
        service = DefaultService(
            models=Models({EndToEnd: self.adapter()}), _websocket=socket
        )
        self.addAsyncCleanup(service.stop)
        await service.input_gateway._handle_text_message(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"xtalk": {"end_to_end_protocol": 1}},
                }
            )
        )
        event = socket.send_json.call_args.args[0]
        self.assertEqual(event["type"], "error")
        self.assertFalse(service.service_config.get("_live_protocol_negotiated", False))
        self.assertEqual(self.connections, 0)


class CompatibilityTests(unittest.TestCase):
    """Keep existing data intact and reject unsupported features before startup."""

    def test_old_database_survives(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.db"
            store = PersistenceStore(path)
            session = store.create_session("u")["session_id"]
            store.append_message(
                user_id="u", session_id=session, role="user", content="cascade"
            )
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE end_to_end_streams (payload TEXT)")
                db.execute("INSERT INTO end_to_end_streams VALUES ('old Live data')")
            store = PersistenceStore(path)
            self.assertEqual(
                store.get_session_detail("u", session)["messages"][0]["content"],
                "cascade",
            )
            with sqlite3.connect(path) as db:
                self.assertEqual(
                    db.execute("SELECT payload FROM end_to_end_streams").fetchone()[0],
                    "old Live data",
                )

    def test_sample_rate_and_unsupported_persistence(self):
        for rate in (16000, 24000):
            model = GPTLive(delegation=DELEGATION, sample_rate=rate)
            self.assertEqual(model.clone().audio_profile.sample_rate, rate)
        from xtalk.models import EndToEnd, Models
        from xtalk.serving.service import DefaultService

        with self.assertRaisesRegex(ValueError, "enable_persistence"):
            DefaultService(
                models=Models({EndToEnd: GPTLive(delegation=DELEGATION)}),
                service_config={"enable_persistence": True},
            )


class LiveManagerTests(LiveFixture):
    """Exercise production/consumption separation against the actual adapter."""

    async def start_manager(self, policy=None, sample_rate=24000, output_handler=None):
        from xtalk.models import EndToEnd, Models
        from xtalk.serving.event_bus import EventBus, EventDispatchMode
        from xtalk.serving.events import (
            AudioStreamReady,
            EndToEndAudioChunkReady,
            EndToEndSignal,
            EndToEndTranscriptDelta,
            ServiceRuntimeStarted,
        )
        from xtalk.serving.modules.end_to_end_manager import (
            EndToEndManager,
            create_stream_profile,
        )

        bus = EventBus()
        model = self.adapter(sample_rate=sample_rate)
        profile = create_stream_profile(model, {"end_to_end": policy or {}})
        manager = EndToEndManager(
            bus, "call", Models({EndToEnd: model}), {"_end_to_end_profile": profile}
        )
        signals, audio, transcripts = [], [], []

        async def signal(event):
            signals.append((event.action, event.data))

        async def output(event):
            audio.append(event.audio_data)

        async def transcript(event):
            transcripts.append(event)

        bus.subscribe(EndToEndSignal, signal)
        bus.subscribe(EndToEndAudioChunkReady, output_handler or output)
        bus.subscribe(EndToEndTranscriptDelta, transcript)
        self.addAsyncCleanup(bus.shutdown)
        self.addAsyncCleanup(manager.shutdown)
        await bus.publish(
            ServiceRuntimeStarted(session_id="call", mode="end_to_end"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        await bus.publish(
            AudioStreamReady(session_id="call", stream_id=profile["stream_id"]),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        await self.wait_until(
            lambda: any(action == "audio_stream_start" for action, _ in signals)
        )
        return manager, signals, audio, transcripts

    async def wait_until(self, predicate):
        for _ in range(200):
            if predicate():
                return
            await asyncio.sleep(0.005)
        self.fail("Asynchronous behavior did not complete")

    async def test_six_output_blocks_wait_and_drain_in_order(self):
        from xtalk.serving.events import AudioFrameReceived, AudioStreamControlRequested

        self.scenario = "burst"
        manager, signals, audio, transcripts = await self.start_manager()
        pcm = b"\1\0" * 2400
        await manager._on_audio(
            AudioFrameReceived(session_id="call", audio_data=pcm, sample_rate=24000)
        )
        await self.wait_until(lambda: len(audio) == 3)
        await asyncio.sleep(0.03)
        self.assertEqual(len(audio), 3)
        self.assertTrue(manager._active)
        self.assertEqual(
            len(transcripts), 2
        )  # duplicate removed, late event remains responsive
        self.assertTrue(
            any(
                action == "end_to_end_error" and not data["fatal"]
                for action, data in signals
            )
        )
        for played in (2400, 4800, 7200):
            await manager._on_control(
                AudioStreamControlRequested(
                    session_id="call",
                    stream_id=manager.stream_id,
                    command="audio_stream_progress",
                    data={"played_samples": played},
                )
            )
            await asyncio.sleep(0.01)
        await self.wait_until(lambda: len(audio) == 6)
        self.assertEqual(audio, [pcm] * 6)
        await asyncio.wait_for(manager.shutdown(), 0.5)
        self.assertEqual(signals[-1][0], "audio_stream_end")
        self.assertEqual(signals[-1][1]["finalization"], "complete")
        self.assertEqual(manager._pending_samples, 0)

    async def test_failed_custom_output_ends_call_without_playback_timeout(self):
        from xtalk.serving.events import AudioFrameReceived

        async def failing_output(event):
            raise OSError("fixture output failed")

        manager, signals, _, _ = await self.start_manager(output_handler=failing_output)
        await manager._on_audio(
            AudioFrameReceived(
                session_id="call",
                audio_data=bytes(4800),
                sample_rate=24000,
            )
        )
        await self.wait_until(lambda: manager._task.done())
        self.assertEqual(manager._budget.outstanding_ms, 0)
        self.assertTrue(manager._failed)
        errors = [
            data["code"] for action, data in signals if action == "end_to_end_error"
        ]
        self.assertEqual(errors, ["session_failed"])
        self.assertEqual(signals[-1][1]["outcome"], "failed")

    async def test_stall_with_outstanding_audio_is_visible_and_cancellable(self):
        from xtalk.serving.events import AudioFrameReceived

        manager, signals, _, _ = await self.start_manager(
            {"playback_stall_timeout_s": 0.04}
        )
        await manager._on_audio(
            AudioFrameReceived(
                session_id="call", audio_data=bytes(480), sample_rate=24000
            )
        )
        await self.wait_until(
            lambda: any(action == "audio_stream_end" for action, _ in signals)
        )
        self.assertTrue(
            any(
                action == "end_to_end_error" and data["code"] == "playback_stalled"
                for action, data in signals
            )
        )
        self.assertTrue(manager._task.done())

    async def test_16k_audio_uses_negotiated_rate(self):
        from xtalk.serving.events import AudioFrameReceived

        manager, _, audio, _ = await self.start_manager(sample_rate=16000)
        await manager._on_audio(
            AudioFrameReceived(
                session_id="call", audio_data=bytes(320), sample_rate=16000
            )
        )
        await self.wait_until(lambda: len(audio) == 1)
        self.assertEqual(self.commands[0]["session"]["audio"]["format"]["rate"], 16000)
        await manager.shutdown()
