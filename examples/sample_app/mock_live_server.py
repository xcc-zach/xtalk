"""Deterministic local Live fixture for single-call browser acceptance."""

import argparse
import asyncio
import base64
import math
import struct
import time

from aiohttp import web


def create_app() -> web.Application:
    """Create a loopback fixture with controllable startup and failure scenarios.

    Returns
    -------
    web.Application
        WebSocket protocol fixture plus local-only test control and metrics.
    """
    app = web.Application()
    settings = {"startup_delay_s": 0, "scenario": "burst"}
    calls: list[dict] = []

    async def configure(request):
        settings.update(await request.json())
        return web.json_response(settings)

    async def metrics(request):
        return web.json_response({"calls": calls})

    async def session(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        call = {
            "rate": 0,
            "frames": 0,
            "nonzero_frames": 0,
            "zero_frames": 0,
            "input_samples": 0,
            "closed": False,
            "first_nonzero_at": None,
            "started_at": None,
            "output_samples": 0,
        }
        calls.append(call)
        started = replied = False
        scenario = settings["scenario"]
        async for packet in socket:
            if packet.type != web.WSMsgType.TEXT:
                continue
            event = packet.json()
            kind = event["type"]
            if not started:
                if kind != "session.start":
                    await socket.close(code=1008)
                    break
                await asyncio.sleep(settings["startup_delay_s"])
                if (
                    socket.closed
                    or request.transport is None
                    or request.transport.is_closing()
                ):
                    break
                started = True
                call["rate"] = event["session"]["audio"]["format"]["rate"]
                call["started_at"] = time.monotonic()
                await socket.send_json(
                    {"type": "session.started", "session": {"id": f"mock-{len(calls)}"}}
                )
            elif kind == "session.input_audio.append":
                pcm = base64.b64decode(event["audio"])
                call["frames"] += 1
                call["input_samples"] += len(pcm) // 2
                call["nonzero_frames" if any(pcm) else "zero_frames"] += 1
                if not any(pcm) or replied:
                    continue
                replied = True
                call["first_nonzero_at"] = time.monotonic()
                if scenario == "disconnect":
                    await socket.close()
                    break
                for speaker, delta in (
                    ("input", " 测试输入"),
                    ("output", "本地协议测试成功。"),
                ):
                    await socket.send_json(
                        {
                            "type": f"session.{speaker}_transcript.delta",
                            "delta": delta,
                            "event_id": speaker,
                            "start_ms": 0,
                            "end_ms": 200,
                        }
                    )
                rate = call["rate"]
                tone = b"".join(
                    struct.pack(
                        "<h", int(3000 * math.sin(2 * math.pi * 440 * i / rate))
                    )
                    for i in range(rate // 10)
                )
                for _ in range(60 if scenario == "congestion" else 6):
                    call["output_samples"] += len(tone) // 2
                    await socket.send_json(
                        {
                            "type": "session.output_audio.delta",
                            "delta": base64.b64encode(tone).decode(),
                        }
                    )
            elif kind in {"session.input_audio.mute", "session.input_audio.unmute"}:
                if scenario == "command_rejection":
                    await socket.send_json(
                        {
                            "type": "error",
                            "error": {
                                "code": "command_rejected",
                                "client_event_id": event["event_id"],
                            },
                        }
                    )
                else:
                    await socket.send_json(
                        {
                            "type": "session.input_audio.muted"
                            if kind.endswith(".mute")
                            else "session.input_audio.unmuted",
                            "client_event_id": event["event_id"],
                        }
                    )
            elif kind == "session.close":
                call["closed"] = True
                await socket.send_json(
                    {"type": "session.closed", "usage": {"test_fixture": True}}
                )
                break
        await socket.close()
        return socket

    app.router.add_get("/v1/live/sessions", session)
    app.router.add_post("/fixture", configure)
    app.router.add_get("/fixture", metrics)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    web.run_app(create_app(), host="127.0.0.1", port=args.port)
