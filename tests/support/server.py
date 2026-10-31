"""Build the mock-backed Xtalk application and serve it for frontend tests.

Run ``python -m tests.support.server`` to start a uvicorn server on a random
port; the first stdout line is a JSON object with the port and WebSocket URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
from typing import Any

from xtalk import Xtalk

from .mock_models import MockLookupTool, MockTimerTool, ScriptedChatModel  # noqa: F401 - registers the mock models

MODEL_ALIAS = "xtalk"


def mock_config(
    *,
    vad: bool = True,
    api_keys: tuple[str, ...] | list[str] = (),
    tools: bool = False,
) -> dict[str, Any]:
    """Return an Xtalk configuration wired to the mock models.

    Parameters
    ----------
    vad : bool, optional
        Whether to configure the server side VAD model.
    api_keys : tuple[str, ...] | list[str], optional
        Realtime API keys; empty disables authentication.
    tools : bool, optional
        Use the real ``DefaultAgent`` with the scripted chat model and mock
        tools instead of ``MockAgent``.
    """
    if tools:
        import xtalk.models.agents.default  # noqa: F401 - register DefaultAgent

        agent: Any = {
            "type": "DefaultAgent",
            "params": {
                "model": ScriptedChatModel(),
                "tools": [MockLookupTool, MockTimerTool],
                "proactive": False,
            },
        }
    else:
        agent = "MockAgent"
    config: dict[str, Any] = {
        "asr": "MockASR",
        "llm_agent": agent,
        "tts": "MockTTS",
        "service_config": {
            "realtime_api": {"model": MODEL_ALIAS, "api_keys": list(api_keys)},
            # Short detection windows keep spoken turns fast; assertions never
            # depend on these values.
            "vad_min_speech_ms": 100,
            "vad_redemption_ms": 200,
            "forced_alignment": {"stop_ack_timeout_ms": 300},
        },
    }
    if vad:
        config["vad"] = "MockVAD"
    return config


def build_mock_xtalk(**kwargs: Any) -> Xtalk:
    """Build an ``Xtalk`` instance from :func:`mock_config`."""
    return Xtalk.configure(mock_config(**kwargs)).build()


async def _serve(host: str, **kwargs: Any) -> None:
    import uvicorn

    app = build_mock_xtalk(**kwargs).create_asgi_app()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    sock.listen(128)
    sock.set_inheritable(True)
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on")
    )
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if serving.done():
            serving.result()
            return
        await asyncio.sleep(0.01)
    print(
        json.dumps({"port": port, "url": f"ws://{host}:{port}/v1/realtime?model={MODEL_ALIAS}"}),
        flush=True,
    )
    await serving


def main() -> None:
    """Parse the command line and serve until SIGTERM or SIGINT."""
    parser = argparse.ArgumentParser(description="Serve the mock-backed Xtalk app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--api-key", action="append", default=[])
    parser.add_argument("--tools", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        _serve(args.host, vad=not args.no_vad, api_keys=tuple(args.api_key), tools=args.tools)
    )


if __name__ == "__main__":
    main()
