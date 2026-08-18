"""Unit tests for sidecar process and Uvicorn lifecycle helpers."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

from backend.config import StartupConfig
from backend.main import main
from backend import server as server_module


TOKEN = "t" * 32


def _startup(tmp_path: Path) -> StartupConfig:
    """Create a temporary launch configuration."""

    return StartupConfig.from_mapping(
        {
            "protocol_version": 1,
            "token": TOKEN,
            "config_path": str(tmp_path / "config.json"),
            "data_dir": str(tmp_path / "data"),
            "origins": [],
            "config_overlay": {},
        }
    )


def test_bind_loopback_socket_uses_an_os_assigned_port() -> None:
    """Bind only IPv4 loopback and let the OS choose the port."""

    bound_socket = server_module.bind_loopback_socket()
    try:
        host, port = bound_socket.getsockname()
        assert host == "127.0.0.1"
        assert isinstance(port, int)
        assert port > 0
    finally:
        bound_socket.close()


def test_uvicorn_configuration_disables_access_and_proxy_logging() -> None:
    """Keep secrets out of access logs and ignore forwarded peer metadata."""

    config = server_module.create_uvicorn_config(object())

    assert config.host == "127.0.0.1"
    assert config.port == 0
    assert config.access_log is False
    assert config.proxy_headers is False


def test_ready_message_is_one_compact_flushed_json_line() -> None:
    """Emit the exact parent-process readiness protocol."""

    class _RecordingOutput(io.StringIO):
        """Track protocol flushes."""

        flushed = False

        def flush(self) -> None:
            """Record flushing before delegating."""

            self.flushed = True
            super().flush()

    output = _RecordingOutput()

    server_module.write_ready_message(output, port=43127)

    assert output.getvalue() == (
        '{"type":"ready","protocol_version":1,"port":43127}\n'
    )
    assert output.flushed is True


def test_serve_sidecar_emits_ready_only_after_server_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Wait for the server's started flag before writing protocol stdout."""

    class _FakeServer:
        """Minimal asynchronous Uvicorn server replacement."""

        def __init__(self, config: Any) -> None:
            self.config = config
            self.started = False
            self.should_exit = False

        async def serve(self, *, sockets: list[Any]) -> None:
            """Mark startup after receiving the pre-bound socket."""

            assert sockets
            self.started = True
            await asyncio.sleep(0)

    monkeypatch.setattr(
        server_module,
        "build_application",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(server_module.uvicorn, "Server", _FakeServer)
    output = io.StringIO()

    asyncio.run(
        server_module.serve_sidecar(
            _startup(tmp_path),
            protocol_output=output,
        )
    )

    payload = json.loads(output.getvalue())
    assert payload["type"] == "ready"
    assert payload["protocol_version"] == 1
    assert payload["port"] > 0


def test_main_reserves_stdout_for_the_ready_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Redirect model/provider chatter so readiness remains stdout's first line."""

    async def _fake_serve(
        startup: StartupConfig,
        *,
        protocol_output: io.StringIO,
    ) -> None:
        """Simulate noisy startup followed by readiness."""

        del startup
        print("provider startup noise")
        server_module.write_ready_message(protocol_output, port=43210)

    monkeypatch.setattr(server_module, "serve_sidecar", _fake_serve)
    payload = {
        "protocol_version": 1,
        "token": TOKEN,
        "config_path": str(tmp_path / "config.json"),
        "data_dir": str(tmp_path / "data"),
        "origins": [],
    }
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = main(
        stdin=io.StringIO(json.dumps(payload) + "\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert result == 0
    assert stdout.getvalue() == (
        '{"type":"ready","protocol_version":1,"port":43210}\n'
    )
    assert "provider startup noise" in stderr.getvalue()


def test_main_reports_invalid_startup_only_on_stderr() -> None:
    """Never contaminate protocol stdout when launch validation fails."""

    stdout = io.StringIO()
    stderr = io.StringIO()

    result = main(
        stdin=io.StringIO("not-json\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert result == 2
    assert stdout.getvalue() == ""
    assert "Invalid sidecar startup configuration" in stderr.getvalue()
