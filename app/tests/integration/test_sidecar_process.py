"""End-to-end startup and shutdown checks for the Python sidecar process."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = APP_ROOT.parent
SAMPLE_CONFIG = REPOSITORY_ROOT / "server_configs" / "sample.json"
VAD_MODEL = APP_ROOT / "resources" / "models" / "audio" / "silero_vad.onnx"
LAUNCH_TOKEN = "sidecar-integration-token-at-least-32-bytes"


def install_test_developer_tool(data_dir: Path) -> Path:
    """Install one representative developer tool into a test data directory.

    Parameters
    ----------
    data_dir : pathlib.Path
        Temporary application data directory used by the sidecar launch.

    Returns
    -------
    pathlib.Path
        Marker written when the configured entrypoint factory is called.
    """

    tools_root = data_dir / "tools"
    tool_directory = tools_root / "developer-timer"
    tool_directory.mkdir(parents=True)
    (tool_directory / "xtalk_tool.json").write_text(
        json.dumps(
            {
                "display_name": "Developer timer",
                "entrypoint": "timer_tool:create_tools",
            }
        ),
        encoding="utf-8",
    )
    (tool_directory / "timer_tool.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "from langchain_core.tools import tool",
                "",
                "@tool",
                "def developer_timer(duration_seconds: float) -> str:",
                '    """Run a representative developer timer."""',
                "    return f'Timer finished after {duration_seconds} seconds.'",
                "",
                "def create_tools():",
                "    Path(__file__).with_name('loaded.txt').write_text(",
                "        'loaded', encoding='utf-8'",
                "    )",
                "    return [developer_timer]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tools_root / "registry.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "id": "developer-timer",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return tool_directory / "loaded.txt"


def sidecar_command() -> list[str]:
    """Resolve the source or packaged sidecar command for integration tests.

    Returns
    -------
    list[str]
        Executable and arguments for one sidecar process.
    """

    packaged_executable = os.environ.get("XTALK_SIDECAR_EXECUTABLE")
    if packaged_executable:
        executable = Path(packaged_executable).expanduser().resolve()
        if not executable.is_file():
            raise ValueError(
                "XTALK_SIDECAR_EXECUTABLE must point to a sidecar executable"
            )
        return [str(executable)]
    return [sys.executable, "-m", "backend.entrypoint"]


def read_line_with_timeout(
    process: subprocess.Popen[str],
    timeout: float,
) -> str:
    """Read one stdout line from a process with a portable timeout.

    Parameters
    ----------
    process : subprocess.Popen[str]
        Running sidecar process.
    timeout : float
        Maximum seconds to wait.

    Returns
    -------
    str
        One line from protocol stdout.
    """

    if process.stdout is None:
        raise RuntimeError("sidecar stdout is not captured")
    result: queue.Queue[str] = queue.Queue(maxsize=1)
    reader = threading.Thread(
        target=lambda: result.put(process.stdout.readline()),
        daemon=True,
    )
    reader.start()
    return result.get(timeout=timeout)


def request_json(
    url: str,
    *,
    method: str = "GET",
) -> dict[str, object]:
    """Call one authenticated sidecar endpoint.

    Parameters
    ----------
    url : str
        Loopback URL.
    method : str, optional
        HTTP method.

    Returns
    -------
    dict[str, object]
        Decoded response object.
    """

    request = Request(
        url,
        method=method,
        headers={
            "Origin": "tauri://localhost",
            "X-XTalk-App-Token": LAUNCH_TOKEN,
        },
    )
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("sidecar response root must be an object")
    return payload


def _exercise_sidecar(config_path: Path, tmp_path: Path) -> None:
    """Start one sidecar configuration and stop it through the control API."""

    overlay = json.loads(os.environ.get("XTALK_TEST_CONFIG_OVERLAY", "{}"))
    if not isinstance(overlay, dict):
        raise ValueError("XTALK_TEST_CONFIG_OVERLAY must contain a JSON object")
    launch = {
        "protocol_version": 1,
        "token": LAUNCH_TOKEN,
        "config_path": str(config_path),
        "data_dir": str(tmp_path),
        "origins": ["tauri://localhost"],
        "config_fallbacks": {
            "vad": {
                "type": "SileroVAD",
                "params": {"model_path": str(VAD_MODEL)},
            },
        },
        "config_overlay": overlay,
    }
    environment = dict(os.environ)
    python_path = os.pathsep.join(
        [
            str(APP_ROOT),
            str(REPOSITORY_ROOT / "src"),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    environment["PYTHONPATH"] = python_path

    process = subprocess.Popen(
        sidecar_command(),
        cwd=APP_ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if process.stdin is None:
            raise RuntimeError("sidecar stdin is not captured")
        process.stdin.write(json.dumps(launch, separators=(",", ":")) + "\n")
        process.stdin.flush()

        ready_line = read_line_with_timeout(process, 30)
        if not ready_line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"sidecar exited before readiness: {stderr}")
        ready = json.loads(ready_line)
        assert ready["type"] == "ready"
        assert ready["protocol_version"] == 1
        port = int(ready["port"])
        assert 0 < port < 65536

        origin = f"http://127.0.0.1:{port}"
        assert request_json(f"{origin}/health")["status"] == "ok"
        assert request_json(f"{origin}/ready")["status"] == "ready"
        assert (
            request_json(
                f"{origin}/app/api/shutdown",
                method="POST",
            )["status"]
            == "shutting_down"
        )
        assert process.wait(timeout=15) == 0
        assert process.stdout.read() == ""
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_provider_free_sidecar_ready_health_and_shutdown(tmp_path: Path) -> None:
    """Start an external setup-state config without remote providers."""

    config_path = tmp_path / "provider-free.json"
    config_path.write_text(
        json.dumps(
            {
                "service_config": {
                    "enable_persistence": True,
                }
            }
        ),
        encoding="utf-8",
    )
    _exercise_sidecar(config_path, tmp_path)


def test_sidecar_loads_enabled_developer_tool_directory(tmp_path: Path) -> None:
    """Load a copied tool directory through the full sidecar process."""

    marker = install_test_developer_tool(tmp_path)
    config_path = tmp_path / "developer-tool.json"
    config_path.write_text(
        json.dumps(
            {
                "llm_agent": {
                    "type": "DefaultAgent",
                    "params": {
                        "model": {
                            "api_key": "test-key",
                            "model": "test-model",
                            "base_url": "http://127.0.0.1:9",
                        },
                        "proactive": False,
                    },
                },
                "service_config": {
                    "enable_persistence": False,
                },
            }
        ),
        encoding="utf-8",
    )

    _exercise_sidecar(config_path, tmp_path)

    assert marker.read_text(encoding="utf-8") == "loaded"


@pytest.mark.model
def test_sample_sidecar_ready_health_and_shutdown(tmp_path: Path) -> None:
    """Start against the sample model config and stop through the control API."""

    if os.environ.get("XTALK_RUN_MODEL_TESTS") != "1":
        pytest.skip("set XTALK_RUN_MODEL_TESTS=1 to start the configured sidecar")

    _exercise_sidecar(SAMPLE_CONFIG, tmp_path)
