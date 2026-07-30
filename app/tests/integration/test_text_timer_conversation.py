"""Black-box text-to-timer conversation test for the desktop sidecar."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
import websockets


APP_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = APP_ROOT.parent
SAMPLE_CONFIG = REPOSITORY_ROOT / "server_configs" / "sample.json"
VAD_MODEL = Path(
    os.environ.get(
        "XTALK_TEST_VAD_MODEL_PATH",
        str(APP_ROOT / "resources" / "models" / "audio" / "silero_vad.onnx"),
    )
).expanduser().resolve()
VOICE_FIXTURE = (
    APP_ROOT
    / "tests"
    / "fixtures"
    / "audio"
    / "vad_speech_then_silence.s16le.pcm"
)
MODEL_TEST_ENVIRONMENT_VARIABLE = "XTALK_RUN_MODEL_TESTS"
STARTUP_TIMEOUT_SECONDS = 60.0
CONVERSATION_TIMEOUT_SECONDS = 180.0
PROCESS_EXIT_TIMEOUT_SECONDS = 20.0
TIMER_DURATION_SECONDS = 2.0
PCM_FRAME_BYTES = 1_024
PCM_FRAME_DURATION_SECONDS = 0.032
TEXT_MESSAGE = (
    "请务必调用 timer 工具启动一个 2 秒的计时器，"
    "不设置进度提醒。"
    "请先简短确认，并在计时完成后告诉我。"
)
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|authorization|bearer|token|secret|password)"
    r"(\s*[\"']?\s*[:=]\s*[\"']?\s*)"
    r"([^,\s\"'}]+)"
)


class _DiagnosticCapture:
    """Drain sidecar stderr and retain only sanitized warning diagnostics."""

    def __init__(self, secrets_to_redact: set[str]) -> None:
        """Initialize a bounded, thread-safe warning capture.

        Parameters
        ----------
        secrets_to_redact : set[str]
            Exact sensitive values that must never enter retained diagnostics.
        """

        self._secrets = set(secrets_to_redact)
        self._warnings: deque[str] = deque(maxlen=40)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def add_secret(self, value: str) -> None:
        """Add a newly issued credential to the in-memory redaction set."""

        if len(value) < 4:
            return
        with self._lock:
            self._secrets.add(value)

    def start(self, process: subprocess.Popen[str]) -> None:
        """Start draining a child process's diagnostic stream."""

        if process.stderr is None:
            raise AssertionError("sidecar diagnostic stderr is unavailable")
        self._thread = threading.Thread(
            target=self._drain,
            args=(process.stderr,),
            daemon=True,
        )
        self._thread.start()

    def warning_summary(self) -> str:
        """Return a bounded single-line rendering of captured warnings."""

        with self._lock:
            warnings = list(self._warnings)
        if not warnings:
            return "none"
        return " | ".join(warnings)[-4_000:]

    def _drain(self, stream: Any) -> None:
        """Consume stderr until EOF without retaining ordinary provider logs."""

        for raw_line in stream:
            if "warning" not in raw_line.casefold():
                continue
            sanitized = self._sanitize(raw_line.strip())
            with self._lock:
                self._warnings.append(sanitized)

    def _sanitize(self, line: str) -> str:
        """Redact exact and labelled credentials from one warning line."""

        with self._lock:
            sensitive_values = tuple(self._secrets)
        sanitized = line
        for sensitive_value in sensitive_values:
            sanitized = sanitized.replace(sensitive_value, "<redacted>")
        return _CREDENTIAL_PATTERN.sub(
            lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
            sanitized,
        )


def _redaction_values(launch_token: str) -> set[str]:
    """Collect environment and sample-config secrets without emitting them.

    Parameters
    ----------
    launch_token : str
        Random app credential for the current process.

    Returns
    -------
    set[str]
        Sensitive values used only by :class:`_DiagnosticCapture`.
    """

    sensitive_values = {launch_token}
    sensitive_name_parts = (
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "authorization",
    )

    def collect(value: Any, *, sensitive: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                normalized_key = str(key).casefold().replace("-", "_")
                collect(
                    nested_value,
                    sensitive=sensitive
                    or any(part in normalized_key for part in sensitive_name_parts),
                )
            return
        if isinstance(value, list):
            for nested_value in value:
                collect(nested_value, sensitive=sensitive)
            return
        if sensitive and isinstance(value, str) and len(value) >= 4:
            sensitive_values.add(value)

    try:
        collect(json.loads(SAMPLE_CONFIG.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        pass
    collect(dict(os.environ))
    return sensitive_values


def _sidecar_command_and_environment() -> tuple[list[str], dict[str, str]]:
    """Resolve a packaged sidecar or the source-tree module command.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        Process command and environment for one isolated sidecar launch.

    Raises
    ------
    ValueError
        Raised when ``XTALK_SIDECAR_EXECUTABLE`` does not name a file.
    """

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    packaged_executable = environment.get("XTALK_SIDECAR_EXECUTABLE")
    if packaged_executable:
        executable = Path(packaged_executable).expanduser().resolve()
        if not executable.is_file():
            raise ValueError(
                "XTALK_SIDECAR_EXECUTABLE must point to a sidecar executable"
            )
        return [str(executable)], environment

    python_path = os.pathsep.join(
        [
            str(APP_ROOT),
            str(REPOSITORY_ROOT / "src"),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    environment["PYTHONPATH"] = python_path
    return [sys.executable, "-m", "backend.entrypoint"], environment


def _read_protocol_line(
    process: subprocess.Popen[str],
    *,
    timeout: float,
) -> str:
    """Read one sidecar stdout protocol line with a portable timeout.

    Parameters
    ----------
    process : subprocess.Popen[str]
        Sidecar process whose stdout is captured.
    timeout : float
        Maximum wait in seconds.

    Returns
    -------
    str
        One complete protocol line.

    Raises
    ------
    AssertionError
        Raised when the process does not produce a line before the timeout.
    """

    if process.stdout is None:
        raise AssertionError("sidecar protocol stdout is unavailable")
    result: queue.Queue[str] = queue.Queue(maxsize=1)
    reader = threading.Thread(
        target=lambda: result.put(process.stdout.readline()),
        daemon=True,
    )
    reader.start()
    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        raise AssertionError(
            "sidecar did not report readiness before timeout"
        ) from None


def _request_json(
    origin: str,
    path: str,
    *,
    launch_token: str,
    method: str = "POST",
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Call one sidecar HTTP endpoint without logging credentials or payloads.

    Parameters
    ----------
    origin : str
        Loopback HTTP origin emitted by the sidecar.
    path : str
        Absolute endpoint path without credentials.
    launch_token : str
        Per-process app authentication token.
    method : str, optional
        HTTP method.
    body : dict[str, Any] | None, optional
        JSON request body.

    Returns
    -------
    tuple[int, dict[str, Any]]
        HTTP status and decoded JSON object.

    Raises
    ------
    AssertionError
        Raised for transport, HTTP, or response-shape failures.
    """

    encoded_body = (
        json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if body is not None
        else b""
    )
    headers = {
        "Accept": "application/json",
        "Origin": "tauri://localhost",
        "X-XTalk-App-Token": launch_token,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        f"{origin}{path}",
        data=encoded_body,
        method=method,
        headers=headers,
    )
    try:
        with urlopen(request, timeout=15) as response:
            status_code = int(response.status)
            payload = json.load(response)
    except HTTPError as exc:
        raise AssertionError(
            f"sidecar HTTP request failed with status {exc.code}"
        ) from None
    except (OSError, TimeoutError, URLError, json.JSONDecodeError) as exc:
        raise AssertionError(
            f"sidecar HTTP request failed with {type(exc).__name__}"
        ) from None
    if not isinstance(payload, dict):
        raise AssertionError("sidecar HTTP response root must be an object")
    return status_code, payload


def _start_sidecar(
    tmp_path: Path,
    *,
    launch_token: str,
) -> tuple[subprocess.Popen[str], str, _DiagnosticCapture]:
    """Launch the sidecar with the sample models and return its loopback origin.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test writable data directory.
    launch_token : str
        Random per-process app authentication token.

    Returns
    -------
    tuple[subprocess.Popen[str], str, _DiagnosticCapture]
        Running process, loopback HTTP origin, and sanitized warning capture.

    Notes
    -----
    Provider diagnostics are drained continuously to prevent child-process
    backpressure. Only warning lines are retained, after exact and labelled
    credentials have been removed.
    """

    command, environment = _sidecar_command_and_environment()
    process = subprocess.Popen(
        command,
        cwd=APP_ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    diagnostics = _DiagnosticCapture(_redaction_values(launch_token))
    diagnostics.start(process)
    launch = {
        "protocol_version": 1,
        "token": launch_token,
        "config_path": str(SAMPLE_CONFIG),
        "data_dir": str(tmp_path),
        "origins": ["tauri://localhost"],
        "config_fallbacks": {
            "vad": {
                "type": "SileroVAD",
                "params": {
                    "model_path": str(VAD_MODEL),
                },
            },
        },
        "config_overlay": {
            "llm_agent": {
                "params": {
                    "proactive": False,
                }
            },
            "service_config": {
                "enable_persistence": False,
            },
        },
    }
    try:
        if process.stdin is None:
            raise AssertionError("sidecar startup stdin is unavailable")
        process.stdin.write(
            json.dumps(launch, separators=(",", ":"), ensure_ascii=False) + "\n"
        )
        process.stdin.flush()
        process.stdin.close()

        ready_line = _read_protocol_line(
            process,
            timeout=STARTUP_TIMEOUT_SECONDS,
        )
        if not ready_line:
            raise AssertionError(
                f"sidecar exited before readiness with code {process.poll()}"
            )
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError:
            raise AssertionError("sidecar readiness line is not valid JSON") from None
        if not isinstance(ready, dict):
            raise AssertionError("sidecar readiness payload must be an object")
        if ready.get("type") != "ready" or ready.get("protocol_version") != 1:
            raise AssertionError("sidecar emitted an incompatible readiness payload")
        port = ready.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise AssertionError("sidecar readiness payload contains an invalid port")
        return process, f"http://127.0.0.1:{port}", diagnostics
    except BaseException:
        _stop_process(process)
        raise


def _stop_process(process: subprocess.Popen[str]) -> int:
    """Stop a sidecar process with terminate-then-kill fallback.

    Parameters
    ----------
    process : subprocess.Popen[str]
        Sidecar process to reap.

    Returns
    -------
    int
        Final process return code.
    """

    if process.poll() is None:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.wait(timeout=5)


async def _exercise_text_timer_conversation(
    *,
    origin: str,
    access_token: str,
    diagnostics: _DiagnosticCapture,
) -> None:
    """Drive text input through the real WebSocket conversation and timer.

    Parameters
    ----------
    origin : str
        Sidecar loopback HTTP origin.
    access_token : str
        XTalk login JWT used only for the WebSocket handshake.
    diagnostics : _DiagnosticCapture
        Sanitized sidecar warning capture used only in failure evidence.

    Raises
    ------
    AssertionError
        Raised when any observable conversation milestone is missing.
    """

    websocket_origin = origin.replace("http://", "ws://", 1)
    websocket_uri = (
        f"{websocket_origin}/ws?"
        f"{urlencode({'access_token': access_token})}"
    )
    finish_asr_seen = False
    timer_call_seen = False
    finish_response_count = 0
    binary_chunk_count = 0
    chunk_ack_count = 0
    tts_finished_count = 0
    playback_finished_ack_count = 0
    action_counts: Counter[str] = Counter()

    def conversation_is_complete() -> bool:
        """Return whether every public timer-conversation milestone is complete."""

        return (
            finish_asr_seen
            and timer_call_seen
            and finish_response_count >= 1
            and binary_chunk_count > 0
            and chunk_ack_count == binary_chunk_count
            and tts_finished_count >= 1
            and playback_finished_ack_count >= 1
        )

    async def run_conversation() -> None:
        nonlocal finish_asr_seen
        nonlocal timer_call_seen
        nonlocal finish_response_count
        nonlocal binary_chunk_count
        nonlocal chunk_ack_count
        nonlocal tts_finished_count
        nonlocal playback_finished_ack_count

        async with websockets.connect(
            websocket_uri,
            max_size=None,
            open_timeout=20,
            close_timeout=5,
        ) as websocket:
            playback_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            conversation_completed = False

            async def playback_worker() -> None:
                """Serialize real-time PCM playback and chunk acknowledgements."""

                nonlocal chunk_ack_count
                while True:
                    chunk = await playback_queue.get()
                    try:
                        if chunk is None:
                            return
                        chunk_duration_seconds = (len(chunk) // 2) / 48_000.0
                        await asyncio.sleep(chunk_duration_seconds)
                        await websocket.send(
                            json.dumps(
                                {"action": "tts_chunk_played"},
                                separators=(",", ":"),
                            )
                        )
                        chunk_ack_count += 1
                    finally:
                        playback_queue.task_done()

            playback_task = asyncio.create_task(
                playback_worker(),
                name="text-timer-playback",
            )
            try:
                await websocket.send(
                    json.dumps(
                        {
                            "action": "attach_session",
                            "session_id": None,
                        },
                        separators=(",", ":"),
                    )
                )

                attached_message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=20,
                )
                if not isinstance(attached_message, str):
                    raise AssertionError("session attachment did not return JSON")
                try:
                    attached = json.loads(attached_message)
                except json.JSONDecodeError:
                    raise AssertionError(
                        "session attachment response is invalid"
                    ) from None
                if (
                    not isinstance(attached, dict)
                    or attached.get("action") != "session_attached"
                    or not isinstance(attached.get("data"), dict)
                ):
                    raise AssertionError("sidecar did not attach the WebSocket session")
                session_id = attached["data"].get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise AssertionError("attached session identifier is missing")

                await websocket.send(
                    json.dumps(
                        {
                            "action": "submit_text",
                            "text": TEXT_MESSAGE,
                        },
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                )

                while True:
                    incoming = await websocket.recv()
                    if isinstance(incoming, bytes):
                        action_counts["binary_audio"] += 1
                        binary_chunk_count += 1
                        await playback_queue.put(incoming)
                        continue

                    try:
                        payload = json.loads(incoming)
                    except json.JSONDecodeError:
                        raise AssertionError(
                            "sidecar emitted a non-JSON text frame"
                        ) from None
                    if not isinstance(payload, dict):
                        raise AssertionError(
                            "sidecar text frame root must be an object"
                        )
                    action = payload.get("action")
                    data = payload.get("data")
                    action_name = action if isinstance(action, str) else "<missing>"
                    action_counts[action_name] += 1

                    if action == "error":
                        raise AssertionError("sidecar emitted an error action")
                    if action == "tts_finished":
                        tts_finished_count += 1
                        try:
                            await asyncio.wait_for(
                                playback_queue.join(),
                                timeout=60,
                            )
                        except asyncio.TimeoutError:
                            raise AssertionError(
                                "timed out draining queued TTS audio"
                            ) from None
                        # InputGateway dispatches client text signals in separate
                        # tasks. Keep a full barrier interval after the final
                        # serialized chunk ACK so playback-finished cannot overtake
                        # any acknowledgement still being handled.
                        await asyncio.sleep(1.0)
                        await websocket.send(
                            json.dumps(
                                {"action": "tts_playback_finished"},
                                separators=(",", ":"),
                            )
                        )
                        playback_finished_ack_count += 1
                        if conversation_is_complete():
                            conversation_completed = True
                            return
                        continue
                    if action == "finish_asr":
                        if (
                            not isinstance(data, dict)
                            or data.get("text") != TEXT_MESSAGE
                            or data.get("origin") != "text"
                        ):
                            raise AssertionError(
                                "text input confirmation did not match the submission"
                            )
                        finish_asr_seen = True
                        continue
                    if action == "tool_called" and isinstance(data, dict):
                        if data.get("name") != "timer":
                            continue
                        args = data.get("args")
                        if not isinstance(args, dict):
                            raise AssertionError(
                                "timer tool call arguments are missing"
                            )
                        try:
                            duration = float(args.get("duration_seconds"))
                        except (TypeError, ValueError):
                            raise AssertionError(
                                "timer duration is not numeric"
                            ) from None
                        if abs(duration - TIMER_DURATION_SECONDS) > 0.25:
                            raise AssertionError(
                                "timer duration differs from the request"
                            )
                        if args.get("reminder_interval_seconds") is not None:
                            raise AssertionError(
                                "timer unexpectedly enabled reminders"
                            )
                        timer_call_seen = True
                        continue
                    if action != "finish_resp" or not isinstance(data, dict):
                        continue

                    response_text = data.get("text")
                    if not isinstance(response_text, str) or not response_text:
                        raise AssertionError(
                            "finish_resp does not contain assistant text"
                        )
                    finish_response_count += 1
                    if conversation_is_complete():
                        conversation_completed = True
                        return
            finally:
                if conversation_completed:
                    await playback_queue.put(None)
                    await playback_task
                else:
                    playback_task.cancel()
                    await asyncio.gather(playback_task, return_exceptions=True)

    try:
        await asyncio.wait_for(
            run_conversation(),
            timeout=CONVERSATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise AssertionError(
            "timed out waiting for text/timer completion "
            f"(finish_asr={finish_asr_seen}, timer_call={timer_call_seen}, "
            f"finish_resp={finish_response_count}, "
            f"audio_chunks={binary_chunk_count}, "
            f"chunk_acks={chunk_ack_count}, "
            f"tts_finished={tts_finished_count}, "
            f"playback_finished_acks={playback_finished_ack_count}, "
            f"actions={dict(action_counts)}, "
            f"warnings={diagnostics.warning_summary()})"
        ) from None
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"text/timer conversation failed with {type(exc).__name__}"
        ) from None


async def _exercise_voice_conversation(
    *,
    origin: str,
    access_token: str,
    diagnostics: _DiagnosticCapture,
) -> None:
    """Drive raw microphone PCM through backend VAD, ASR, and the agent.

    Parameters
    ----------
    origin : str
        Sidecar loopback HTTP origin.
    access_token : str
        XTalk login JWT used only for the WebSocket handshake.
    diagnostics : _DiagnosticCapture
        Sanitized warning capture used only in failure evidence.

    Raises
    ------
    AssertionError
        Raised when server VAD boundaries, final ASR, or an assistant response
        are missing.
    """

    websocket_origin = origin.replace("http://", "ws://", 1)
    websocket_uri = (
        f"{websocket_origin}/ws?"
        f"{urlencode({'access_token': access_token})}"
    )
    audio = VOICE_FIXTURE.read_bytes()
    if not audio or len(audio) % PCM_FRAME_BYTES:
        raise AssertionError("voice fixture does not contain complete PCM frames")

    vad_sequence: list[str] = []
    finish_asr_text = ""
    finish_response_text = ""
    action_counts: Counter[str] = Counter()

    def conversation_is_complete() -> bool:
        """Return whether the complete microphone-to-response path succeeded."""

        return (
            vad_sequence == ["start", "end"]
            and bool(finish_asr_text)
            and bool(finish_response_text)
        )

    async def run_conversation() -> None:
        nonlocal finish_asr_text
        nonlocal finish_response_text

        async with websockets.connect(
            websocket_uri,
            max_size=None,
            open_timeout=20,
            close_timeout=5,
        ) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "action": "attach_session",
                        "session_id": None,
                    },
                    separators=(",", ":"),
                )
            )
            attached_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=20,
            )
            if not isinstance(attached_message, str):
                raise AssertionError("session attachment did not return JSON")
            try:
                attached = json.loads(attached_message)
            except json.JSONDecodeError:
                raise AssertionError(
                    "session attachment response is invalid"
                ) from None
            if (
                not isinstance(attached, dict)
                or attached.get("action") != "session_attached"
            ):
                raise AssertionError("sidecar did not attach the voice session")

            async def stream_microphone_audio() -> None:
                """Send PCM frames at the capture cadence without client VAD."""

                for offset in range(0, len(audio), PCM_FRAME_BYTES):
                    await websocket.send(audio[offset : offset + PCM_FRAME_BYTES])
                    await asyncio.sleep(PCM_FRAME_DURATION_SECONDS)

            microphone_task = asyncio.create_task(
                stream_microphone_audio(),
                name="voice-fixture-stream",
            )
            try:
                while True:
                    incoming = await websocket.recv()
                    if isinstance(incoming, bytes):
                        action_counts["binary_audio"] += 1
                        await websocket.send(
                            json.dumps(
                                {"action": "tts_chunk_played"},
                                separators=(",", ":"),
                            )
                        )
                        continue

                    try:
                        payload = json.loads(incoming)
                    except json.JSONDecodeError:
                        raise AssertionError(
                            "sidecar emitted a non-JSON text frame"
                        ) from None
                    if not isinstance(payload, dict):
                        raise AssertionError(
                            "sidecar text frame root must be an object"
                        )
                    action = payload.get("action")
                    data = payload.get("data")
                    action_name = action if isinstance(action, str) else "<missing>"
                    action_counts[action_name] += 1

                    if action == "error":
                        raise AssertionError("sidecar emitted an error action")
                    if action in {"vad_speech_start", "vad_speech_end"}:
                        if (
                            not isinstance(data, dict)
                            or data.get("origin") != "server"
                        ):
                            raise AssertionError(
                                "voice boundary was not produced by backend VAD"
                            )
                        boundary = (
                            "start" if action == "vad_speech_start" else "end"
                        )
                        vad_sequence.append(boundary)
                        if vad_sequence not in (["start"], ["start", "end"]):
                            raise AssertionError(
                                "backend VAD emitted duplicate or unordered boundaries"
                            )
                        continue
                    if action == "finish_asr":
                        if not isinstance(data, dict):
                            raise AssertionError("finish_asr data is missing")
                        text = data.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise AssertionError("finish_asr text is empty")
                        finish_asr_text = text.strip()
                        continue
                    if action == "finish_resp":
                        if not isinstance(data, dict):
                            raise AssertionError("finish_resp data is missing")
                        text = data.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise AssertionError("finish_resp text is empty")
                        finish_response_text = text.strip()
                        if conversation_is_complete():
                            await microphone_task
                            return
                        continue
                    if action == "tts_finished":
                        await websocket.send(
                            json.dumps(
                                {"action": "tts_playback_finished"},
                                separators=(",", ":"),
                            )
                        )
            finally:
                if not microphone_task.done():
                    microphone_task.cancel()
                await asyncio.gather(microphone_task, return_exceptions=True)

    try:
        await asyncio.wait_for(
            run_conversation(),
            timeout=CONVERSATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise AssertionError(
            "timed out waiting for voice completion "
            f"(vad={vad_sequence}, finish_asr={bool(finish_asr_text)}, "
            f"finish_resp={bool(finish_response_text)}, "
            f"actions={dict(action_counts)}, "
            f"warnings={diagnostics.warning_summary()})"
        ) from None
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"voice conversation failed with {type(exc).__name__}"
        ) from None


@pytest.mark.model
def test_text_message_invokes_timer_and_completes_conversation(
    tmp_path: Path,
) -> None:
    """Verify the sample-config sidecar runs a typed timer conversation."""

    if os.environ.get(MODEL_TEST_ENVIRONMENT_VARIABLE) != "1":
        pytest.skip(
            "set XTALK_RUN_MODEL_TESTS=1 to run the configured conversation"
        )
    if not SAMPLE_CONFIG.is_file():
        raise AssertionError("repository sample model configuration is missing")

    launch_token = secrets.token_urlsafe(32)
    process, origin, diagnostics = _start_sidecar(
        tmp_path,
        launch_token=launch_token,
    )
    shutdown_requested = False
    try:
        login_status, login = _request_json(
            origin,
            "/api/auth/login",
            launch_token=launch_token,
        )
        if login_status != 200:
            raise AssertionError("XTalk login did not return HTTP 200")
        access_token = login.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise AssertionError("XTalk login response omitted its access token")
        diagnostics.add_secret(access_token)

        asyncio.run(
            _exercise_text_timer_conversation(
                origin=origin,
                access_token=access_token,
                diagnostics=diagnostics,
            )
        )

        shutdown_status, shutdown = _request_json(
            origin,
            "/app/api/shutdown",
            launch_token=launch_token,
        )
        shutdown_requested = True
        if shutdown_status != 200 or shutdown.get("status") != "shutting_down":
            raise AssertionError("sidecar did not accept graceful shutdown")
        if process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS) != 0:
            raise AssertionError("sidecar exited with a non-zero status")
        if process.stdout is not None and process.stdout.read() != "":
            raise AssertionError("sidecar wrote unexpected protocol output")
    finally:
        if not shutdown_requested and process.poll() is None:
            try:
                _request_json(
                    origin,
                    "/app/api/shutdown",
                    launch_token=launch_token,
                )
            except AssertionError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _stop_process(process)


@pytest.mark.model
def test_voice_audio_produces_asr_text_and_assistant_response(
    tmp_path: Path,
) -> None:
    """Verify sample-config speech crosses server VAD, ASR, and the agent."""

    if os.environ.get(MODEL_TEST_ENVIRONMENT_VARIABLE) != "1":
        pytest.skip(
            "set XTALK_RUN_MODEL_TESTS=1 to run the configured conversation"
        )
    for required_path in (SAMPLE_CONFIG, VAD_MODEL, VOICE_FIXTURE):
        if not required_path.is_file():
            raise AssertionError(
                f"required voice input is missing: {required_path.name}"
            )

    launch_token = secrets.token_urlsafe(32)
    process, origin, diagnostics = _start_sidecar(
        tmp_path,
        launch_token=launch_token,
    )
    shutdown_requested = False
    try:
        login_status, login = _request_json(
            origin,
            "/api/auth/login",
            launch_token=launch_token,
        )
        if login_status != 200:
            raise AssertionError("XTalk login did not return HTTP 200")
        access_token = login.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise AssertionError("XTalk login response omitted its access token")
        diagnostics.add_secret(access_token)

        asyncio.run(
            _exercise_voice_conversation(
                origin=origin,
                access_token=access_token,
                diagnostics=diagnostics,
            )
        )

        shutdown_status, shutdown = _request_json(
            origin,
            "/app/api/shutdown",
            launch_token=launch_token,
        )
        shutdown_requested = True
        if shutdown_status != 200 or shutdown.get("status") != "shutting_down":
            raise AssertionError("sidecar did not accept graceful shutdown")
        if process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS) != 0:
            raise AssertionError("sidecar exited with a non-zero status")
        if process.stdout is not None and process.stdout.read() != "":
            raise AssertionError("sidecar wrote unexpected protocol output")
    finally:
        if not shutdown_requested and process.poll() is None:
            try:
                _request_json(
                    origin,
                    "/app/api/shutdown",
                    launch_token=launch_token,
                )
            except AssertionError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _stop_process(process)
