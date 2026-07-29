import argparse
import asyncio
import copy
import json
import logging
import mimetypes
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import Field

from xtalk import Xtalk
from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")

FRONTEND_UTILITIES_ROUTE = "/xtalk/frontend-utilities"
FRONTEND_UTILITIES_DIR = Path(__file__).parent / "dist"
DOWNLOAD_TIMEOUT_SECONDS = 60
ORT_VERSIONS = ("1.22.0", "1.17.0")
VAD_VERSION = "0.0.27"
FASTENHANCER_URL = (
    "https://github.com/aask1357/fastenhancer/releases/download/"
    "onnx-48khz-v1/fastenhancer_s.onnx"
)


TIMER_AGENT_INSTRUCTION = """
你可以使用 timer 启动后台定时器。调用 timer 时，如果
reminder_interval_seconds 不为空，收到 timer 的初始工具结果后，必须立即使用
其中的工具调用 ID 调用 subscribe_async_tool。订阅成功后停止调用工具，等待系统
主动发送进度。reminder_interval_seconds 为空时不要订阅。只有用户明确询问当前
进度时才调用 id_to_async_tool_status，不要轮询状态。
"""


class TimerInput(ToolInput):
    """Input accepted by the asynchronous timer."""

    duration_seconds: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="定时器总时长，单位为秒，必须大于零。",
    )
    reminder_interval_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "可选提醒间隔，单位为秒。传入时，启动定时器后必须立即调用 "
            "subscribe_async_tool 订阅过程提醒。"
        ),
    )


@dataclass
class TimerState(ToolState):
    """Mutable state for one asynchronous timer invocation."""

    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    stopped: bool = False


class TimerOutput(ToolOutput):
    """Final result returned when an asynchronous timer expires."""

    content: str
    elapsed_seconds: float

    def to_content(self) -> str:
        """Return the human-readable timer completion message."""
        return self.content


class TimerTool(AsyncTool):
    """启动后台定时器，可选按固定秒数间隔主动提醒用户。"""

    name = "timer"
    subscribe_by_default = False

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format a duration compactly for a human-readable message."""
        return f"{seconds:.1f}".rstrip("0").rstrip(".")

    @classmethod
    def _elapsed_seconds(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
    ) -> float:
        """Return the timer's current elapsed time, clamped to its duration."""
        if tool_state.started_at <= 0:
            return tool_state.elapsed_seconds
        current = max(
            tool_state.elapsed_seconds,
            time.monotonic() - tool_state.started_at,
        )
        return min(tool_input.duration_seconds, current)

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Running:
        """Start the timer and immediately return its protocol result."""
        del global_state
        tool_state.started_at = time.monotonic()
        duration = cls._format_seconds(tool_input.duration_seconds)
        message = f"定时器已启动，将在 {duration} 秒后结束。工具调用 ID：{tool_call_id}。"
        if tool_input.reminder_interval_seconds is not None:
            interval = cls._format_seconds(tool_input.reminder_interval_seconds)
            message += (
                f"用户要求每 {interval} 秒提醒一次，请立即调用 "
                f"subscribe_async_tool，source_call_id 为 {tool_call_id}。"
            )
        return Running(message)

    @classmethod
    def emit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[TimerOutput]]:
        """Yield no synchronous updates because the timer is natively async."""
        del tool_input, tool_state, global_state
        return iter(())

    @classmethod
    async def aemit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> AsyncIterator[ToolResult[TimerOutput]]:
        """Emit optional interval reminders and finish at the requested time."""
        del global_state
        duration = tool_input.duration_seconds
        interval = tool_input.reminder_interval_seconds
        next_reminder = interval

        while next_reminder is not None and next_reminder < duration:
            wake_at = tool_state.started_at + next_reminder
            await asyncio.sleep(max(0.0, wake_at - time.monotonic()))
            if tool_state.stopped:
                return
            tool_state.elapsed_seconds = cls._elapsed_seconds(tool_input, tool_state)
            elapsed = cls._format_seconds(tool_state.elapsed_seconds)
            total = cls._format_seconds(duration)
            yield Running(f"定时器已经过 {elapsed} 秒，共 {total} 秒。")
            next_reminder += interval

        finish_at = tool_state.started_at + duration
        await asyncio.sleep(max(0.0, finish_at - time.monotonic()))
        if tool_state.stopped:
            return
        tool_state.elapsed_seconds = duration
        total = cls._format_seconds(duration)
        yield Finished(
            TimerOutput(
                content=f"定时器结束，设定的 {total} 秒已到。",
                elapsed_seconds=duration,
            )
        )

    @classmethod
    def status(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> str:
        """Return how many seconds have elapsed for the timer."""
        del global_state
        elapsed_seconds = cls._elapsed_seconds(tool_input, tool_state)
        elapsed = cls._format_seconds(elapsed_seconds)
        total = cls._format_seconds(tool_input.duration_seconds)
        if tool_state.stopped:
            return f"定时器已停止，停止前经过 {elapsed} 秒，共 {total} 秒。"
        return f"定时器已经过 {elapsed} 秒，共 {total} 秒。"

    @classmethod
    def stop(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        """Record elapsed time before ToolEngine cancels the timer task."""
        del global_state
        tool_state.elapsed_seconds = cls._elapsed_seconds(tool_input, tool_state)
        tool_state.stopped = True


def build_timer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config copy with the timer tool and its Agent instruction."""
    updated_config = copy.deepcopy(config)
    agent_config = updated_config.get("llm_agent")
    if not isinstance(agent_config, dict):
        raise ValueError("The config must define an llm_agent object.")

    params = agent_config.get("params")
    if params is None:
        params = {}
        agent_config["params"] = params
    if not isinstance(params, dict):
        raise ValueError("llm_agent.params must be an object.")

    configured_tools = params.get("tools", [])
    if not isinstance(configured_tools, list):
        raise ValueError("llm_agent.params.tools must be a list when provided.")
    params["tools"] = [*configured_tools, TimerTool]

    configured_prompt = params.get("system_prompt", "")
    if not isinstance(configured_prompt, str):
        raise ValueError("llm_agent.params.system_prompt must be a string.")
    params["system_prompt"] = (
        f"{configured_prompt.rstrip()}\n{TIMER_AGENT_INSTRUCTION.strip()}"
    ).lstrip()
    return updated_config


def download_frontend_utilities() -> None:
    """Download browser-side runtime and model assets for the sample app.

    The assets are stored under ``examples/sample_app/dist`` and are mounted by
    this server at ``/xtalk/frontend-utilities``.

    Raises
    ------
    RuntimeError
        Raised when any missing asset cannot be downloaded.
    """

    def download_file(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_name(f"{target.name}.tmp")
        try:
            with urllib.request.urlopen(
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                with tmp_target.open("wb") as file:
                    shutil.copyfileobj(response, file)
            tmp_target.replace(target)
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as exc:
            if tmp_target.exists():
                tmp_target.unlink()
            raise RuntimeError(
                "Failed to download frontend utilities. "
                "Please download the missing file manually to "
                f"{target}, or run this server once on a machine with internet "
                f"access. Source URL: {url}. Error: {exc}"
            ) from exc

    def extract_npm_dist_files(
        *,
        package_name: str,
        tarball_url: str,
        target_dir: Path,
        required_names: set[str],
        include_wasm: bool = False,
        include_mjs: bool = False,
        require_mjs: bool = False,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        has_required_files = all((target_dir / name).exists() for name in required_names)
        has_wasm_files = any(target_dir.glob("*.wasm"))
        has_mjs_files = any(target_dir.glob("*.mjs"))
        if has_required_files:
            if (not include_wasm or has_wasm_files) and (not include_mjs or has_mjs_files):
                return

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "package.tgz"
            download_file(tarball_url, archive_path)
            try:
                with tarfile.open(archive_path, "r:gz") as archive:
                    for member in archive.getmembers():
                        member_path = Path(member.name)
                        if (
                            not member.isfile()
                            or len(member_path.parts) < 3
                            or member_path.parts[0] != "package"
                            or member_path.parts[1] != "dist"
                        ):
                            continue
                        file_name = member_path.name
                        should_extract_wasm = include_wasm and file_name.endswith(".wasm")
                        should_extract_mjs = include_mjs and file_name.endswith(".mjs")
                        if (
                            file_name not in required_names
                            and not should_extract_wasm
                            and not should_extract_mjs
                        ):
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        with source:
                            with (target_dir / file_name).open("wb") as target:
                                shutil.copyfileobj(source, target)
            except (OSError, tarfile.TarError) as exc:
                raise RuntimeError(
                    "Failed to extract frontend utilities. "
                    "Please download the missing files manually, or run this "
                    "server once on a machine with internet access. "
                    f"Package: {package_name}. Error: {exc}"
                ) from exc

        missing = [name for name in required_names if not (target_dir / name).exists()]
        if include_wasm and not any(target_dir.glob("*.wasm")):
            missing.append("*.wasm")
        if require_mjs and not any(target_dir.glob("*.mjs")):
            missing.append("*.mjs")
        if missing:
            raise RuntimeError(
                "Failed to prepare frontend utilities. "
                "Please download the missing files manually, or run this server "
                "once on a machine with internet access. "
                f"Package: {package_name}. Missing files: {', '.join(missing)}"
            )

    for version in ORT_VERSIONS:
        extract_npm_dist_files(
            package_name=f"onnxruntime-web@{version}",
            tarball_url=(
                "https://registry.npmjs.org/onnxruntime-web/-/"
                f"onnxruntime-web-{version}.tgz"
            ),
            target_dir=(
                FRONTEND_UTILITIES_DIR
                / "onnxruntime-web"
                / version
                / "dist"
            ),
            required_names={"ort.js"},
            include_wasm=True,
            include_mjs=version == "1.22.0",
            require_mjs=version == "1.22.0",
        )

    extract_npm_dist_files(
        package_name=f"@ricky0123/vad-web@{VAD_VERSION}",
        tarball_url=(
            "https://registry.npmjs.org/@ricky0123/vad-web/-/"
            f"vad-web-{VAD_VERSION}.tgz"
        ),
        target_dir=FRONTEND_UTILITIES_DIR / "vad-web" / VAD_VERSION / "dist",
        required_names={"bundle.min.js", "silero_vad_v5.onnx"},
    )

    fastenhancer_target = (
        FRONTEND_UTILITIES_DIR / "xtalk" / "models" / "fastenhancer_s.onnx"
    )
    if not fastenhancer_target.exists():
        download_file(FASTENHANCER_URL, fastenhancer_target)


parser = argparse.ArgumentParser(description="Xtalk Dev Server")
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()

app = FastAPI(title="Xtalk Dev Server")

with open(args.config, "r", encoding="utf-8") as config_file:
    server_config = json.load(config_file)
xtalk_instance = Xtalk.from_config(build_timer_config(server_config))
xtalk_instance.mount_routes(app)


logs_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(logs_path / "templates"))
app.mount("/static", StaticFiles(directory=str(logs_path / "static")), name="static")
download_frontend_utilities()
app.mount(
    FRONTEND_UTILITIES_ROUTE,
    StaticFiles(directory=str(FRONTEND_UTILITIES_DIR)),
    name="frontend_utilities",
)
try:
    app.mount(
        "/xtalk",
        StaticFiles(
            directory=str(Path(__file__).parent.parent.parent / "frontend" / "dist")
        ),
        name="xtalk",
    )
except Exception:
    print("No local Xtalk frontend library found. Will load frontend library from CDN.")


@app.get("/api/voices")
async def get_reference_audios():
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        try:
            voices = config["tts"]["params"]["voices"]
        except (KeyError, TypeError):
            voices = []
    return JSONResponse(content={"audios": voices})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
