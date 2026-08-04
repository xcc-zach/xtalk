"""Public XTalk runtime integration for the desktop sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xtalk import Xtalk

from .desktop_agent import DesktopDefaultAgent
from .tool_registry import load_enabled_tools
from .tool_ui import ToolUIBroker


def _register_managed_model_clients() -> None:
    """Import App-supported managed clients before generic model discovery.

    PyInstaller cannot rely on XTalk's dynamic package scan to import these
    modules: unrelated optional model modules may be unavailable in the slim
    desktop runtime. Direct imports both register the decorated model classes
    and make the required modules explicit frozen-backend dependencies.
    """

    from xtalk.models.asr.sherpa_onnx_asr import SherpaOnnxASR
    from xtalk.models.tts.moss_tts_nano import MossTTSNano
    from xtalk.models.tts.sherpa_onnx_tts import SherpaOnnxTTS

    # Keep references until all decorator side effects have completed.
    _ = (SherpaOnnxASR, MossTTSNano, SherpaOnnxTTS)


def build_xtalk_runtime(
    config: dict[str, Any],
    *,
    tools_root: Path | None = None,
    builtin_tools_root: Path | None = None,
    anonymous_user_id: str | None = None,
    tool_ui_broker: ToolUIBroker | None = None,
) -> Any:
    """Build XTalk with App-managed built-in and user-installed tools.

    Parameters
    ----------
    config : dict[str, Any]
        Fully composed generic XTalk configuration.
    tools_root : pathlib.Path | None, optional
        Application data directory containing user-installed tools.
    builtin_tools_root : pathlib.Path | None, optional
        Read-only App resource containing built-in tool directories.
    anonymous_user_id : str | None, optional
        Runtime-only stable identity for the built-in anonymous login.
    tool_ui_broker : ToolUIBroker | None, optional
        App-only observer for developer tools that declare a UI entrypoint.

    Returns
    -------
    Any
        Public XTalk runtime wrapper exposing ``mount_routes``.
    """

    _register_managed_model_clients()
    builder = Xtalk.configure(config)
    if config.get("llm_agent") is not None:
        builder.set_model(DesktopDefaultAgent)
        if tools_root is None and builtin_tools_root is None:
            tools = []
        else:
            resolved_tools_root = (
                tools_root
                if tools_root is not None
                else builtin_tools_root.parent / "user-tools"
            )
            tools = load_enabled_tools(
                resolved_tools_root,
                builtin_tools_root=builtin_tools_root,
                tool_ui_broker=tool_ui_broker,
            )
        builder.add_agent_tools(tools)
    runtime = builder.build()
    if anonymous_user_id is not None:
        # Keep the desktop-owned login identity outside user model/service config.
        runtime._anonymous_user_id = anonymous_user_id
    return runtime


def mount_xtalk_routes(runtime: Any, app: Any) -> None:
    """Mount public XTalk HTTP and WebSocket routes on a FastAPI app.

    Parameters
    ----------
    runtime : Any
        Runtime returned by :func:`build_xtalk_runtime`.
    app : Any
        FastAPI application receiving XTalk's public routes.
    """

    runtime.mount_routes(app)
