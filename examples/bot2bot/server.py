from __future__ import annotations

import argparse
import copy
import json
import logging
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from xtalk import Xtalk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")


@dataclass
class BotDraft:
    """Editable bot settings submitted by the demo frontend.

    Parameters
    ----------
    name : str
        Display name shown in the browser timeline.
    system_prompt : str
        Per-bot prompt injected into ``llm_agent.params.system_prompt``.
    proactive : bool
        Whether the bot should proactively start the conversation.
    voice : str
        Optional TTS voice override injected into ``tts.params.voice``.
    """

    name: str
    system_prompt: str
    proactive: bool
    voice: str


@dataclass
class ActiveBotRuntime:
    """One active bot runtime backed by a dedicated Xtalk sub-application.

    Parameters
    ----------
    bot_id : str
        Stable bot identifier used by the frontend bridge and runtime URLs.
    name : str
        Display name returned to the browser.
    app : FastAPI
        Sub-application exposing standard Xtalk auth, session, upload, and
        websocket routes.
    """

    bot_id: str
    name: str
    app: FastAPI


@dataclass
class ActiveRun:
    """Active bot-to-bot run published through the runtime dispatcher.

    Parameters
    ----------
    run_id : str
        Unique run identifier embedded into runtime URLs.
    bots : dict[str, ActiveBotRuntime]
        Bot runtimes keyed by bot id.
    """

    run_id: str
    bots: dict[str, ActiveBotRuntime]


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON object from disk.

    Parameters
    ----------
    path : str
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    """

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_template_vad_config(template_config: dict[str, Any]) -> None:
    """Ensure the bot2bot template config defines a backend VAD model.

    Parameters
    ----------
    template_config : dict[str, Any]
        Template server config passed through ``--config``.

    Raises
    ------
    ValueError
        Raised when the template config does not define a valid ``vad`` block.
    """

    vad_config = template_config.get("vad")
    if isinstance(vad_config, dict) and vad_config.get("type"):
        return

    raise ValueError(
        "Bot2Bot requires backend VAD in the template config. "
        "Install the dependency with `pip install -e '.[silero-vad]'`, then add "
        'the following block to your config:\n'
        '"vad": {\n'
        '    "type": "SileroVAD",\n'
        '    "params": {}\n'
        "}"
    )


def parse_bot_drafts(payload: dict[str, Any]) -> list[BotDraft]:
    """Parse and validate a bot list from the frontend payload.

    Parameters
    ----------
    payload : dict[str, Any]
        JSON payload submitted to the start endpoint.

    Returns
    -------
    list[BotDraft]
        Normalized bot settings.

    Raises
    ------
    ValueError
        Raised when the payload is malformed or violates demo constraints.
    """

    raw_bots = payload.get("bots")
    if not isinstance(raw_bots, list):
        raise ValueError("Payload must contain a bots array.")
    if len(raw_bots) < 2:
        raise ValueError("At least two bots are required.")

    bots: list[BotDraft] = []
    proactive_count = 0
    for index, raw_bot in enumerate(raw_bots, start=1):
        if not isinstance(raw_bot, dict):
            raise ValueError(f"Bot {index} must be an object.")
        name_value = raw_bot.get("name")
        system_prompt_value = raw_bot.get("system_prompt")
        proactive_value = raw_bot.get("proactive")
        voice_value = raw_bot.get("voice")

        name = name_value.strip() if isinstance(name_value, str) else ""
        system_prompt = (
            system_prompt_value.strip()
            if isinstance(system_prompt_value, str)
            else ""
        )
        proactive = bool(proactive_value)
        voice = voice_value.strip() if isinstance(voice_value, str) else ""

        if not name:
            raise ValueError(f"Bot {index} name must be non-empty.")
        if not system_prompt:
            raise ValueError(f"Bot {index} system prompt must be non-empty.")

        if proactive:
            proactive_count += 1
        bots.append(
            BotDraft(
                name=name,
                system_prompt=system_prompt,
                proactive=proactive,
                voice=voice,
            )
        )

    if proactive_count != 1:
        raise ValueError("Exactly one bot must set proactive=true.")
    return bots


def build_runtime_paths(run_id: str, bot_id: str) -> dict[str, str]:
    """Build standard Xtalk route paths for one runtime bot.

    Parameters
    ----------
    run_id : str
        Active run id.
    bot_id : str
        Bot identifier inside the run.

    Returns
    -------
    dict[str, str]
        Standard route paths rooted at ``/runtime/{run_id}/{bot_id}``.
    """

    base_path = f"/runtime/{run_id}/{bot_id}"
    return {
        "base_path": base_path,
        "websocket_path": f"{base_path}/ws",
        "login_path": f"{base_path}/api/auth/login",
        "sessions_path": f"{base_path}/api/sessions",
        "upload_path": f"{base_path}/api/upload",
    }


def build_bot_config(template_config: dict[str, Any], bot: BotDraft) -> dict[str, Any]:
    """Create one per-bot config by overriding the template agent params.

    Parameters
    ----------
    template_config : dict[str, Any]
        Template config passed through ``--config``.
    bot : BotDraft
        Normalized bot settings.

    Returns
    -------
    dict[str, Any]
        Deep-copied config ready for ``Xtalk.from_config()``.
    """

    config = copy.deepcopy(template_config)
    llm_agent_config = config.get("llm_agent")
    if not isinstance(llm_agent_config, dict):
        raise ValueError("Template config must define llm_agent.")

    llm_agent_params = llm_agent_config.get("params")
    if not isinstance(llm_agent_params, dict):
        llm_agent_params = {}
        llm_agent_config["params"] = llm_agent_params

    llm_agent_params["system_prompt"] = bot.system_prompt
    llm_agent_params["proactive"] = bot.proactive

    if bot.voice:
        tts_config = config.get("tts")
        if not isinstance(tts_config, dict):
            tts_config = {}
            config["tts"] = tts_config

        tts_params = tts_config.get("params")
        if not isinstance(tts_params, dict):
            tts_params = {}
            tts_config["params"] = tts_params

        tts_params["voice"] = bot.voice
    return config


class BotRuntimeDispatcher:
    """Dispatch ``/runtime/{run_id}/{bot_id}/...`` traffic to active sub-apps.

    Parameters
    ----------
    manager : Bot2BotRuntimeManager
        Runtime manager that resolves the currently active bot sub-app.
    """

    def __init__(self, manager: "Bot2BotRuntimeManager") -> None:
        self.manager = manager

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """Route an incoming HTTP or websocket scope to the active bot app.

        Parameters
        ----------
        scope : dict[str, Any]
            Incoming ASGI scope.
        receive : Any
            ASGI receive callable.
        send : Any
            ASGI send callable.
        """

        path = scope.get("path", "/")
        parts = [segment for segment in path.split("/") if segment]
        if parts and parts[0] == "runtime":
            parts = parts[1:]
        if len(parts) < 3:
            await self._respond_not_found(scope, receive, send)
            return

        run_id = parts[0]
        bot_id = parts[1]
        child_app = self.manager.resolve_bot_app(run_id=run_id, bot_id=bot_id)
        if child_app is None:
            await self._respond_not_found(scope, receive, send)
            return

        child_path = "/" + "/".join(parts[2:])
        child_scope = dict(scope)
        child_scope["path"] = child_path
        child_scope["root_path"] = f"{scope.get('root_path', '')}/{run_id}/{bot_id}"
        raw_path = scope.get("raw_path")
        if isinstance(raw_path, bytes):
            child_scope["raw_path"] = child_path.encode("utf-8")
        await child_app(child_scope, receive, send)

    async def _respond_not_found(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """Return a simple 404 response for missing runtime paths.

        Parameters
        ----------
        scope : dict[str, Any]
            Incoming ASGI scope.
        receive : Any
            ASGI receive callable.
        send : Any
            ASGI send callable.
        """

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        response = PlainTextResponse("Runtime not found", status_code=404)
        await response(scope, receive, send)


class Bot2BotRuntimeManager:
    """Manage one active bot-to-bot run and its per-bot Xtalk sub-apps.

    Parameters
    ----------
    template_config : dict[str, Any]
        Template config loaded from ``--config``.
    agent_type : str
        Human-readable configured agent type.
    template_system_prompt : str
        Template prompt used to seed frontend defaults.
    """

    def __init__(
        self,
        *,
        template_config: dict[str, Any],
        agent_type: str,
    ) -> None:
        self.template_config = template_config
        self.agent_type = agent_type
        self.active_run: ActiveRun | None = None

    def describe_template(self) -> dict[str, Any]:
        """Return frontend metadata derived from the template config.

        Returns
        -------
        dict[str, Any]
            Template metadata and starter bot defaults.
        """

        return {
            "agent_type": self.agent_type,
            "supports_proactive": True,
            "default_bots": [
                {
                    "name": "Bot A",
                    "system_prompt": "",
                    "proactive": True,
                    "voice": "",
                },
                {
                    "name": "Bot B",
                    "system_prompt": "",
                    "proactive": False,
                    "voice": "",
                },
            ],
        }

    def start(self, bots: list[BotDraft]) -> dict[str, Any]:
        """Create a fresh active run from the template config.

        Parameters
        ----------
        bots : list[BotDraft]
            Normalized bot settings submitted by the frontend.

        Returns
        -------
        dict[str, Any]
            Run metadata returned to the browser.
        """

        self.stop()

        run_id = uuid.uuid4().hex
        runtimes: dict[str, ActiveBotRuntime] = {}
        response_bots: list[dict[str, Any]] = []

        for index, bot in enumerate(bots, start=1):
            bot_id = f"bot-{index}"
            bot_config = build_bot_config(self.template_config, bot)
            xtalk_instance = Xtalk.from_config(bot_config)
            sub_app = FastAPI(title=f"Xtalk Bot2Bot Runtime {bot_id}")
            xtalk_instance.mount_routes(sub_app)

            runtimes[bot_id] = ActiveBotRuntime(
                bot_id=bot_id,
                name=bot.name,
                app=sub_app,
            )
            runtime_paths = build_runtime_paths(run_id, bot_id)
            response_bots.append(
                {
                    "id": bot_id,
                    "name": bot.name,
                    "proactive": bot.proactive,
                    **runtime_paths,
                }
            )

        self.active_run = ActiveRun(run_id=run_id, bots=runtimes)
        return {
            "run_id": run_id,
            "bots": response_bots,
        }

    def stop(self) -> None:
        """Drop the current active run from the dispatcher."""

        self.active_run = None

    def resolve_bot_app(self, *, run_id: str, bot_id: str) -> FastAPI | None:
        """Resolve one active bot sub-app for the runtime dispatcher.

        Parameters
        ----------
        run_id : str
            Requested run id from the incoming path.
        bot_id : str
            Requested bot id from the incoming path.

        Returns
        -------
        FastAPI | None
            Matching sub-app when the run is active, else ``None``.
        """

        active_run = self.active_run
        if active_run is None or active_run.run_id != run_id:
            return None
        runtime = active_run.bots.get(bot_id)
        return runtime.app if runtime is not None else None


def create_app(config_path: str) -> FastAPI:
    """Create the bot-to-bot demo application.

    Parameters
    ----------
    config_path : str
        Template config file passed through ``--config``.

    Returns
    -------
    FastAPI
        Configured demo server.
    """

    template_config = load_json(config_path)
    validate_template_vad_config(template_config)
    llm_agent_config = template_config.get("llm_agent")
    agent_type = ""
    if isinstance(llm_agent_config, dict):
        raw_agent_type = llm_agent_config.get("type")
        if isinstance(raw_agent_type, str):
            agent_type = raw_agent_type
    manager = Bot2BotRuntimeManager(
        template_config=template_config,
        agent_type=agent_type,
    )

    app = FastAPI(title="Xtalk Bot2Bot Demo")

    example_root = Path(__file__).parent
    repo_root = example_root.parent.parent
    frontend_dist = repo_root / "frontend" / "dist"
    templates = Jinja2Templates(directory=str(example_root / "templates"))
    app.mount(
        "/static",
        StaticFiles(directory=str(example_root / "static")),
        name="static",
    )
    if frontend_dist.exists():
        app.mount("/xtalk", StaticFiles(directory=str(frontend_dist)), name="xtalk")

    app.mount("/runtime", BotRuntimeDispatcher(manager), name="runtime")

    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request) -> HTMLResponse:
        """Render the bot-to-bot demo page.

        Parameters
        ----------
        request : Request
            Current HTTP request.

        Returns
        -------
        HTMLResponse
            Rendered demo HTML.
        """

        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/api/bot2bot/template")
    async def read_template() -> dict[str, Any]:
        """Return frontend starter defaults derived from the template config.

        Returns
        -------
        dict[str, Any]
            Frontend defaults and capability metadata.
        """

        return manager.describe_template()

    @app.post("/api/bot2bot/start")
    async def start_bot2bot(request: Request) -> dict[str, Any]:
        """Create one active bot-to-bot run from the submitted bot list.

        Parameters
        ----------
        request : Request
            Incoming HTTP request with the bot payload.

        Returns
        -------
        dict[str, Any]
            Run metadata including one websocket path per bot.

        Raises
        ------
        HTTPException
            Raised when the payload is invalid.
        """

        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be a JSON object.")
        try:
            bots = parse_bot_drafts(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            return manager.start(bots)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/bot2bot/stop")
    async def stop_bot2bot() -> dict[str, str]:
        """Stop the active run.

        Returns
        -------
        dict[str, str]
            Stop status payload.
        """

        manager.stop()
        return {"status": "stopped"}

    @app.get("/api/bot2bot/health")
    async def healthcheck() -> dict[str, str]:
        """Return a lightweight health payload for the example UI.

        Returns
        -------
        dict[str, str]
            Static health response.
        """

        return {"status": "ok"}

    return app


def main() -> None:
    """Parse flags and run the demo server."""

    parser = argparse.ArgumentParser(description="Xtalk Bot-to-Bot Demo")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the template server config JSON, for example server_configs/experimental.json",
    )
    parser.add_argument("--port", type=int, default=11996, help="Port to listen on")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.config), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
