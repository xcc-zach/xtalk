import json
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Callable, Type, Any
from fastapi import (
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    status,
)
from langchain_core.tools import BaseTool

from .auth import JWTAuth, extract_bearer_token, resolve_auth_config
from .persistence import PersistenceStore
from .serving.service_manager import ServiceManager
from .pipelines import Pipeline
from .pipelines.default import DefaultPipeline
from .serving.session_limiter import SessionLimiter
from .serving.events import TextForEmbeddingReady
from .serving.service import Service, DefaultService
from .model_loader import (
    ensure_model_types_registered,
    init_configured_model,
    is_registered_model_slot,
    _registered_model_slots,
)


class Xtalk:
    """Create Xtalk pipelines, services, and session entrypoints.

    Notes
    -----
    ``Xtalk`` is the main integration surface used by the sample applications.
    It builds pipelines from configuration, stores a prototype service, and
    accepts WebSocket sessions on demand.
    """
    def __init__(self, *, service_prototype: Service, max_sessions: int | None = None):
        """Initialize an ``Xtalk`` application wrapper.

        Parameters
        ----------
        service_prototype : Service
            Prototype service used to clone per-session service instances.
        max_sessions : int | None, optional
            Maximum number of concurrent sessions. If omitted, no session limit
            is enforced.
        """
        service_config = dict(service_prototype.service_config)
        self._persistence_enabled = self._is_persistence_enabled(service_config)
        data_dir = service_config.get("data_dir") or "data"
        auth_secret, auth_ttl_seconds = resolve_auth_config(service_config)

        self._persistence = (
            PersistenceStore(Path(data_dir).expanduser().resolve() / "chat_history.sqlite3")
            if self._persistence_enabled
            else None
        )
        self._auth = JWTAuth(secret=auth_secret, ttl_seconds=auth_ttl_seconds)
        self._service_manager = ServiceManager(
            service_prototype=service_prototype,
            persistence_store=self._persistence,
        )
        self._pipeline = service_prototype.pipeline
        self._session_limiter = (
            SessionLimiter(max_sessions) if max_sessions is not None else None
        )

    # -------------------------
    # Config / construction
    # -------------------------
    @staticmethod
    def _get_config_dict(path_or_dict: str | dict) -> dict:
        if isinstance(path_or_dict, str):
            with open(path_or_dict, "r") as f:
                config = json.load(f)
        else:
            config = path_or_dict
        return config

    @classmethod
    def from_config(cls, path_or_dict: str | dict) -> "Xtalk":
        """Build an ``Xtalk`` instance from configuration data.

        Parameters
        ----------
        path_or_dict : str | dict
            JSON file path or already loaded configuration dictionary.

        Returns
        -------
        Xtalk
            Configured application wrapper backed by a ``DefaultPipeline`` and
            ``DefaultService``.

        Examples
        --------
        >>> xtalk = Xtalk.from_config("server_config.json")
        """
        config = cls._get_config_dict(path_or_dict)
        pipeline = cls._load_pipeline(DefaultPipeline, config)
        service_prototype = DefaultService(
            pipeline=pipeline, service_config=cls._load_service_config(config)
        )
        max_sessions = cls._max_sessions(config)
        return cls(service_prototype=service_prototype, max_sessions=max_sessions)

    @classmethod
    def create_pipeline_from_config(
        cls,
        *,
        pipeline_cls: Type[Pipeline],
        config_path_or_dict: str | dict,
        additional_model_registry: dict[str, Any],
    ) -> Pipeline:
        """Instantiate a custom pipeline class from configuration.

        Parameters
        ----------
        pipeline_cls : Type[Pipeline]
            Concrete pipeline type to instantiate.
        config_path_or_dict : str | dict
            JSON file path or already loaded configuration dictionary.
        additional_model_registry : dict[str, Any]
            Extra slot-to-instance mapping merged on top of the default model
            registry before the pipeline is created.

        Returns
        -------
        Pipeline
            Pipeline instance created from the supplied configuration.

        Examples
        --------
        >>> pipeline = Xtalk.create_pipeline_from_config(
        ...     pipeline_cls=DefaultPipeline,
        ...     config_path_or_dict="server_config.json",
        ...     additional_model_registry={},
        ... )
        """
        config = cls._get_config_dict(config_path_or_dict)
        pipeline = cls._load_pipeline(pipeline_cls, config, additional_model_registry)
        return pipeline

    def set_session_limit(self, limit: int):
        """Set or replace the concurrent session limit.

        Parameters
        ----------
        limit : int
            Maximum number of active sessions allowed at the same time.
        """
        self._session_limiter = SessionLimiter(limit)

    async def embed_text(self, session_id: str, text: str, user_id: str | None = None):
        """Queue text for session-scoped embedding storage.

        Parameters
        ----------
        session_id : str
            Session identifier returned to the frontend.
        text : str
            Text content that should be embedded and persisted for retrieval.

        Raises
        ------
        ValueError
            Raised if the target session does not exist.
        """
        if (
            self._persistence is not None
            and user_id is not None
            and not self._persistence.user_owns_session(
            user_id, session_id
            )
        ):
            raise ValueError(f"Session {session_id} not found for user {user_id}.")
        service = self._service_manager.get_service(session_id)
        if service is None:
            raise ValueError(f"Session {session_id} not found.")
        await service.event_bus.publish(
            TextForEmbeddingReady(session_id=session_id, text=text)
        )

    def add_agent_tools(
        self, tools_or_factories: list[BaseTool | Callable[[], BaseTool]]
    ):
        """Attach tools to the prototype agent before sessions are created.

        Parameters
        ----------
        tools_or_factories : list[BaseTool | Callable[[], BaseTool]]
            Tool instances or zero-argument factories that produce tool
            instances.

        Raises
        ------
        RuntimeError
            Raised if at least one service session has already been created.
        """
        if self._service_manager.get_service_count() > 0:
            raise RuntimeError("Cannot add tools after services have been created.")
        self._pipeline.get_agent().add_tools(tools_or_factories)

    def _login(self) -> dict[str, Any]:
        user_id = str(uuid.uuid4())
        user = (
            self._persistence.ensure_user(user_id)
            if self._persistence is not None
            else {"id": user_id}
        )
        return {
            "access_token": self._auth.issue_token(sub=user_id),
            "user": user,
        }

    def _verify_access_token(self, token: str) -> str:
        user_id = self._auth.verify_token(token)
        if self._persistence is not None:
            self._persistence.ensure_user(user_id)
        return user_id

    def _list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        if self._persistence is None:
            return []
        self._persistence.ensure_user(user_id)
        return self._persistence.list_sessions(user_id)

    def _get_session_detail(self, user_id: str, session_id: str) -> dict[str, Any]:
        if self._persistence is None:
            raise KeyError(session_id)
        self._persistence.ensure_user(user_id)
        return self._persistence.get_session_detail(user_id, session_id)

    def mount_routes(
        self,
        app: Any,
        *,
        login_path: str = "/api/auth/login",
        sessions_path: str = "/api/sessions",
        session_detail_path: str = "/api/sessions/{session_id}",
        upload_path: str = "/api/upload",
        ws_path: str = "/ws",
    ) -> None:
        """Mount the built-in auth, session, upload, and websocket routes."""

        def _require_http_user(request: Request) -> str:
            token = extract_bearer_token(request.headers.get("authorization"))
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing bearer token",
                )
            try:
                return self._verify_access_token(token)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(exc),
                ) from exc

        def _require_ws_user(websocket: WebSocket) -> str:
            token = websocket.query_params.get("access_token")
            if token is None:
                token = extract_bearer_token(websocket.headers.get("authorization"))
            if not token:
                raise ValueError("Missing access token")
            return self._verify_access_token(token)

        @app.post(login_path)
        async def _login_route() -> dict[str, Any]:
            return self._login()

        @app.get(sessions_path)
        async def _list_sessions_route(request: Request) -> dict[str, Any]:
            user_id = _require_http_user(request)
            return {"sessions": self._list_sessions(user_id)}

        @app.get(session_detail_path)
        async def _session_detail_route(
            request: Request, session_id: str
        ) -> dict[str, Any]:
            user_id = _require_http_user(request)
            try:
                return self._get_session_detail(user_id, session_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                ) from exc

        @app.post(upload_path)
        async def _upload_route(
            request: Request,
            session_id: str = Form(...),
            file: UploadFile = File(...),
        ) -> dict[str, str]:
            user_id = _require_http_user(request)
            content_type = (file.content_type or "").lower()
            is_text = content_type.startswith("text/") if content_type else False
            if content_type and not is_text:
                raise HTTPException(
                    status_code=400, detail="Only text files are supported."
                )
            text = (await file.read()).decode("utf-8", errors="ignore")
            try:
                await self.embed_text(session_id=session_id, text=text, user_id=user_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"status": "ok"}

        @app.websocket(ws_path)
        async def _websocket_route(websocket: WebSocket) -> None:
            try:
                user_id = _require_ws_user(websocket)
            except ValueError:
                await websocket.accept()
                await websocket.close(code=1008, reason="Unauthorized")
                return
            await self.connect(websocket, user_id=user_id)

    async def connect(self, websocket: WebSocket, user_id: str | None = None):
        """Accept a WebSocket session and hand it to the service manager.

        Parameters
        ----------
        websocket : WebSocket
            FastAPI WebSocket connection from the client.
        user_id : str | None, optional
            Authenticated user identifier. When omitted, the connection falls
            back to the legacy connection-scoped session behavior.

        Notes
        -----
        If a session limit is configured, the socket is first admitted through
        the session limiter queue.
        """
        if self._session_limiter:
            await websocket.accept()
            waiter = await self._session_limiter.acquire(websocket)
            if waiter is None:
                try:
                    await websocket.close(code=1000)
                except Exception:
                    pass
                return
            await self._service_manager.connect(
                websocket=websocket, already_accepted=True, user_id=user_id
            )
            await self._session_limiter.release(waiter)
            return

        await self._service_manager.connect(websocket=websocket, user_id=user_id)

    @staticmethod
    def _max_sessions(config: dict):
        if "max_connections" in config:
            try:
                return int(config["max_connections"])
            except Exception:
                pass
        return None

    @staticmethod
    def _load_service_config(config: dict):
        return config.get("service_config", {})

    @staticmethod
    def _is_persistence_enabled(service_config: dict[str, Any]) -> bool:
        """Return whether session persistence is enabled for the service."""
        value = service_config.get("enable_persistence")
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"0", "false", "no", "off"}:
                return False
            if normalized in {"1", "true", "yes", "on"}:
                return True
        return bool(value)

    @classmethod
    def _load_pipeline(
        cls,
        pipeline_cls: Type[Pipeline],
        config: dict,
        additional_model_registry: dict | None = None,
    ):
        model_map: dict[str, Any] = {}
        for slot in cls._pipeline_model_init_keys(pipeline_cls):
            model_map[slot] = init_configured_model(
                slot=slot,
                config=config,
            )

        if additional_model_registry:
            model_map = model_map | additional_model_registry

        return pipeline_cls(**model_map)

    @staticmethod
    def _pipeline_model_init_keys(pipeline_cls: Type[Pipeline]) -> list[str]:
        """Return pipeline constructor keys that correspond to model types."""
        if not is_dataclass(pipeline_cls):
            ensure_model_types_registered()
            return [
                slot
                for slot in _registered_model_slots()
                if is_registered_model_slot(slot)
            ]

        init_keys: list[str] = []
        for dataclass_field in fields(pipeline_cls):
            init_key = dataclass_field.metadata.get("init_key")
            if not isinstance(init_key, str):
                continue
            if is_registered_model_slot(init_key):
                init_keys.append(init_key)
        return init_keys
