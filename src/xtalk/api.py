from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .model_loader import (
    ensure_model_types_registered,
    init_configured_model,
)
from .models.agents.tools import Tool
from .models.container import Models
from .models.registry import ModelImplInfo
from .realtime.auth import authorize
from .realtime.config import RealtimeConfig
from .realtime.connection import RealtimeConnection
from .serving.service import DefaultService, Service
from .serving.service_manager import ServiceManager
from .serving.session_limiter import SessionLimiter

_ConfigTransform = Callable[[dict[str, Any]], dict[str, Any]]


def _copy_config_value(value: Any) -> Any:
    """Copy JSON-style containers while retaining runtime object identities."""

    if isinstance(value, dict):
        return {key: _copy_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_config_value(item) for item in value)
    return value


class XtalkBuilder:
    """Collect staged configuration changes and runtime-only Xtalk bindings.

    Parameters
    ----------
    xtalk_type : type[Xtalk]
        ``Xtalk`` class, or a subclass, created by ``build()``.
    path_or_dict : str | dict
        JSON file path or already loaded configuration dictionary.
    """

    def __init__(
        self,
        *,
        xtalk_type: type[Xtalk],
        path_or_dict: str | dict,
    ) -> None:
        self._xtalk_type = xtalk_type
        self._path_or_dict = path_or_dict
        self._config_transforms: list[_ConfigTransform] = []

    def transform_config(
        self,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> XtalkBuilder:
        """Append an arbitrary configuration transformation.

        Transformations run in registration order when ``build()`` is called.
        The first transformation receives a structural copy of the source
        configuration, so it may either mutate and return that copy or return
        a new dictionary.

        Parameters
        ----------
        transform : Callable[[dict[str, Any]], dict[str, Any]]
            Function that receives the current effective configuration and
            returns the configuration for the next build stage.

        Returns
        -------
        XtalkBuilder
            This builder, allowing fluent configuration calls.
        """

        if not callable(transform):
            raise TypeError("config transform must be callable.")
        self._config_transforms.append(transform)
        return self

    def set_model(self, model_class: type[Any]) -> XtalkBuilder:
        """Replace a configured model with a registered Python model class.

        The model slot and canonical configuration name are inferred from the
        class's ``@xtalk.model`` registration. Existing model parameters and
        other model-level configuration fields are preserved.

        Parameters
        ----------
        model_class : type[Any]
            Model implementation class decorated with ``@xtalk.model``.

        Returns
        -------
        XtalkBuilder
            This builder, allowing fluent configuration calls.

        Raises
        ------
        TypeError
            Raised when ``model_class`` is not a registered model
            implementation.
        """

        if not isinstance(model_class, type):
            raise TypeError("model_class must be a model implementation class.")

        ensure_model_types_registered()
        from .models.registry import _get_model_impl_info

        model_info = _get_model_impl_info(model_class)
        if model_info is None:
            raise TypeError(
                f"{model_class.__name__} must be registered with @xtalk.model."
            )

        def replace_model(config: dict[str, Any]) -> dict[str, Any]:
            return self._replace_model(config, model_info)

        return self.transform_config(replace_model)

    def add_agent_tools(
        self,
        tools: list[Tool | Callable[[], Tool]],
    ) -> XtalkBuilder:
        """Append runtime-only tools to the configured Agent.

        Parameters
        ----------
        tools : list[Tool | Callable[[], Tool]]
            LangChain tool instances, native Xtalk tool classes, or factories
            that create either kind of tool.

        Returns
        -------
        XtalkBuilder
            This builder, allowing fluent configuration calls.
        """

        registered_tools = list(tools)

        def attach_agent_tools(config: dict[str, Any]) -> dict[str, Any]:
            return self._attach_agent_tools(config, registered_tools)

        return self.transform_config(attach_agent_tools)

    def build(self) -> Xtalk:
        """Instantiate Xtalk after applying all staged configuration changes.

        Returns
        -------
        Xtalk
            Configured application wrapper.
        """

        source_config = self._xtalk_type._get_config_dict(self._path_or_dict)
        if not isinstance(source_config, dict):
            raise TypeError("Xtalk config must be a dictionary.")

        effective_config = _copy_config_value(source_config)
        for transform in self._config_transforms:
            transformed_config = transform(effective_config)
            if not isinstance(transformed_config, dict):
                raise TypeError("config transform must return a dictionary.")
            effective_config = transformed_config

        return self._xtalk_type._build_from_config_dict(effective_config)

    @staticmethod
    def _replace_model(
        config: dict[str, Any],
        model_info: ModelImplInfo,
    ) -> dict[str, Any]:
        """Return a config copy using one registered model implementation."""

        from .models.registry import resolve_config_slot

        model_slot = resolve_config_slot(model_info.config_key, config)
        configured_model = config.get(model_slot)
        if configured_model is None:
            model_config: dict[str, Any] = {"params": {}}
        elif isinstance(configured_model, str):
            model_config = {"params": {}}
        elif isinstance(configured_model, dict):
            model_config = dict(configured_model)
            configured_params = model_config.get("params")
            if configured_params is None:
                model_config["params"] = {}
            elif isinstance(configured_params, dict):
                model_config["params"] = dict(configured_params)
            else:
                raise ValueError(f"{model_slot}.params must be an object.")
        else:
            raise ValueError(f"{model_slot} must be a model name or an object.")

        model_config["type"] = model_info.name
        effective_config = dict(config)
        effective_config[model_slot] = model_config
        return effective_config

    @staticmethod
    def _attach_agent_tools(
        config: dict[str, Any],
        tools: list[Tool | Callable[[], Tool]],
    ) -> dict[str, Any]:
        """Return a structural config copy containing registered Agent tools."""

        if not tools:
            return config

        ensure_model_types_registered()
        from .models.registry import resolve_config_slot

        agent_slot = resolve_config_slot("llm_agent", config)
        if agent_slot not in config:
            raise ValueError("The config must define an llm_agent model.")

        configured_agent = config[agent_slot]
        if isinstance(configured_agent, str):
            agent_config: dict[str, Any] = {
                "type": configured_agent,
                "params": {},
            }
        elif isinstance(configured_agent, dict):
            agent_config = dict(configured_agent)
        else:
            raise ValueError(f"{agent_slot} must be a model name or an object.")  # noqa: TRY004 - preserve builder validation contract

        configured_params = agent_config.get("params")
        if configured_params is None:
            params: dict[str, Any] = {}
        elif isinstance(configured_params, dict):
            params = dict(configured_params)
        else:
            raise ValueError(f"{agent_slot}.params must be an object.")

        configured_tools = params.get("tools", [])
        if not isinstance(configured_tools, list):
            raise ValueError(f"{agent_slot}.params.tools must be a list.")  # noqa: TRY004 - preserve builder validation contract
        params["tools"] = [*configured_tools, *tools]
        agent_config["params"] = params

        effective_config = dict(config)
        effective_config[agent_slot] = agent_config
        return effective_config


class Xtalk:
    """Create Xtalk model services and session entrypoints.

    Notes
    -----
    ``Xtalk`` is the main integration surface used by the sample applications.
    It builds model containers from configuration, stores a prototype service, and
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
        self._realtime_config = RealtimeConfig.from_dict(
            service_config.get("realtime_api", {})
        )
        self._service_manager = ServiceManager(service_prototype=service_prototype)
        self._asgi_app = None
        self._models = service_prototype.models
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
    def configure(cls, path_or_dict: str | dict) -> XtalkBuilder:
        """Start a staged Xtalk configuration.

        Parameters
        ----------
        path_or_dict : str | dict
            JSON file path or already loaded configuration dictionary.

        Returns
        -------
        XtalkBuilder
            Builder that accepts transformations and runtime-only bindings
            before model creation.

        Examples
        --------
        >>> builder = Xtalk.configure("server_config.json")
        """

        return XtalkBuilder(xtalk_type=cls, path_or_dict=path_or_dict)

    @classmethod
    def from_config(cls, path_or_dict: str | dict) -> Xtalk:
        """Build an ``Xtalk`` instance from configuration data.

        This is equivalent to ``Xtalk.configure(path_or_dict).build()``.

        Parameters
        ----------
        path_or_dict : str | dict
            JSON file path or already loaded configuration dictionary.

        Returns
        -------
        Xtalk
            Configured application wrapper backed by a ``DefaultService``.

        Examples
        --------
        >>> xtalk = Xtalk.from_config("server_config.json")
        """

        return cls.configure(path_or_dict).build()

    @classmethod
    def _build_from_config_dict(cls, config: dict) -> Xtalk:
        """Build an Xtalk instance from an effective config dictionary."""

        models = cls._load_models(config)
        service_prototype = DefaultService(
            models=models, service_config=cls._load_service_config(config)
        )
        max_sessions = cls._max_sessions(config)
        return cls(service_prototype=service_prototype, max_sessions=max_sessions)

    @classmethod
    def create_models_from_config(
        cls,
        *,
        config_path_or_dict: str | dict,
        additional_models: dict[type[Any], Any] | None = None,
    ) -> Models:
        """Instantiate configured models from configuration.

        Parameters
        ----------
        config_path_or_dict : str | dict
            JSON file path or already loaded configuration dictionary.
        additional_models : dict[type[Any], Any] | None, optional
            Extra interface-to-instance mappings merged into the configured
            models.

        Returns
        -------
        Models
            Model container created from the supplied configuration.

        Examples
        --------
        >>> models = Xtalk.create_models_from_config(
        ...     config_path_or_dict="server_config.json",
        ...     additional_models={},
        ... )
        """
        config = cls._get_config_dict(config_path_or_dict)
        return cls._load_models(config, additional_models=additional_models)

    def set_session_limit(self, limit: int):
        """Set or replace the concurrent session limit.

        Parameters
        ----------
        limit : int
            Maximum number of active sessions allowed at the same time.
        """
        self._session_limiter = SessionLimiter(limit)

    def create_asgi_app(self):
        """Return the cached framework-independent ASGI application.

        Returns
        -------
        callable
            HTTP, WebSocket and lifespan ASGI application.
        """
        if self._asgi_app is not None:
            return self._asgi_app
        import asyncio
        from urllib.parse import parse_qs

        started = False
        closed = False
        tasks = set()

        async def application(scope, receive, send):
            nonlocal started, closed
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        if started or closed:
                            await send(
                                {
                                    "type": "lifespan.startup.failed",
                                    "message": "Application already started or closed",
                                }
                            )
                            return
                        started = True
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        started = False
                        closed = True
                        for task in tuple(tasks):
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            path = scope["path"]
            root = scope.get("root_path", "")
            if root and path.startswith(root + "/"):
                path = path[len(root) :]
            allowed = authorize(scope, self._realtime_config.api_keys)
            if scope["type"] == "http":
                code = (
                    503
                    if not started
                    else 401
                    if not allowed
                    else 200
                    if path == "/v1/models" and scope["method"] == "GET"
                    else 404
                )
                payload = (
                    {
                        "object": "list",
                        "data": [
                            {"id": self._realtime_config.model, "object": "model"}
                        ],
                    }
                    if code == 200
                    else {"error": {"code": str(code)}}
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": code,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {"type": "http.response.body", "body": json.dumps(payload).encode()}
                )
                return
            if scope["type"] != "websocket":
                return
            await receive()
            model = parse_qs(
                scope.get("query_string", b"").decode("utf-8", errors="replace"),
                keep_blank_values=True,
            ).get("model", [self._realtime_config.model])[0]
            if (
                not started
                or not allowed
                or path != "/v1/realtime"
                or model != self._realtime_config.model
            ):
                await send({"type": "websocket.close", "code": 1008})
                return
            connection = RealtimeConnection(receive, send)
            await connection.accept()
            task = asyncio.current_task()
            tasks.add(task)
            waiter = None
            try:
                if self._session_limiter:
                    waiter = await self._session_limiter.acquire(connection)
                    if waiter is None:
                        return
                await self._service_manager._connect(connection, already_accepted=True)
            finally:
                if self._session_limiter:
                    await self._session_limiter.release(waiter)
                tasks.discard(task)
                await connection.close()

        self._asgi_app = application
        return application

    @staticmethod
    def _max_sessions(config: dict):
        if "max_connections" in config:
            try:
                return int(config["max_connections"])
            except (TypeError, ValueError):
                raise ValueError("max_connections must be an integer") from None
        return None

    @staticmethod
    def _load_service_config(config: dict):
        return config.get("service_config", {})

    @classmethod
    def _load_models(
        cls,
        config: dict,
        additional_models: dict[type[Any], Any] | None = None,
    ) -> Models:
        """Instantiate configured model types into a model container."""
        ensure_model_types_registered()

        from .models.registry import iter_model_type_infos, resolve_config_slot

        models = Models()
        for info in iter_model_type_infos():
            config_slot = resolve_config_slot(info.config_key, config)
            if config_slot not in config:
                continue
            models.set(
                info.interface,
                init_configured_model(slot=info.config_key, config=config),
            )

        if additional_models:
            for interface, model in additional_models.items():
                models.set(interface, model)

        return models
