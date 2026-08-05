"""Desktop XTalk runtime composition reusing the standard service stack."""

from __future__ import annotations

import uuid
from typing import Any

from xtalk import Xtalk
from xtalk.serving.modules.output_gateway import OutputGateway
from xtalk.serving.service import DefaultService, Service

from .desktop_gateway import DesktopTextProjectionGateway


class DesktopXtalk(Xtalk):
    """Build XTalk sessions around the desktop service pipeline.

    The desktop sidecar deliberately reuses the generic XTalk serving
    pipeline (agent, ASR/TTS managers, persistence, and playback
    finalization) so conversation semantics stay identical to the sample
    applications. The only desktop-specific difference is the text-projection
    output gateway, which keeps the frontend text stream complete and
    monotonic even when TTS playback tracking restarts mid-turn.
    """

    def __init__(
        self,
        *,
        service_prototype: Service,
        max_sessions: int | None = None,
    ) -> None:
        """Initialize the desktop runtime with an optional stable identity.

        Parameters
        ----------
        service_prototype : Service
            Prototype service used to clone per-session service instances.
        max_sessions : int | None, optional
            Maximum number of concurrent sessions. If omitted, no session
            limit is enforced.
        """

        self._anonymous_user_id: str | None = None
        super().__init__(
            service_prototype=service_prototype,
            max_sessions=max_sessions,
        )

    def _login(self) -> dict[str, Any]:
        """Issue a login token for the desktop runtime identity.

        Desktop launches bind a stable anonymous identity through the app
        adapter; every login resolves to that identity so persisted sessions
        survive restarts without exposing user management.

        Returns
        -------
        dict[str, Any]
            Access token and the user record it authenticates.
        """

        user_id = self._anonymous_user_id or str(uuid.uuid4())
        user = (
            self._persistence.ensure_user(user_id)
            if self._persistence is not None
            else {"id": user_id}
        )
        return {
            "access_token": self._auth.issue_token(sub=user_id),
            "user": user,
        }

    @classmethod
    def _build_from_config_dict(cls, config: dict[str, Any]) -> DesktopXtalk:
        """Build a desktop runtime from an effective XTalk configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration after all public builder transforms have run.

        Returns
        -------
        DesktopXtalk
            Runtime whose sessions use the standard XTalk service pipeline.
        """

        models = cls.create_models_from_config(config_path_or_dict=config)
        service_config = config.get("service_config", {})
        if not isinstance(service_config, dict):
            raise ValueError("service_config must be an object")

        max_sessions: int | None = None
        if "max_connections" in config:
            max_sessions = int(config["max_connections"])
        service_prototype = DesktopService(
            models=models,
            service_config=service_config,
        )
        return cls(
            service_prototype=service_prototype,
            max_sessions=max_sessions,
        )


class DesktopService(DefaultService):
    """Default service stack with the desktop text-projection gateway."""

    def __init__(
        self,
        *,
        models: Any,
        service_config: dict[str, Any] | None = None,
        manager_classes: list[type] | None = None,
        _websocket: Any = None,
        _session_id: str | None = None,
        _event_overrides: dict[type, Any] | None = None,
    ) -> None:
        """Build the standard stack and swap in the projection gateway.

        Parameters
        ----------
        models : Models
            Model container prototype cloned for the session.
        service_config : dict[str, Any] | None, optional
            Session configuration shared with managers and gateways.
        manager_classes : list[type] | None, optional
            Manager classes to instantiate for live sessions.
        _websocket : WebSocket | None, optional
            Live WebSocket handle for session clones.
        _session_id : str | None, optional
            Identifier used when cloning a live session.
        _event_overrides : dict[type, Any] | None, optional
            Internal event subscription overrides copied into clones.
        """

        super().__init__(
            models=models,
            service_config=service_config,
            manager_classes=manager_classes,
            _websocket=_websocket,
            _session_id=_session_id,
            _event_overrides=_event_overrides,
        )
        if _websocket is not None and hasattr(self, "output_gateway"):
            self.output_gateway = _replace_output_gateway(self)


def _replace_output_gateway(service: DefaultService) -> OutputGateway:
    """Swap one live service's gateway for the desktop text projection.

    The base ``Service`` hardcodes the generic :class:`OutputGateway`. This
    helper unsubscribes its handlers from the session event bus and installs
    the desktop projection gateway with the same websocket and overrides.

    Parameters
    ----------
    service : DefaultService
        Live session service whose output gateway should be replaced.

    Returns
    -------
    OutputGateway
        The newly installed desktop projection gateway.
    """

    previous = service.output_gateway
    for base in reversed(previous.__class__.mro()):
        for method_name, meta_list in getattr(base, "__event_handlers_meta__", []):
            method = getattr(previous, method_name, None)
            if method is None:
                continue
            for meta in meta_list:
                service.event_bus.unsubscribe(meta["event_type"], method)

    return DesktopTextProjectionGateway(
        service.event_bus,
        service.session_id,
        previous.websocket,
        config=service.service_config,
        _event_overrides=service._event_overrides.get(OutputGateway),
    )
