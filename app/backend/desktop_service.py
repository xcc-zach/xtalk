"""Desktop XTalk runtime composition reusing the standard service stack."""

from __future__ import annotations

import uuid
from typing import Any

from xtalk import Xtalk
from xtalk.serving.service import DefaultService, Service


class DesktopXtalk(Xtalk):
    """Build XTalk sessions around the desktop service pipeline.

    The desktop sidecar deliberately reuses the generic XTalk serving
    pipeline (agent, ASR/TTS managers, persistence, and playback
    finalization) so conversation semantics stay identical to the sample
    applications. The stock output gateway is kept unchanged, so
    frontend-facing text follows the standard playback-driven updates.
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
    """Standard service stack used by desktop sessions.

    No gateway override is applied: every session uses the original
    ``xtalk.serving.modules.output_gateway.OutputGateway`` exactly like the
    generic XTalk pipeline.
    """
