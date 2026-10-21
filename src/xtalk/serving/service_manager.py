import logging
from typing import Any

from ..models import Models
from .interfaces import Connection as WebSocket
from .service import DefaultService, Service

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manage active Service instances for WebSocket sessions."""

    def __init__(
        self,
        models: Models | None = None,
        service_config: dict[str, Any] | None = None,
        service_prototype: Service | None = None,
    ):
        self.active_services: dict[str, Service] = {}
        if models is None and service_prototype is None:
            raise ValueError("Must provide either models or a service prototype.")
        self._service_prototype = service_prototype
        self._models = models
        self._service_config = service_config

    async def create_service(
        self,
        websocket: WebSocket,
        *,
        session_id: str | None = None,
    ) -> Service:
        """Create a new Service instance bound to the given WebSocket."""
        service_config_overrides: dict[str, Any] = {}
        service = (
            DefaultService(
                models=self._models,
                service_config=dict(
                    self._service_config or {}, **service_config_overrides
                ),
                _websocket=websocket,
                _session_id=session_id,
            )
            if self._service_prototype is None
            else self._service_prototype.clone(
                new_websocket=websocket,
                session_id=session_id,
                service_config_overrides=service_config_overrides,
            )
        )
        self.active_services[service.session_id] = service
        return service

    async def remove_service(self, session_id: str) -> bool:
        """Remove and shut down the Service with the given session id."""
        if session_id in self.active_services:
            service = self.active_services[session_id]
            await service.stop()
            del self.active_services[session_id]
            return True

        logger.warning(
            "Attempted to remove non-existent service - session_id: %s", session_id
        )
        return False

    def get_service(self, session_id: str) -> Service | None:
        return self.active_services.get(session_id)

    def get_service_count(self) -> int:
        return len(self.active_services)

    async def _connect(self, websocket, already_accepted=False):
        """Run a temporary conversation and always release its resources.

        Parameters
        ----------
        websocket : Connection
            Framework-independent transport.
        already_accepted : bool
            Whether the caller accepted the WebSocket handshake.
        """
        service = await self.create_service(websocket)
        try:
            if not already_accepted:
                await websocket.accept()
            await service.output_gateway._send_session_created()
            await service.handle_message_loop(already_accepted=True)
        finally:
            await self.remove_service(service.session_id)
