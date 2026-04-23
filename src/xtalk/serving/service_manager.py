# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Any

from fastapi import WebSocket

from ..log_utils import logger

from .service import Service, DefaultService
from ..pipelines import Pipeline


class ServiceManager:
    """Manage active Service instances for WebSocket sessions."""

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        service_config: dict[str, Any] | None = None,
        service_prototype: Service | None = None,
    ):
        """Initialize the service manager."""
        self.active_services: Dict[str, Service] = {}
        if pipeline is None and service_prototype is None:
            raise ValueError("Must provide either a pipeline or a service prototype.")
        self._service_prototype = service_prototype
        self._pipeline = pipeline
        self._service_config = service_config

    async def create_service(
        self,
        websocket: WebSocket,
    ) -> Service:
        """Create a new Service instance bound to the given WebSocket."""
        service = (
            DefaultService(
                pipeline=self._pipeline,
                service_config=self._service_config,
                _websocket=websocket,
            )
            if self._service_prototype is None
            else self._service_prototype.clone(new_websocket=websocket)
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
            f"Attempted to remove non-existent service - session_id: {session_id}"
        )
        return False

    def get_service(self, session_id: str) -> Optional[Service]:
        """Return the active Service for the session id, if any."""
        return self.active_services.get(session_id)

    def get_all_services(self) -> List[Service]:
        """Return all active Service instances."""
        return list(self.active_services.values())

    def get_service_count(self) -> int:
        """Return the number of active sessions."""
        return len(self.active_services)

    def get_session_ids(self) -> List[str]:
        """Return a list of active session ids."""
        return list(self.active_services.keys())

    async def shutdown_all_services(self) -> int:
        """Shut down every active Service and return the number closed."""
        success_count = 0
        failed_count = 0

        for session_id in list(self.active_services.keys()):
            try:
                if await self.remove_service(session_id):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to stop service - session_id: {session_id}, error: {e}"
                )
                failed_count += 1

        return success_count

    async def connect(
        self,
        websocket: WebSocket,
        already_accepted: bool = False,
    ):
        """Start a new service and connect to it for the given WebSocket connection.

        Args:
            websocket: FastAPI WebSocket
            already_accepted: skip websocket.accept() if caller already accepted.
        """
        service = None
        try:
            service = await self.create_service(websocket)
            await service.handle_message_loop(already_accepted=already_accepted)
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
            await websocket.close(code=1011, reason="Internal server error")
        finally:
            if service:
                await self.remove_service(service.session_id)
