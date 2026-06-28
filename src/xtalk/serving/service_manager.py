# -*- coding: utf-8 -*-
import json
from typing import Dict, Optional, Any

from fastapi import WebSocket

from ..log_utils import logger
from ..models import Models
from ..persistence import PersistenceStore

from .service import Service, DefaultService


class ServiceManager:
    """Manage active Service instances for WebSocket sessions."""

    def __init__(
        self,
        models: Models | None = None,
        service_config: dict[str, Any] | None = None,
        service_prototype: Service | None = None,
        persistence_store: PersistenceStore | None = None,
    ):
        self.active_services: Dict[str, Service] = {}
        if models is None and service_prototype is None:
            raise ValueError("Must provide either models or a service prototype.")
        self._service_prototype = service_prototype
        self._models = models
        self._service_config = service_config
        self._persistence = persistence_store

    async def create_service(
        self,
        websocket: WebSocket,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Service:
        """Create a new Service instance bound to the given WebSocket."""
        service_config_overrides: dict[str, Any] = {}
        if self._persistence is not None and user_id is not None:
            service_config_overrides["persistence_store"] = self._persistence
            service_config_overrides["user_id"] = user_id

        service = (
            DefaultService(
                models=self._models,
                service_config=self._build_service_config(service_config_overrides),
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

    def get_service(self, session_id: str) -> Optional[Service]:
        return self.active_services.get(session_id)

    def get_service_count(self) -> int:
        return len(self.active_services)

    async def connect(
        self,
        websocket: WebSocket,
        already_accepted: bool = False,
        user_id: str | None = None,
    ) -> None:
        """Start a new service and connect to it for the given WebSocket."""
        if user_id is None:
            await self._connect_legacy(websocket, already_accepted=already_accepted)
            return

        if self._persistence is None:
            await self._connect_without_persistence(
                websocket, already_accepted=already_accepted
            )
            return

        service: Service | None = None
        try:
            if not already_accepted:
                await websocket.accept()
            attach_data = await self._receive_attach_request(websocket)
            session = self._resolve_session(user_id, attach_data)
            session_id = str(session["session_id"])
            if session_id in self.active_services:
                await websocket.close(code=1013, reason="Session already connected")
                return

            service = await self.create_service(
                websocket,
                session_id=session_id,
                user_id=user_id,
            )
            service.restore_conversation(
                messages=self._persistence.list_messages(user_id, session_id),
            )
            await service.send_session_attached()
            await service.handle_message_loop(already_accepted=True)
        except Exception as e:
            logger.error("Error handling authenticated WebSocket connection: %s", e)
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except Exception:
                pass
        finally:
            if service:
                await self.remove_service(service.session_id)

    async def _connect_without_persistence(
        self, websocket: WebSocket, *, already_accepted: bool = False
    ) -> None:
        """Handle authenticated websocket sessions without persistence state."""
        service: Service | None = None
        try:
            if not already_accepted:
                await websocket.accept()
            await self._receive_attach_request(websocket)
            service = await self.create_service(websocket)
            await service.send_session_attached()
            await service.handle_message_loop(already_accepted=True)
        except Exception as e:
            logger.error(
                "Error handling non-persistent authenticated WebSocket connection: %s",
                e,
            )
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except Exception:
                pass
        finally:
            if service:
                await self.remove_service(service.session_id)

    async def _connect_legacy(
        self, websocket: WebSocket, *, already_accepted: bool = False
    ) -> None:
        service: Service | None = None
        try:
            service = await self.create_service(websocket)
            await service.handle_message_loop(already_accepted=already_accepted)
        except Exception as e:
            logger.error("Error handling WebSocket connection: %s", e)
            await websocket.close(code=1011, reason="Internal server error")
        finally:
            if service:
                await self.remove_service(service.session_id)

    async def _receive_attach_request(self, websocket: WebSocket) -> dict[str, Any]:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            raise RuntimeError("WebSocket disconnected before attach_session")

        payload = message.get("text")
        if not isinstance(payload, str):
            raise ValueError("First WebSocket message must be attach_session JSON")

        data = json.loads(payload)
        if data.get("action") != "attach_session":
            raise ValueError("First WebSocket message must be attach_session")
        return data

    def _resolve_session(
        self, user_id: str, attach_data: dict[str, Any]
    ) -> dict[str, Any]:
        requested_session_id = self._resolve_requested_session_id(attach_data)
        if requested_session_id is None:
            return self._persistence.create_session(user_id)

        return self._persistence.get_or_create_session(user_id, requested_session_id)

    @staticmethod
    def _resolve_requested_session_id(attach_data: dict[str, Any]) -> str | None:
        """Normalize the optional session identifier from attach_session."""
        requested_session_id = attach_data.get("session_id")
        if isinstance(requested_session_id, str):
            return requested_session_id.strip() or None
        return None

    def _build_service_config(
        self, service_config_overrides: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self._service_config is None and not service_config_overrides:
            return None
        service_config = dict(self._service_config or {})
        service_config.update(service_config_overrides)
        return service_config
