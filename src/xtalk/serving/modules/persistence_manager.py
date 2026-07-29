# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any

from ...persistence import PersistenceStore
from ..event_bus import EventBus
from ..events import ASRResultFinal, ResponseFinish
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class PersistenceManager(Manager):
    """Persist final user and assistant messages for an authenticated session."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

        persistence_store = self.config.get("persistence_store")
        self._persistence = (
            persistence_store
            if isinstance(persistence_store, PersistenceStore)
            else None
        )

        user_id = self.config.get("user_id")
        self._user_id = str(user_id) if isinstance(user_id, str) and user_id else None

    @Manager.event_handler(ASRResultFinal, priority=-100)
    async def _persist_user_message(self, event: ASRResultFinal) -> None:
        if self._persistence is None or self._user_id is None:
            return
        try:
            self._persistence.append_message(
                user_id=self._user_id,
                session_id=self.session_id,
                role="user",
                content=event.text,
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist user message - session: %s, error: %s",
                self.session_id,
                exc,
            )

    @Manager.event_handler(ResponseFinish, priority=-100)
    async def _persist_assistant_message(
        self, event: ResponseFinish
    ) -> None:
        if self._persistence is None or self._user_id is None:
            return
        try:
            self._persistence.append_message(
                user_id=self._user_id,
                session_id=self.session_id,
                role="assistant",
                content=event.text,
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist assistant message - session: %s, error: %s",
                self.session_id,
                exc,
            )

    async def shutdown(self) -> None:
        return
