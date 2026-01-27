# -*- coding: utf-8 -*-
import uuid
from inspect import signature
from fastapi import WebSocket
from typing import Any, Type, Callable, Coroutine

from ..log_utils import logger

from .event_bus import EventBus
from .interfaces import Manager
from .modules.asr_manager import ASRManager
from .modules.tts_manager import TTSManager
from .modules.output_gateway import OutputGateway
from .modules.input_gateway import InputGateway
from .modules.llm_agent_manager import LLMAgentManager
from .modules.captioner_manager import CaptionerManager
from .modules.retrieval_manager import RetrievalManager
from .modules.turn_taking_manager import TurnTakingManager
from .modules.thought_manager import ThoughtManager
from .modules.latency_manager import LatencyManager
from .modules.vad_manager import VADManager
from .modules.enhancer_manager import EnhancerManager
from .modules.speaker_manager import SpeakerManager
from .modules.embeddings_manager import EmbeddingsManager
from .modules.recording_manager import RecordingManager
from .events import BaseEvent
from ..pipelines import Pipeline
from .interfaces import EventListenerMixin, EventOverrides


class Service:
    """Session-scoped orchestrator wiring pipeline managers to a WebSocket connection."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        service_config: dict[str, Any] | None = None,
        manager_classes: list[Type[Manager]] | None = None,
        _websocket: WebSocket | None = None,
        _event_overrides: dict[Type[EventListenerMixin], EventOverrides] | None = None,
    ):
        self.session_id = str(uuid.uuid4())
        pipeline = pipeline.clone()
        self.pipeline = pipeline  # Keep pipeline reference for later model switches
        # Per-session config (shared with managers/gateways)
        base_config: dict[str, Any] = dict(service_config) if service_config else {}
        # Default data directory inside current workspace
        if not base_config.get("data_dir"):
            base_config["data_dir"] = "data"
        self.service_config = base_config

        self.event_bus = EventBus(enable_history=False, max_history=0)

        self._manager_classes: list[Type[Manager]] = list(manager_classes or [])
        self._managers: list[Manager] = []
        self._event_overrides: dict[Type[EventListenerMixin], EventOverrides] = (
            self._clone_event_overrides(_event_overrides)
        )

        # When _websocket is None this is a prototype instance with no runtime logic
        if _websocket is None:
            return
        for manager_cls in self._manager_classes:
            manager = self._instantiate_manager(manager_cls)
            self._managers.append(manager)
        self.input_gateway = InputGateway(
            self.event_bus, self.session_id, _websocket, config=self.service_config
        )
        self.output_gateway = OutputGateway(
            self.event_bus,
            self.session_id,
            _websocket,
            config=self.service_config,
            _event_overrides=self._event_overrides.get(OutputGateway),
        )

    # ------------------------ override utilities ------------------------

    def _get_or_create_overrides(
        self, event_listner_cls: Type[Manager]
    ) -> EventOverrides:
        overrides = self._event_overrides.get(event_listner_cls)
        if overrides is None:
            overrides = EventOverrides()
            self._event_overrides[event_listner_cls] = overrides
        return overrides

    def unsubscribe_event(
        self,
        *,
        event_listener_cls: Type[EventListenerMixin],
        event_type: Type[BaseEvent],
        method_name: str | None = None,
    ) -> None:
        """
        Remove automatic event subscription for an EventListener.

        - method_name None: disable every handler for that event (including ones
          added via @event_handler or overrides)
        - method_name str: disable only that specific handler
        """
        overrides = self._get_or_create_overrides(event_listener_cls)

        if method_name is None:
            overrides.disable.append(("*", event_type))
        else:
            overrides.disable.append((method_name, event_type))

    def subscribe_event(
        self,
        *,
        event_listener_cls: Type[EventListenerMixin],
        event_type: Type[BaseEvent],
        method_or_handler: (
            str
            | Callable[[EventListenerMixin, BaseEvent], Any]
            | Callable[[BaseEvent], Any]
            | Callable[[EventListenerMixin, BaseEvent], Coroutine[Any, Any, Any]]
            | Callable[[BaseEvent], Coroutine[Any, Any, Any]]
        ),
        priority: int = 0,
        enabled_if: Callable[[EventListenerMixin], bool] | None = None,
    ) -> None:
        """
        Add extra event subscriptions for an EventListener.

        - method_or_handler str: treat as method name; resolved via getattr on the
          listener instance.
        - method_or_handler callable: external handler supporting signatures
          h(event) / async h(event) or h(listener, event) / async variant.
        """
        overrides = self._get_or_create_overrides(event_listener_cls)

        if isinstance(method_or_handler, str):
            overrides.extra.append(
                {
                    "method_name": method_or_handler,
                    "event_type": event_type,
                    "priority": priority,
                    "enabled_if": enabled_if,
                }
            )
        elif callable(method_or_handler):
            overrides.extra.append(
                {
                    "handler": method_or_handler,
                    "event_type": event_type,
                    "priority": priority,
                    "enabled_if": enabled_if,
                }
            )
        else:
            raise TypeError(
                "method_or_handler must be a method name (str) or a callable"
            )

    # ---------------------------------------------------------------

    def _instantiate_manager(self, manager_cls: Type[Manager]) -> Manager:
        # Locate overrides for this manager class
        overrides = self._event_overrides.get(manager_cls)

        # Determine whether pipeline is needed
        if "pipeline" in signature(manager_cls.__init__).parameters:
            kwargs: dict[str, Any] = dict(
                event_bus=self.event_bus,
                session_id=self.session_id,
                pipeline=self.pipeline,
                config=self.service_config,
            )
        else:
            kwargs = dict(
                event_bus=self.event_bus,
                session_id=self.session_id,
                config=self.service_config,
            )

        # Pass overrides via hidden parameter to EventListenerMeta.__call__
        if overrides is not None:
            kwargs["_event_overrides"] = overrides

        manager = manager_cls(**kwargs)
        return manager

    def register_manager(self, manager_cls: Type[Manager]):
        self._manager_classes.append(manager_cls)
        # Instantiate immediately when running in a live session
        if hasattr(self, "input_gateway"):
            self._managers.append(self._instantiate_manager(manager_cls))

    def unregister_manager(self, manager_cls: Type[Manager]):
        self._manager_classes.remove(manager_cls)
        # Remove concrete manager if already active
        if hasattr(self, "input_gateway"):
            self._managers = [
                m for m in self._managers if not isinstance(m, manager_cls)
            ]

    async def handle_message_loop(self, already_accepted: bool = False) -> None:
        """Process the WebSocket message loop."""
        if not hasattr(self, "input_gateway") or not hasattr(self, "output_gateway"):
            raise RuntimeError(
                "This Service instance is a prototype and cannot handle messages."
            )
        await self.input_gateway.handle_connection(already_accepted=already_accepted)
        # Send session_id to frontend immediately for tracking uploads, etc.
        await self.output_gateway.send_session_info()
        await self.input_gateway.handle_message_loop()

    async def stop(self) -> None:
        """Stop the service and shut down managers."""
        try:
            for manager in self._managers:
                await manager.shutdown()
            await self.event_bus.shutdown()

        except Exception as e:
            logger.error(
                "Error while stopping service - session: %s, error: %s",
                self.session_id,
                e,
            )

    def clone(self, new_websocket: WebSocket) -> "Service":
        new_service = type(self)(
            pipeline=self.pipeline,
            service_config=self.service_config,
            manager_classes=self._manager_classes,
            _websocket=new_websocket,
            _event_overrides=self._event_overrides,
        )
        return new_service

    @staticmethod
    def _clone_event_overrides(
        source: dict[Type[EventListenerMixin], EventOverrides] | None,
    ) -> dict[Type[EventListenerMixin], EventOverrides]:
        if source is None:
            return {}
        cloned: dict[Type[EventListenerMixin], EventOverrides] = {}
        for listener_cls, overrides in source.items():
            cloned[listener_cls] = EventOverrides(
                disable=[
                    (method_name, event_type)
                    for method_name, event_type in overrides.disable
                ],
                extra=[dict(item) for item in overrides.extra],
            )
        return cloned


class DefaultService(Service):
    """Convenience Service pre-configured with the default manager stack."""

    MANAGER_CLASSES: list[Type[Manager]] = [
        ASRManager,
        LLMAgentManager,
        TTSManager,
        CaptionerManager,
        ThoughtManager,
        RetrievalManager,
        TurnTakingManager,
        LatencyManager,
        VADManager,
        EnhancerManager,
        SpeakerManager,
        EmbeddingsManager,
        RecordingManager,
    ]

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        service_config: dict[str, Any] | None = None,
        manager_classes: list[Type[Manager]] | None = None,
        _websocket: WebSocket | None = None,
        _event_overrides: dict[Type[EventListenerMixin], EventOverrides] | None = None,
    ):
        super().__init__(
            pipeline=pipeline,
            service_config=service_config,
            manager_classes=(
                self.MANAGER_CLASSES if manager_classes is None else manager_classes
            ),
            _websocket=_websocket,
            _event_overrides=_event_overrides,
        )
