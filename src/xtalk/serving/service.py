# -*- coding: utf-8 -*-
import logging
import uuid
from inspect import signature
from typing import Any, Callable, Coroutine, Type

from fastapi import WebSocket

from ..models import Agent, Models
from .event_bus import EventBus
from .events import Event, LLMAgentLoop
from .interfaces import EventListenerMixin, EventOverrides, Manager
from .modules.asr_manager import ASRManager
from .modules.captioner_manager import CaptionerManager
from .modules.direct_audio_manager import DirectAudioManager
from .modules.embeddings_manager import EmbeddingsManager
from .modules.enhancer_manager import EnhancerManager
from .modules.input_gateway import InputGateway
from .modules.latency_manager import LatencyManager
from .modules.llm_agent_context_manager import LLMAgentContextManager
from .modules.llm_agent_generation_manager import LLMAgentConsumptionManager
from .modules.multi_speaker_turn_context_manager import MultiSpeakerTurnContextManager
from .modules.output_gateway import OutputGateway
from .modules.persistence_manager import PersistenceManager
from .modules.recording_manager import RecordingManager
from .modules.retrieval_manager import RetrievalManager
from .modules.speaker_manager import SpeakerManager
from .modules.tts_manager import TTSManager
from .modules.tts_playback_manager import TTSPlaybackManager
from .modules.turn_detector_manager import TurnDetectorManager
from .modules.turn_taking_manager import TurnTakingManager
from .modules.vad_manager import VADManager

logger = logging.getLogger(__name__)


class Service:
    """Orchestrate a session-scoped model container and manager stack.

    Parameters
    ----------
    models : Models
        Model container prototype that will be cloned for the session.
    service_config : dict[str, Any] | None, optional
        Session configuration shared with managers and gateways.
    manager_classes : list[Type[Manager]] | None, optional
        Manager classes to instantiate for live sessions.
    _websocket : WebSocket | None, optional
        Internal WebSocket handle for live sessions. ``None`` means the instance
        acts as a prototype only.
    _event_overrides : dict[Type[EventListenerMixin], EventOverrides] | None, optional
        Internal event subscription overrides copied into cloned sessions.
    """

    def __init__(
        self,
        *,
        models: Models,
        service_config: dict[str, Any] | None = None,
        manager_classes: list[Type[Manager]] | None = None,
        _websocket: WebSocket | None = None,
        _session_id: str | None = None,
        _event_overrides: dict[Type[EventListenerMixin], EventOverrides] | None = None,
    ):
        self.session_id = _session_id or str(uuid.uuid4())
        models = models.clone()
        self.models = models
        # Per-session config (shared with managers/gateways)
        base_config: dict[str, Any] = dict(service_config) if service_config else {}
        # Default data directory inside current workspace
        if not base_config.get("data_dir"):
            base_config["data_dir"] = "data"
        self.service_config = base_config

        self.event_bus = EventBus(enable_history=False, max_history=0)

        self._manager_classes: list[Type[Manager]] = list(manager_classes or [])
        if self._should_enable_persistence_manager():
            if PersistenceManager not in self._manager_classes:
                self._manager_classes.append(PersistenceManager)
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
            self.event_bus,
            self.session_id,
            _websocket,
            config=self.service_config,
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
        event_type: Type[Event],
        method_name: str | None = None,
    ) -> None:
        """Disable an automatic event subscription for a listener class.

        Parameters
        ----------
        event_listener_cls : Type[EventListenerMixin]
            Listener class whose subscription should be disabled.
        event_type : Type[Event]
            Event type to unsubscribe.
        method_name : str | None, optional
            Specific method name to disable. If omitted, every handler for the
            event is disabled for the listener class.
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
        event_type: Type[Event],
        method_or_handler: (
            str
            | Callable[[EventListenerMixin, Event], Any]
            | Callable[[Event], Any]
            | Callable[[EventListenerMixin, Event], Coroutine[Any, Any, Any]]
            | Callable[[Event], Coroutine[Any, Any, Any]]
        ),
        priority: int = 0,
        enabled_if: Callable[[EventListenerMixin], bool] | None = None,
    ) -> None:
        """Register an additional event subscription override.

        Parameters
        ----------
        event_listener_cls : Type[EventListenerMixin]
            Listener class that should receive the event.
        event_type : Type[Event]
            Event type to subscribe to.
        method_or_handler : str | Callable
            Method name on the listener instance or an external sync/async
            handler accepting ``event`` or ``(listener, event)``.
        priority : int, optional
            Subscription priority. Higher values run first.
        enabled_if : Callable[[EventListenerMixin], bool] | None, optional
            Predicate used to decide whether the subscription should be
            installed for a concrete listener instance.
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

        # Determine whether models are needed
        if "models" in signature(manager_cls.__init__).parameters:
            kwargs: dict[str, Any] = dict(
                event_bus=self.event_bus,
                session_id=self.session_id,
                models=self.models,
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
        """Register a manager class on the service.

        Parameters
        ----------
        manager_cls : Type[Manager]
            Manager class to add to the service prototype or live session.
        """
        self._manager_classes.append(manager_cls)
        # Instantiate immediately when running in a live session
        if hasattr(self, "input_gateway"):
            self._managers.append(self._instantiate_manager(manager_cls))

    def unregister_manager(self, manager_cls: Type[Manager]):
        """Remove a manager class from the service.

        Parameters
        ----------
        manager_cls : Type[Manager]
            Manager class to remove.
        """
        self._manager_classes.remove(manager_cls)
        # Remove concrete manager if already active and unsubscribe its handlers
        if hasattr(self, "input_gateway"):
            for m in self._managers:
                if isinstance(m, manager_cls):
                    self._unsubscribe_manager_handlers(m)
            self._managers = [
                m for m in self._managers if not isinstance(m, manager_cls)
            ]

    def _unsubscribe_manager_handlers(self, manager: Manager) -> None:
        """Unsubscribe all event handlers registered by a manager instance."""
        for base in reversed(manager.__class__.mro()):
            handlers_meta = getattr(base, "__event_handlers_meta__", [])
            for method_name, meta_list in handlers_meta:
                method = getattr(manager, method_name, None)
                if method is None:
                    continue
                for meta in meta_list:
                    ev_type = meta["event_type"]
                    self.event_bus.unsubscribe(ev_type, method)

    async def handle_message_loop(self, already_accepted: bool = False) -> None:
        """Run the full WebSocket message loop for a live session.

        Parameters
        ----------
        already_accepted : bool, optional
            Whether the WebSocket has already been accepted by an upstream
            limiter or caller.

        Raises
        ------
        RuntimeError
            Raised if the service instance is still a prototype without runtime
            gateways.
        """
        if not hasattr(self, "input_gateway") or not hasattr(self, "output_gateway"):
            raise RuntimeError(
                "This Service instance is a prototype and cannot handle messages."
            )
        await self.input_gateway.handle_connection(already_accepted=already_accepted)
        await self.event_bus.publish(
            LLMAgentLoop(session_id=self.session_id),
            wait_for_completion=True,
        )
        await self.input_gateway.handle_message_loop()

    def restore_conversation(self, *, messages: list[dict[str, Any]]) -> None:
        """Restore persisted conversation history into the session agent."""
        agent = self.models.get(Agent)
        if agent is None:
            return
        agent.restore_history(messages)

    async def send_session_attached(self) -> None:
        """Notify the client that the session is attached."""
        if not hasattr(self, "output_gateway"):
            raise RuntimeError(
                "This Service instance is a prototype and cannot send session events."
            )
        await self.output_gateway.send_session_attached()

    async def stop(self) -> None:
        """Stop the service and shut down all managers."""
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

    def clone(
        self,
        new_websocket: WebSocket,
        *,
        session_id: str | None = None,
        service_config_overrides: dict[str, Any] | None = None,
    ) -> "Service":
        """Clone the service prototype for a new WebSocket session.

        Parameters
        ----------
        new_websocket : WebSocket
            WebSocket assigned to the new live session.

        Returns
        -------
        Service
            Cloned service instance of the same concrete type.
        """
        cloned_service_config = dict(self.service_config)
        if service_config_overrides:
            cloned_service_config.update(service_config_overrides)
        new_service = type(self)(
            models=self.models,
            service_config=cloned_service_config,
            manager_classes=self._manager_classes,
            _websocket=new_websocket,
            _session_id=session_id,
            _event_overrides=self._event_overrides,
        )
        return new_service

    def _should_enable_persistence_manager(self) -> bool:
        persistence_store = self.service_config.get("persistence_store")
        user_id = self.service_config.get("user_id")
        return (
            persistence_store is not None and isinstance(user_id, str) and bool(user_id)
        )

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
    """Convenience ``Service`` with the standard Xtalk manager stack.

    Notes
    -----
    Sample applications usually instantiate ``DefaultService`` directly and then
    register or override managers for custom behavior.
    """

    MANAGER_CLASSES: list[Type[Manager]] = [
        ASRManager,
        MultiSpeakerTurnContextManager,
        LLMAgentContextManager,
        LLMAgentConsumptionManager,
        DirectAudioManager,
        TTSManager,
        TTSPlaybackManager,
        CaptionerManager,
        RetrievalManager,
        TurnTakingManager,
        LatencyManager,
        VADManager,
        EnhancerManager,
        SpeakerManager,
        EmbeddingsManager,
        RecordingManager,
        TurnDetectorManager,
    ]

    def __init__(
        self,
        *,
        models: Models,
        service_config: dict[str, Any] | None = None,
        manager_classes: list[Type[Manager]] | None = None,
        _websocket: WebSocket | None = None,
        _session_id: str | None = None,
        _event_overrides: dict[Type[EventListenerMixin], EventOverrides] | None = None,
    ):
        super().__init__(
            models=models,
            service_config=service_config,
            manager_classes=(
                self.MANAGER_CLASSES if manager_classes is None else manager_classes
            ),
            _websocket=_websocket,
            _session_id=_session_id,
            _event_overrides=_event_overrides,
        )
