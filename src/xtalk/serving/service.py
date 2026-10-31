import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from inspect import signature
from typing import Any, ClassVar

from ..models import Agent, EndToEnd, Models
from ..realtime.protocol import InputHandler, OutputHandler
from . import events as runtime_events
from .event_bus import EventBus, EventDispatchMode
from .events import Event, LLMAgentLoop
from .interfaces import Connection as WebSocket
from .interfaces import EventListenerMixin, EventOverrides, Manager
from .modules.asr_manager import ASRManager
from .modules.captioner_manager import CaptionerManager
from .modules.end_to_end_manager import EndToEndManager, create_stream_profile
from .modules.enhancer_manager import EnhancerManager
from .modules.input_gateway import InputGateway
from .modules.latency_manager import LatencyManager
from .modules.llm_agent_context_manager import LLMAgentContextManager
from .modules.llm_agent_generation_manager import LLMAgentConsumptionManager
from .modules.multi_speaker_turn_context_manager import MultiSpeakerTurnContextManager
from .modules.output_gateway import OutputGateway
from .modules.recording_manager import RecordingManager
from .modules.retrieval_manager import RetrievalManager
from .modules.speaker_manager import SpeakerManager
from .modules.tts_manager import TTSManager
from .modules.tts_playback_manager import TTSPlaybackManager
from .modules.tts_response_coordinator import TTSResponseCoordinator
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
        manager_classes: list[type[Manager]] | None = None,
        _websocket: WebSocket | None = None,
        _session_id: str | None = None,
        _event_overrides: dict[type[EventListenerMixin], EventOverrides] | None = None,
    ):
        self.session_id = _session_id or str(uuid.uuid4())
        models = models.clone()
        self.models = models
        # Per-session config (shared with managers/gateways)
        base_config: dict[str, Any] = dict(service_config) if service_config else {}
        base_config.setdefault("output_modalities", ["audio"])
        live = models.get(EndToEnd)
        if live is not None:
            base_config["_end_to_end_profile"] = create_stream_profile(live, base_config)
            base_config["input_sample_rate"] = live.audio_profile.sample_rate
        self.service_config = base_config

        self.event_bus = EventBus(enable_history=False, max_history=0)

        self._input_handlers = {}
        self._output_handlers = {}
        self._manager_classes: list[type[Manager]] = list(manager_classes or [])
        self._managers: list[Manager] = []
        self._event_overrides: dict[type[EventListenerMixin], EventOverrides] = (
            self._clone_event_overrides(_event_overrides)
        )

        if _event_overrides is None:
            self._register_builtin_handlers()
            if live is not None:
                self._register_stream_handlers()

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
        self.input_gateway._service = self
        self.input_gateway._handlers = dict(self._input_handlers)
        self.output_gateway = OutputGateway(
            self.event_bus,
            self.session_id,
            _websocket,
            config=self.service_config,
            models=self.models,
            _event_overrides=self._event_overrides.get(OutputGateway),
        )

        self.output_gateway._register_custom_outputs(self._effective_output_handlers())

    def _register_stream_handlers(self) -> None:
        """Register continuous controls once on the service prototype."""
        for kind in (
            "xtalk.audio_stream.ready",
            "xtalk.audio_stream.progress",
            "xtalk.audio_stream.input_muted",
            "xtalk.audio_stream.close",
        ):
            self.register_input_handler(
                type=kind, handler=InputGateway._handle_stream_control
            )
        for kind in (
            "input_audio_buffer.commit",
            "input_audio_buffer.clear",
            "input_audio_buffer.speech_start",
            "conversation.item.create",
            "response.cancel",
            "response.output_audio.playback.chunk_played",
            "response.output_audio.playback.finished",
            "response.output_audio.playback.stopped",
        ):
            self.register_input_handler(
                type=kind, handler=InputGateway._unsupported_live_input
            )
        for event, method in (
            (runtime_events.EndToEndSignal, "_stream_signal"),
            (runtime_events.EndToEndAudioChunkReady, "_stream_audio"),
            (runtime_events.EndToEndTranscriptDelta, "_stream_transcript"),
        ):
            self.subscribe_event(
                event_listener_cls=OutputGateway,
                event_type=event,
                method_or_handler=method,
                priority=5,
            )

    def _register_builtin_handlers(self) -> None:
        """Assemble protocol handlers through the public registration APIs."""
        for kind, handler in (
            ("session.update", InputGateway._update_session),
            ("input_audio_buffer.append", InputGateway._handle_audio_append),
            ("input_audio_buffer.commit", InputGateway._handle_audio_commit),
            ("input_audio_buffer.clear", InputGateway._handle_audio_clear),
            ("input_audio_buffer.speech_start", InputGateway._handle_speech_start),
            ("conversation.item.create", InputGateway._handle_item_create),
            ("response.cancel", InputGateway._handle_response_cancel),
            (
                "response.output_audio.playback.chunk_played",
                InputGateway._handle_playback_ack,
            ),
            (
                "response.output_audio.playback.finished",
                InputGateway._handle_playback_ack,
            ),
            (
                "response.output_audio.playback.stopped",
                InputGateway._handle_playback_ack,
            ),
        ):
            self.register_input_handler(type=kind, handler=handler)
        for source, kind, handler in (
            (
                runtime_events.ResponseUpdate,
                "response.output_audio.playback.updated",
                OutputGateway._map_playback,
            ),
            (
                runtime_events.ResponseFinish,
                "response.output_audio.playback.done",
                OutputGateway._map_playback,
            ),
            (
                runtime_events.SpeakerRecognized,
                "conversation.speaker.updated",
                OutputGateway._map_speaker,
            ),
            (
                runtime_events.CaptionUpdated,
                "conversation.caption.updated",
                OutputGateway._map_caption,
            ),
            (
                runtime_events.RetrievalUpdated,
                "conversation.retrieval.updated",
                OutputGateway._map_retrieval,
            ),
            (
                runtime_events.LatencyMetricsUpdated,
                "session.latency.updated",
                OutputGateway._map_latency,
            ),
            (
                runtime_events.ToolCallOccurred,
                "response.tool.called",
                OutputGateway._map_tool,
            ),
        ):
            self.register_output_handler(
                source_event=source, type=kind, handler=handler
            )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TurnASREndRequested,
            method_or_handler="_on_input_commit",
            priority=100,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.LLMAgentResponseUpdate,
            method_or_handler="_on_generated_text",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.LLMAgentResponseFinish,
            method_or_handler="_on_generated_done",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSTextDeliveryFinished,
            method_or_handler="_on_sentence",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.ASRResultPartial,
            method_or_handler="_send_gated_update_asr_signal",
            priority=40,
            enabled_if=lambda gateway: gateway._speaker_history_gate_enabled,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.ASRResultPartial,
            method_or_handler="_send_update_asr_signal",
            priority=5,
            enabled_if=lambda gateway: not gateway._speaker_history_gate_enabled,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.ASRFailed,
            method_or_handler="_send_asr_failure",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.ASRResultFinal,
            method_or_handler="_send_finish_asr_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSStarted,
            method_or_handler="_on_start_tts",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSStopped,
            method_or_handler="_send_stop_tts_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.ErrorOccurred,
            method_or_handler="_send_error_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSPaused,
            method_or_handler="_send_pause_tts_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSResumed,
            method_or_handler="_send_resume_tts_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSFinished,
            method_or_handler="_send_tts_finished_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSChunkReady,
            method_or_handler="_send_tts_chunk_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.FullAudioFrameReady,
            method_or_handler="_send_full_audio_frame_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSVoiceChange,
            method_or_handler="_send_voice_changed_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.VADSpeechStart,
            method_or_handler="_send_vad_start_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.VADSpeechEnd,
            method_or_handler="_send_vad_end_signal",
            priority=5,
        )
        self.subscribe_event(
            event_listener_cls=OutputGateway,
            event_type=runtime_events.TTSEmotionChange,
            method_or_handler="_send_emotion_changed_signal",
            priority=5,
        )

    def _effective_output_handlers(self):
        overrides = self._event_overrides.get(OutputGateway)
        return {
            key: handler
            for key, handler in self._output_handlers.items()
            if overrides is None or not overrides.should_skip("*", key[0])
        }

    def register_input_handler(self, *, type: str, handler: InputHandler) -> None:
        """Register or replace an external input handler.

        Parameters
        ----------
        type : str
            External event type.
        handler : callable
            Async handler accepting the current gateway and complete message.
        """
        self._input_handlers[type] = handler
        if hasattr(self, "input_gateway"):
            self.input_gateway._handlers[type] = handler

    def register_output_handler(
        self, *, source_event: type[Event], type: str, handler: OutputHandler
    ) -> None:
        """Register or replace an internal-event output mapping.

        Parameters
        ----------
        source_event : type[Event]
            Internal source event.
        type : str
            Outbound event type.
        handler : callable
            Synchronous or async payload builder, optionally returning None.
        """
        self._output_handlers[(source_event, type)] = handler
        if hasattr(self, "output_gateway"):
            self.output_gateway._register_custom_outputs(
                {(source_event, type): handler}
            )

    # ------------------------ override utilities ------------------------

    def _get_or_create_overrides(
        self, event_listner_cls: type[Manager]
    ) -> EventOverrides:
        overrides = self._event_overrides.get(event_listner_cls)
        if overrides is None:
            overrides = EventOverrides()
            self._event_overrides[event_listner_cls] = overrides
        return overrides

    def unsubscribe_event(
        self,
        *,
        event_listener_cls: type[EventListenerMixin],
        event_type: type[Event],
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
        event_listener_cls: type[EventListenerMixin],
        event_type: type[Event],
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

    def _instantiate_manager(self, manager_cls: type[Manager]) -> Manager:
        # Locate overrides for this manager class
        overrides = self._event_overrides.get(manager_cls)

        # Determine whether models are needed
        if "models" in signature(manager_cls.__init__).parameters:
            kwargs: dict[str, Any] = {
                "event_bus": self.event_bus,
                "session_id": self.session_id,
                "models": self.models,
                "config": self.service_config,
            }
        else:
            kwargs = {
                "event_bus": self.event_bus,
                "session_id": self.session_id,
                "config": self.service_config,
            }

        # Pass overrides via hidden parameter to EventListenerMeta.__call__
        if overrides is not None:
            kwargs["_event_overrides"] = overrides

        manager = manager_cls(**kwargs)
        return manager

    def register_manager(self, manager_cls: type[Manager]):
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

    def unregister_manager(self, manager_cls: type[Manager]):
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
            runtime_events.ServiceRuntimeStarted(session_id=self.session_id, mode="end_to_end")
            if self.models.get(EndToEnd) is not None else LLMAgentLoop(session_id=self.session_id),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        await self.input_gateway.handle_message_loop()

    async def stop(self) -> None:
        """Stop the service and shut down all managers."""
        for manager in self._managers:
            results = await asyncio.gather(manager.shutdown(), return_exceptions=True)
            if isinstance(results[0], BaseException):
                logger.error(
                    "Manager shutdown failed - session: %s, error: %s",
                    self.session_id,
                    results[0],
                )
        await self.event_bus.shutdown()

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
        new_service._input_handlers = dict(self._input_handlers)
        new_service._output_handlers = dict(self._output_handlers)
        new_service.input_gateway._handlers = dict(self._input_handlers)
        new_service.output_gateway._register_custom_outputs(
            new_service._effective_output_handlers()
        )
        return new_service

    @staticmethod
    def _clone_event_overrides(
        source: dict[type[EventListenerMixin], EventOverrides] | None,
    ) -> dict[type[EventListenerMixin], EventOverrides]:
        if source is None:
            return {}
        cloned: dict[type[EventListenerMixin], EventOverrides] = {}
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

    MANAGER_CLASSES: ClassVar[list[type[Manager]]] = [
        ASRManager,
        MultiSpeakerTurnContextManager,
        LLMAgentContextManager,
        LLMAgentConsumptionManager,
        TTSManager,
        TTSResponseCoordinator,
        TTSPlaybackManager,
        CaptionerManager,
        RetrievalManager,
        TurnTakingManager,
        LatencyManager,
        VADManager,
        EnhancerManager,
        SpeakerManager,
        RecordingManager,
        TurnDetectorManager,
    ]

    def __init__(
        self,
        *,
        models: Models,
        service_config: dict[str, Any] | None = None,
        manager_classes: list[type[Manager]] | None = None,
        _websocket: WebSocket | None = None,
        _session_id: str | None = None,
        _event_overrides: dict[type[EventListenerMixin], EventOverrides] | None = None,
    ):
        if models.get(EndToEnd) is not None and manager_classes is None:
            from ..models import (
                ASR,
                TTS,
                VAD,
                SpeakerEncoder,
                SpeechEnhancer,
                TurnDetector,
            )
            if any(models.get(kind) is not None for kind in (Agent, ASR, TTS, VAD, SpeechEnhancer, SpeakerEncoder, TurnDetector)):
                raise ValueError("EndToEnd default graph cannot mix cascade models")
        super().__init__(
            models=models,
            service_config=service_config,
            manager_classes=(
                ([EndToEndManager] if models.get(EndToEnd) is not None else self.MANAGER_CLASSES)
                if manager_classes is None else manager_classes
            ),
            _websocket=_websocket,
            _session_id=_session_id,
            _event_overrides=_event_overrides,
        )
