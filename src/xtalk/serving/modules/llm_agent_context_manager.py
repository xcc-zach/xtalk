# -*- coding: utf-8 -*-
"""Bridge serving-layer context events into the session LLM agent."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

from ...models import Agent, Models, SpeakerDiarization
from ...models.agents import AgentContext
from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    ASRResultFinal,
    ASRResultPartial,
    CaptionUpdated,
    ConsumeLLMAgentGenerationRequested,
    EmbeddingStatusUpdated,
    Event,
    LLMAgentLoop,
    MultiSpeakerTurnReady,
    ResponseFinish,
    ResponseUpdate,
    SpeakerRecognized,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)

_BASE_EVENT_FIELD_NAMES = frozenset(field.name for field in fields(Event))


class LLMAgentContextManager(Manager):
    """Forward session context events into the configured LLM agent.

    Parameters
    ----------
    event_bus : EventBus
        Shared event bus for the current session.
    session_id : str
        Current session identifier.
    models : Models
        Session model container that owns the LLM agent.
    config : dict[str, Any] | None, optional
        Unused manager config kept for interface consistency.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}
        self.llm_agent = models.get(Agent)
        self.multi_speaker_enabled = models.get(SpeakerDiarization) is not None

    @Manager.event_handler(ASRResultPartial, priority=20)
    async def _handle_asr_result_partial(self, event: ASRResultPartial) -> None:
        """Forward ``ASRResultPartial`` into the agent."""

        await self._accept_event_context(event, context_type="asr_partial")

    @Manager.event_handler(ASRResultFinal, priority=20)
    async def _handle_asr_result_final(self, event: ASRResultFinal) -> None:
        """Forward ``ASRResultFinal`` into the agent."""

        if self.multi_speaker_enabled:
            return
        await self._accept_event_context(event, context_type="asr_final")

    @Manager.event_handler(MultiSpeakerTurnReady, priority=20)
    async def _handle_multi_speaker_turn_ready(
        self,
        event: MultiSpeakerTurnReady,
    ) -> None:
        """Forward the joined ASR/MTD hard turn into the agent."""

        if not self.multi_speaker_enabled:
            return
        await self._accept_event_context(event, context_type="multi_speaker_final")

    @Manager.event_handler(ResponseUpdate, priority=20)
    async def _handle_response_update(self, event: ResponseUpdate) -> None:
        """Forward ``ResponseUpdate`` into the agent."""

        await self._accept_event_context(event, context_type="response_update")

    @Manager.event_handler(ResponseFinish, priority=20)
    async def _handle_response_finish(self, event: ResponseFinish) -> None:
        """Forward ``ResponseFinish`` into the agent."""

        await self._accept_event_context(event, context_type="response_finish")

    @Manager.event_handler(CaptionUpdated, priority=20)
    async def _handle_caption_updated(self, event: CaptionUpdated) -> None:
        """Forward ``CaptionUpdated`` into the agent."""

        await self._accept_event_context(event, context_type="caption")

    @Manager.event_handler(SpeakerRecognized, priority=20)
    async def _handle_speaker_recognized(self, event: SpeakerRecognized) -> None:
        """Forward ``SpeakerRecognized`` into the agent."""

        await self._accept_event_context(event, context_type="speaker")

    @Manager.event_handler(EmbeddingStatusUpdated, priority=20)
    async def _handle_embedding_status_updated(
        self,
        event: EmbeddingStatusUpdated,
    ) -> None:
        """Forward ``EmbeddingStatusUpdated`` into the agent."""

        await self._accept_event_context(event, context_type="embedding")

    @Manager.event_handler(LLMAgentLoop, priority=20)
    async def _handle_llm_agent_loop(self, event: LLMAgentLoop) -> None:
        """Forward ``LLMAgentLoop`` into the agent."""

        await self._accept_event_context(event, context_type="loop")

    async def _accept_event_context(
        self,
        event: Event,
        *,
        context_type: str,
    ) -> None:
        """Convert one event into an ``AgentContext`` payload.

        Parameters
        ----------
        event : Event
            Source event published within the current session.
        context_type : str
            Logical agent-context stream name.
        """

        if self.llm_agent is None:
            return

        context: AgentContext = {
            "type": context_type,
            "data": {
                field.name: getattr(event, field.name)
                for field in fields(event)
                if field.name not in _BASE_EVENT_FIELD_NAMES
            },
        }
        try:
            await self.event_bus.publish(
                ConsumeLLMAgentGenerationRequested(
                    session_id=self.session_id,
                    stream=self.llm_agent.async_accept(context),
                    persistent=context_type == "loop",
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )
        except Exception as e:
            logger.warning(
                "Failed to forward %s context to LLM agent - session: %s, error: %s",
                context_type,
                self.session_id,
                e,
            )

    async def shutdown(self) -> None:
        """Release manager resources."""

        return None
