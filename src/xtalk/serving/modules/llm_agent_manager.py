# -*- coding: utf-8 -*-
import asyncio
import json
from typing import Any, Optional
from langchain_core.messages import ToolCall
from ...log_utils import logger
from ..event_bus import EventBus
from ..events import (
    TurnLLMAgentStartRequested,
    TurnLLMAgentResumeRequested,
    TurnLLMAgentPauseRequested,
    TurnLLMAgentStopRequested,
    TurnTTSStartRequested,
    TurnTTSTextAppendRequested,
    TurnTTSStopRequested,
    TurnTTSPauseRequested,
    TurnTTSResumeRequested,
    TurnTTSFlushRequested,
    LLMAgentResponseUpdate,
    LLMAgentResponseFinish,
    ToolCallOccurred,
    ErrorOccurred,
    LLMModelSwitchRequested,
    LLMFirstChunk,
)
from ..interfaces import Manager
from ...pipelines import Pipeline


class LLMAgentManager(Manager):
    """Drive LLM agent generation and coordinate TTS streaming."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: dict | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.pipeline = pipeline
        self.config: dict[str, Any] = config or {}
        self.llm_agent = pipeline.get_agent()

        self._llm_task: Optional[asyncio.Task] = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()

        self._turn_id = 0

    @Manager.event_handler(LLMModelSwitchRequested, priority=100)
    async def _handle_llm_model_switch(self, event: LLMModelSwitchRequested) -> None:
        """Handle runtime LLM model switch requests."""
        model_name = event.model_name
        base_url = event.base_url
        api_key = event.api_key
        extra_body = event.extra_body

        try:
            if hasattr(self.pipeline, "set_llm_model"):
                self.pipeline.set_llm_model(model_name, base_url, api_key, extra_body)
            else:
                logger.warning(
                    "Pipeline does not support set_llm_model - session: %s",
                    self.session_id,
                )
        except Exception as e:
            logger.error(
                "Failed to switch LLM model: %s - session: %s",
                e,
                self.session_id,
            )

    @Manager.event_handler(TurnLLMAgentStartRequested, priority=98)
    async def _handle_generation_request(
        self, event: TurnLLMAgentStartRequested
    ) -> None:
        """Start generation for a new turn."""
        text = event.text

        await self._cancel_running_task()
        self._llm_task = asyncio.create_task(self._generate_response(text))
        self._turn_id += 1

    @Manager.event_handler(TurnLLMAgentResumeRequested, priority=95)
    async def _handle_generation_resume(
        self, event: TurnLLMAgentResumeRequested
    ) -> None:
        """Resume a paused generation task."""
        if not self._llm_task or self._llm_task.done():
            return
        if not self._resume_event.is_set():
            self._resume_event.set()
            await self.event_bus.publish(
                TurnTTSResumeRequested(session_id=self.session_id)
            )
        else:
            logger.warning(
                "Try to resume not paused LLM generation - session: %s",
                self.session_id,
            )

    @Manager.event_handler(TurnLLMAgentPauseRequested, priority=96)
    async def _handle_generation_pause(self, event: TurnLLMAgentPauseRequested) -> None:
        """Pause the running generation task."""
        if not self._llm_task or self._llm_task.done():
            return
        if not self._resume_event.is_set():
            logger.warning(
                "Try to pause paused LLM generation - session: %s",
                self.session_id,
            )
            return
        self._resume_event.clear()
        await self.event_bus.publish(
            TurnTTSPauseRequested(
                session_id=self.session_id,
            )
        )

    @Manager.event_handler(TurnLLMAgentStopRequested, priority=95)
    async def _handle_generation_stop(self, event: TurnLLMAgentStopRequested) -> None:
        """Stop the running generation task and flush downstream components."""
        reason = event.reason
        await self._cancel_running_task()
        await self.event_bus.publish(
            TurnTTSStopRequested(
                session_id=self.session_id,
                reason=reason,
            ),
            wait_for_completion=True,
        )
        self._resume_event.set()

    async def _cancel_running_task(self) -> None:
        """Cancel any in-flight LLM generation task."""
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
            try:
                await self._llm_task
            except asyncio.CancelledError:
                pass
        self._llm_task = None
        self._resume_event.set()

    async def _generate_response(self, text: str) -> None:
        """Invoke the LLM agent and trigger streaming TTS."""
        agent = self.llm_agent
        if not agent:
            logger.error("LLM agent not configured - session: %s", self.session_id)
            await self._publish_error("llm_not_configured", "LLM agent missing")
            return

        llm_input = {"content": text, "context": self.pipeline.context}
        try:
            await self._stream_tts(agent, llm_input)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("LLM generation failed: %s - session: %s", e, self.session_id)
            await self._publish_error("llm_generation_error", str(e))
            return

    async def _stream_tts(self, agent: Any, llm_input: dict[str, Any]) -> None:
        """Stream agent output to TTS by appending chunks and flushing at the end."""
        text_for_tts_started = False
        first_chunk_generated = False
        accumulated_text = ""
        tool_call: Optional[ToolCall] = None
        stream = agent.async_generate_stream(llm_input)
        iterator = stream.__aiter__()

        try:
            while True:
                if not self._resume_event.is_set():
                    await self._resume_event.wait()
                    continue
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                if not first_chunk_generated:
                    first_chunk_generated = True
                    await self.event_bus.publish(
                        LLMFirstChunk(session_id=self.session_id)
                    )
                chunk_text = ""
                if isinstance(chunk, dict):
                    # Handle tool-call payloads emitted mid-stream
                    # Verify it's actually a tool call by checking required fields
                    if "name" in chunk:
                        tool_call = chunk
                        await self._publish_tool_call(tool_call)
                    else:
                        logger.warning(
                            "Received dict chunk without 'name' field, skipping - session: %s",
                            self.session_id,
                        )
                else:
                    chunk_text = str(chunk)

                if not chunk_text:
                    continue
                accumulated_text += chunk_text
                await self._publish_llm_response_update(accumulated_text)

                if not text_for_tts_started:
                    await self.event_bus.publish(
                        TurnTTSStartRequested(
                            session_id=self.session_id,
                        ),
                        wait_for_completion=True,
                    )
                    text_for_tts_started = True
                await self.event_bus.publish(
                    TurnTTSTextAppendRequested(
                        session_id=self.session_id,
                        text=chunk_text,
                        reason="llm_stream",
                    ),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("LLM stream failed: %s - session: %s", e, self.session_id)
            await self._publish_error("llm_stream_error", str(e))
        finally:
            try:
                if text_for_tts_started:
                    await self.event_bus.publish(
                        TurnTTSFlushRequested(session_id=self.session_id)
                    )
                    await self._publish_llm_response_finish(accumulated_text)
            except Exception as e:
                logger.error(
                    "Error in finally block cleanup: %s - session: %s",
                    e,
                    self.session_id,
                )
            finally:
                # Ensure iterator is always closed
                try:
                    await iterator.aclose()
                except Exception as e:
                    logger.error(
                        "Failed to close iterator: %s - session: %s",
                        e,
                        self.session_id,
                    )

    async def _publish_llm_response_update(self, text: str) -> None:
        """Publish incremental LLM response updates."""
        await self.event_bus.publish(
            LLMAgentResponseUpdate(
                session_id=self.session_id, text=text, turn_id=self._turn_id
            )
        )

    async def _publish_llm_response_finish(self, text: str) -> None:
        """Publish final LLM response event."""
        await self.event_bus.publish(
            LLMAgentResponseFinish(
                session_id=self.session_id, text=text, turn_id=self._turn_id
            )
        )

    async def _publish_tool_call(self, tool_call: ToolCall) -> None:
        """Forward tool-call payloads to downstream consumers."""
        # Validate required fields exist
        if "name" not in tool_call:
            logger.warning(
                "Tool call missing 'name' field - session: %s",
                self.session_id,
            )
            return
        
        name = tool_call["name"]
        args = tool_call.get("args", {})
        
        # Validate name is not empty
        if not name:
            logger.warning(
                "Tool call has empty 'name' field - session: %s",
                self.session_id,
            )
            return
        
        try:
            await self.event_bus.publish(
                ToolCallOccurred(
                    session_id=self.session_id,
                    name=str(name),
                    args=dict(args) if args else {},
                )
            )
        except Exception as e:
            logger.error(
                "Failed to publish tool call '%s': %s - session: %s",
                name,
                e,
                self.session_id,
            )

    async def _publish_error(self, error_type: str, message: str) -> None:
        await self.event_bus.publish(
            ErrorOccurred(
                session_id=self.session_id,
                error_type=error_type,
                error_message=message,
                component="LLMAgentManager",
            )
        )

    async def shutdown(self) -> None:
        await self._cancel_running_task()