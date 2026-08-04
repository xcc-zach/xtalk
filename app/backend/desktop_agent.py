"""Desktop-only agent behavior for asynchronous App tools."""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, SystemMessage, ToolCall, ToolMessage

from xtalk import model
from xtalk.models.agents.default import DefaultAgent
from xtalk.models.agents.interfaces import AgentOutput
from xtalk.models.agents.tools import ToolEngine
from xtalk.models.agents.tools.utils import build_tool_call_result


@model
class DesktopDefaultAgent(DefaultAgent):
    """Expose App tool progress UI while preserving normal LLM narration."""

    def clone(self) -> "DesktopDefaultAgent":
        """Create an equivalent desktop agent for a new conversation."""

        return DesktopDefaultAgent(
            model=self.model,
            backchannel_model=self.backchannel_model,
            backchannel_source_dir=self.backchannel_source_dir,
            system_prompt=self._additional_system_prompt,
            tools=self._base_tools,
            proactive=self.proactive,
        )

    async def _stream_messages_unlocked(
        self,
        *,
        allow_tools: bool,
        transient_instruction: str | None = None,
    ) -> AsyncIterator[AgentOutput]:
        """Stream a turn and let the LLM respond to each initial tool receipt."""

        streaming_model = (
            self.model_with_tools if allow_tools else self.model_for_async_updates
        )
        while True:
            response_message = AIMessage(content="")
            gathered = None
            prompt_messages = list(self.messages)
            if transient_instruction:
                prompt_messages.append(SystemMessage(content=transient_instruction))
            async for chunk in streaming_model.astream(prompt_messages):
                text = self.content_to_text(chunk.content)
                if text:
                    response_message.content += text
                    yield text
                gathered = chunk if gathered is None else gathered + chunk
            tool_calls: list[ToolCall] = (
                list(gathered.tool_calls or []) if gathered else []
            )
            if not tool_calls:
                return

            response_message.content = ""
            response_message.tool_calls = tool_calls
            self._chat_history.append_message(response_message)
            for tool_call in tool_calls:
                yield tool_call
                tool_name = tool_call["name"]
                try:
                    tool_result = await self.tool_engine.ainvoke_and_append(
                        tool_call,
                        self._chat_history.messages,
                    )
                except Exception as exc:
                    tool_result = ToolMessage(
                        content=f"Error invoking tool: {exc}",
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                    ToolEngine.append_tool_message(
                        tool_call,
                        tool_result,
                        self._chat_history.messages,
                    )
                yield build_tool_call_result(
                    tool_call=tool_call,
                    result_content=str(tool_result.content),
                )

    def _record_async_tool_update(
        self,
        tool_call: ToolCall,
        tool_message: ToolMessage,
        output: AgentOutput,
    ) -> None:
        """Store every update but wake speech generation only for a final one."""

        ToolEngine.append_tool_message(
            tool_call=tool_call,
            tool_message=tool_message,
            messages=self._chat_history.messages,
        )
        if not self._is_final_tool_update(tool_message):
            return
        if not self._human_input_finished:
            self._pending_final_reports.append(output)
            return
        self._async_tool_update_queue.put_nowait(output)
