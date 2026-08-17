"""Built-in tool for putting the desktop conversation into sleep mode."""

from __future__ import annotations

from xtalk.models.agents.tools import (
    SyncTool,
    ToolEngineState,
    ToolInput,
    ToolOutput,
)


class EnterSleepModeInput(ToolInput):
    """Empty input accepted by the sleep-mode request."""


class EnterSleepModeOutput(ToolOutput):
    """Instruction returned after the desktop accepts a sleep request."""

    content: str

    def to_content(self) -> str:
        """Return the follow-up instruction as plain text."""

        return self.content


class EnterSleepModeTool(SyncTool):
    """让桌面应用进入后台沉睡模式。

    仅当用户明确要求休息、沉睡、待机或结束当前对话时调用。不要根据沉默、
    暂时没有问题、普通的“谢谢”或模型自己的判断调用。调用后用一句简短的话
    向用户告别，不要再调用其他工具或继续展开对话。

    Enter desktop sleep mode only when the user explicitly asks to sleep,
    stand by, or end the current conversation. Never infer this intent from
    silence, inactivity, thanks, or a casual goodbye. After calling, give one
    brief farewell and do not invoke other tools or continue the conversation.
    """

    name = "enter_sleep_mode"

    @classmethod
    def invoke(
        cls,
        tool_input: EnterSleepModeInput,
        global_state: ToolEngineState,
    ) -> EnterSleepModeOutput:
        """Ask the desktop shell to sleep after its final spoken reply."""

        del cls, tool_input, global_state
        return EnterSleepModeOutput(
            content=(
                "The desktop received the sleep request. Give the user one "
                "brief farewell now. Do not call another tool or continue the "
                "conversation after that sentence."
            )
        )


def create_tools() -> list[type[SyncTool]]:
    """Create the always-enabled desktop sleep-mode tool.

    Returns
    -------
    list[type[SyncTool]]
        Tool classes loaded by the desktop backend.
    """

    return [EnterSleepModeTool]
