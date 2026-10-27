# Attach Tools

This tutorial ends with the Agent calling a tool you wrote: a LangChain tool that counts characters, attached through the builder and executed by the Agent during a conversation. You will then make the tool per-session with a factory and see how a custom Agent receives the tools.

Before starting, complete [Introduce an LLM Agent](introduce_an_llm_agent.md). The built-in `DefaultAgent` already executes tools, so the first steps work with the Quickstart config; the custom-Agent step builds on the `ToolAgent` sketch below.

## 1. Define and attach a tool

Define a LangChain tool, then attach it to the configured Agent with `XtalkBuilder.add_agent_tools()`:

```python
from langchain_core.tools import tool

from xtalk import Xtalk


@tool
def text_length(text: str) -> int:
    """Return the number of characters in text."""

    return len(text)


xtalk_instance = (
    Xtalk.configure("config.json")
    .add_agent_tools([text_length])
    .build()
)
```

Restart the server and ask "how many characters are in the word hello". The reply contains `5`, computed by `text_length`.

`add_agent_tools()` accepts LangChain tool instances, native X-Talk `SyncTool` or `AsyncTool` classes, and zero-argument factories that return those tools. Repeated calls append tools in order without modifying the source configuration dictionary. The other tool types are described in [Tools](../reference/tools.md).

## 2. Receive the tools in a custom Agent

Tools are ultimately passed to the Agent constructor through `llm_agent.params.tools`, so a custom Agent must accept and use its `tools` parameter. Until it does, the attached tools are silently ignored by that Agent.

A custom Agent can create a `ToolEngine` in its constructor and bind the tools to its chat model:

```python
from langchain_core.language_models.chat_models import BaseChatModel

from xtalk.model_types import Agent
from xtalk.models.agents.tools import Tool, ToolEngine


class ToolAgent(Agent):
    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Tool] | None = None,
    ) -> None:
        self.model = model
        self.tools = list(tools or [])
        self.tool_engine = ToolEngine(tools=self.tools, state={})
        self.model_with_tools = self.tool_engine.bind(model)
```

Calls to `self.model_with_tools` can then produce tool calls, which the Agent executes through `self.tool_engine`. The complete Agent that executes those calls and appends results to history is in [ToolEngine](../reference/tools.md#toolengine).

## 3. Give each session its own tool instance

Pass a tool factory when each session needs an independent tool instance:

```python
from langchain_core.tools import tool


def create_counter_tool():
    count = 0

    @tool
    def increment_counter() -> int:
        """Increment and return this session's counter."""

        nonlocal count
        count += 1
        return count

    return increment_counter


xtalk_instance = (
    Xtalk.configure("config.json")
    .add_agent_tools([create_counter_tool])
    .build()
)
```

`create_counter_tool()` is called whenever a session Agent is created, so sessions do not share `count`. Open two browser tabs and ask each to increment the counter: both answer `1`. A tool factory must be a zero-argument callable.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Customize the Service →](customize_the_service.md)** · Tutorial

-   **[Tools: built-in tools, `SyncTool`, `AsyncTool` and `ToolEngine` →](../reference/tools.md)** · Reference

</div>
