## Custom Tools

X-Talk attaches Python-only tools during staged configuration, before the
configured Agent is instantiated:

```python
xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .add_agent_tools([TimerTool])
    .build()
)
```

`add_agent_tools` is an `XtalkBuilder` method. It accepts LangChain tool
instances, native X-Talk `SyncTool` or `AsyncTool` classes, and zero-argument
tool factories. Repeated calls append tools in order without modifying the
source configuration dictionary.

Use a factory when every session needs an independent tool instance. Native
asynchronous tools keep their mutable state in each tool call. See
[`examples/sample_app/custom_async_tool.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_async_tool.py)
for a complete timer example.

## Built-in Tools
    
> **Note**
> See source code under [`src/xtalk/models/agents/tools`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/agents/tools) for all built-in tools.
    
Built-in tools include agent-scope ones like `web_search` and `get_time`, and pipeline-control ones such as silence, speech speed, and, when configured, voice and emotion switching. `DefaultAgent` registers `web_search`, `get_time`, `set_speed`, and `silence` by default. `set_voice` and `set_emotion` are only registered when the corresponding configuration is available.
    
> **Note**
> To enable the `web_search` tool, set `SERPER_API_KEY` or `GOOGLE_SERPER_API_KEY`. See [SerperDev](https://serper.dev/).
