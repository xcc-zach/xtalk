## 自定义工具

X-Talk 在配置阶段挂载仅存在于 Python 运行时中的工具，并在完成挂载后实例化配置的
Agent：

```python
xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .add_agent_tools([TimerTool])
    .build()
)
```

`add_agent_tools` 是 `XtalkBuilder` 的方法。它支持 LangChain 工具实例、X-Talk 原生
`SyncTool` 或 `AsyncTool` 类，以及无参数工具工厂。多次调用会按顺序追加工具，且不会
修改传入的原始配置字典。

如果每个会话都需要独立的工具实例，请使用工具工厂。原生异步工具的可变状态由每次
工具调用分别保存。完整定时器示例请参阅
[`examples/sample_app/custom_async_tool.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_async_tool.py)。

## 内置工具

> **Note**
> 所有内置工具请参阅 [`src/xtalk/models/agents/tools`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/agents/tools) 下的源码。

内置工具包括作用于 agent 级别的工具，例如 `web_search` 和 `get_time`，也包括用于控制流水线的工具，例如静音、语速，以及在配置可用时的音色和情绪切换。`DefaultAgent` 默认会注册 `web_search`、`get_time`、`set_speed` 和 `silence`；`set_voice` 与 `set_emotion` 只会在对应配置存在时注册。

> **Note**
> 要启用 `web_search` 工具，需要设置 `SERPER_API_KEY` 或 `GOOGLE_SERPER_API_KEY`。详见 [SerperDev](https://serper.dev/)。
