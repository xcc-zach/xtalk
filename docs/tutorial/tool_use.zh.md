> **Note**
> 详情请参阅 [`examples/sample_app/mental_consultant_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/mental_consultant_server.py)。

X-Talk 支持通过 `add_agent_tools` 自定义文本工具：
```python
xtalk_instance.add_agent_tools([build_mental_questionnaire_tool])
```

请注意，`add_agent_tools` 需要在任何会话创建之前调用。

这里的工具应当是一个 [Langchain tool](https://docs.langchain.com/oss/python/langchain/tools)：
```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """在客户数据库中搜索与查询匹配的记录。

    Args:
        query: 要搜索的关键词
        limit: 返回结果的最大数量
    """
    return f"Found {limit} results for '{query}'"
```

如果您希望每个会话中的工具都维持彼此独立的内部状态，也可以使用工具工厂（参见 [`examples/sample_app/mental_consultant_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/mental_consultant_server.py) 中的 `build_mental_questionnaire_tool`）。

## 内置工具

> **Note**
> 所有内置工具请参阅 [`src/xtalk/models/agents/tools`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/agents/tools) 下的源码。

内置工具包括作用于 agent 级别的工具，例如 `web_search` 和 `get_time`，也包括用于控制流水线的工具，例如静音、语速，以及在配置可用时的音色和情绪切换。`DefaultAgent` 默认会注册 `web_search`、`get_time`、`set_speed` 和 `silence`；`set_voice` 与 `set_emotion` 只会在对应配置存在时注册。

> **Note**
> 要启用 `web_search` 工具，需要设置 `SERPER_API_KEY` 或 `GOOGLE_SERPER_API_KEY`。详见 [SerperDev](https://serper.dev/)。
