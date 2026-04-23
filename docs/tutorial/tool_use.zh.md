> [!NOTE]
> 详情请参阅 `examples/sample_app/mental_consultant_server.py`。

X-Talk 支持通过 `add_agent_tools` 自定义文本工具：
```python
xtalk_instance.add_agent_tools([build_mental_questionnaire_tool])
```

这里的工具应当是一个 [Langchain tool](https://docs.langchain.com/oss/python/langchain/tools)：
```python
from langchain.tools import tool
def search_database(query: str, limit: int = 10) -> str:
    """在客户数据库中搜索与查询匹配的记录。

    Args:
        query: 要搜索的关键词
        limit: 返回结果的最大数量
    """
    return f"Found {limit} results for '{query}'"
```

为了让 echo agent 中的工具维持彼此独立的状态，您也可以使用工具工厂来维护内部状态（参见 `examples/sample_app/mental_consultant_server.py` 中的 `build_mental_questionnaire_tool`）。

## 内置工具

> [!NOTE]
> 所有内置工具请参阅 `src/xtalk/llm_agent/tools` 下的源码。

内置工具包括作用于 agent 级别的工具，例如 `web_search` 和 `get_time`，也包括用于控制流水线的工具，例如情绪、音色和语速。`DefaultAgent` 默认已经注册了这些内置工具。

> [!NOTE]
> 要启用 `web_search` 工具，需要设置 `SERPER_API_KEY`。详见 [SerperDev](https://serper.dev/)。
