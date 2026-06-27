> **Note**
> See [`examples/sample_app/mental_consultant_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/mental_consultant_server.py) for details.
    
X-Talk supports textual tool customization through `add_agent_tools`:
```python
xtalk_instance.add_agent_tools([build_mental_questionnaire_tool])
```

`add_agent_tools` must be called before any sessions are created.
    
Here tool should be a [Langchain tool](https://docs.langchain.com/oss/python/langchain/tools):
```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

If you want each session to keep an independent internal tool state, you can also use a tool factory (see `build_mental_questionnaire_tool` in [`examples/sample_app/mental_consultant_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/mental_consultant_server.py)).

## Built-in Tools
    
> **Note**
> See source code under [`src/xtalk/models/agents/tools`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/agents/tools) for all built-in tools.
    
Built-in tools include agent-scope ones like `web_search` and `get_time`, and pipeline-control ones such as silence, speech speed, and, when configured, voice and emotion switching. `DefaultAgent` registers `web_search`, `get_time`, `set_speed`, and `silence` by default. `set_voice` and `set_emotion` are only registered when the corresponding configuration is available.
    
> **Note**
> To enable the `web_search` tool, set `SERPER_API_KEY` or `GOOGLE_SERPER_API_KEY`. See [SerperDev](https://serper.dev/).
