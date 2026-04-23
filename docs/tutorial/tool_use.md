> [!NOTE]
> See `examples/sample_app/mental_consultant_server.py` for details.
    
X-Talk supports textual tool customization through `add_agent_tools`:
```python
xtalk_instance.add_agent_tools([build_mental_questionnaire_tool])
```
    
Here tool should be a [Langchain tool](https://docs.langchain.com/oss/python/langchain/tools):
```python
from langchain.tools import tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

In order to maintain seperate states for a tool in echo agent, you can also use a tool factory to maintain internal states (see `build_mental_questionnaire_tool` `examples/sample_app/mental_consultant_server.py`)

## Built-in Tools
    
> [!NOTE]
> See source code under `src/xtalk/llm_agent/tools` for all built-in tools.
    
Built-in tools include agent-scope ones like `web_search` and `get_time`, and pipeline control ones like emotion, timbre and speed of speech. `DefaultAgent` has built-in tools registered by default.
    
> [!NOTE]
> In order to enable `web_search` tool, `SERPER_API_KEY` needs to be set. See [SerperDev](https://serper.dev/).