如前文在 [启动服务](start_the_server.zh.md) 中所述，X-Talk 实例可以通过 JSON 配置创建，该配置用于自定义所使用的模型以及控制并发行为。

对于模型配置，配置内容应与模型 Python 类名及其初始化参数一致。例如，`DefaultAgent` 的定义位于 `src/xtalk/llm_agent/default.py`：
```python
class DefaultAgent(Agent):
    def __init__(
            self,
            model: BaseChatModel | dict,
            system_prompt: str = _BASE_PROMPT,
            voice_names: Optional[List[str]] = None,
            emotions: Optional[List[str]] = None,
            tools: Optional[List[Union[BaseTool, Callable[[], BaseTool]]]] = None,
        ):
    ...
```

为了与初始化参数匹配，配置项应写成这样：
```
"llm_agent": {
    "type": "DefaultAgent",
    "params": {
      "model": {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
      },
      "voice_names": [
        "Man",
        "Woman",
        "Child"
      ],
      "emotions": [
        "happy",
        "angry",
        "sad",
        "fear",
        "disgust",
        "depressed",
        "surprised",
        "calm",
        "normal"
      ]
    }
  },
```

像 `voice_names`、`emotions` 和 `tools`（目前尚不支持在配置中使用）这样的可选键可以省略。

完整的模型类型（`slot`）、对应的可选依赖以及其在源码中的适配位置，请参阅[支持的模型](../docs/supported_models.zh.md)。
> [!NOTE]
> 大多数模型实现都是客户端适配器。您可能还需要按照相应说明启动模型实例本身。

此外，您还可以通过以下配置限制并发数：
```json
    "max_connections": 1
```
