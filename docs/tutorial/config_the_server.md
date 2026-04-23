As mentioned [before](start_the_server.md), X-Talk instance can be created from a JSON config, which customizes models used and controls concurrency behavior.
    
For model config, config should match model Python class name and init args. For example, the definition of `DefaultAgent` lies in `src/xtalk/llm_agent/default.py`:
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
    
In order to match with the init args, the config item should look like:
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

Optional keys like `voice_names`, `emotions` and `tools`(not supported in config yet) can be ignored.
    
See [supported models](../docs/supported_models.md) for the full list of model types (`slot`), their optional dependencies, and their adapting location in source code.
> [!NOTE]
> Most model implementations are client-side adaptors. You may need to start the model instance following coresponding instructions.
    
Also, you can restrict concurrency through:
```json
    "max_connections": 1
```
