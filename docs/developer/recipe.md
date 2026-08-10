# Recipe

The examples below show how to extend the framework by modifying it directly.

## Introduce a New ASR Model

Assume you want to add `Qwen3ASRFlashRealtime`, whose implementation currently lives in [`src/xtalk/models/asr/qwen3_asr_flash_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/qwen3_asr_flash_realtime.py).

1. Create `qwen3_asr_flash_realtime.py` under [`src/xtalk/models/asr`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/asr).
2. Prepare the class skeleton and implement the required methods. For model interfaces, refer to [`src/xtalk/models/*/interfaces.py`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models); for interface details, see docs such as [ASR Design](../docs/asr_design.md).

```python
from xtalk import model

from ..interfaces import ASR


@model
class Qwen3ASRFlashRealtime(ASR):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        config: Optional[Qwen3ASRFlashConfig] = None,
    ) -> None:
        ...

    def recognize(self, audio: bytes) -> str:
        ...

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        ...

    def stream_chunk_bytes_hint(self) -> int | None:
        ...

    def reset(self) -> None:
        ...

    def clone(self) -> "ASR":
        ...

    async def async_recognize(self, audio: bytes) -> str:
        ...

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        ...
```

3. Decorate the implementation class with `@model` so it can be discovered from config.
4. Use it in the configuration:

```json
"asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "your key"
        }
    }
```

## Introduce a New Agent

Refer to [`src/xtalk/models/agents/experimental.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/experimental.py). The implementation and configuration process is similar to the section above: inherit the interface and decorate the implementation class with `@model`.

### `accept` Logic

```python
async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]:
    pass
```

The `accept` method subscribes to external inputs and starts the related processing logic. `AgentContext` comes from [`src/xtalk/serving/modules/llm_agent_context_manager.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_context_manager.py). The currently stable context types include `asr_partial`, `asr_final`, and `loop`.

The `loop` event is triggered once when the connection is established. It can be used for any proactive logic, or to start an output loop. [`src/xtalk/models/agents/experimental.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/experimental.py) uses it to trigger proactive dialogue.

`AgentOutput` can be a string, a tool call, a tool call result, or an
`AgentTurnBoundary`. After a tool call returns, the `Manager` can use it to
trigger related logic. For example, in
[`src/xtalk/serving/modules/llm_agent_context_manager.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_context_manager.py),
the `direct_audio` tool call triggers downstream logic in
[`src/xtalk/serving/modules/direct_audio_manager.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/direct_audio_manager.py)
to generate directly playable audio events.

Finite output streams finish the current response when their iterator ends.
Long-lived streams started by `loop` should yield `AgentTurnBoundary()` after
each response to trigger TTS flush and response finish.

From a design perspective, `Agent` is expected to be the main reasoning core of the whole system and to integrate information from other components into output.

## Introduce a New Manager

The `Agent` in the previous section requires a new [`src/xtalk/serving/modules/direct_audio_manager.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/direct_audio_manager.py) to forward tool-call output into audio events. All `Manager` implementations can be created directly under [`src/xtalk/serving/modules`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/serving/modules), and then registered in [`src/xtalk/serving/service.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py) and [`src/xtalk/serving/module_types.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/module_types.py).

`Manager` uses the observer pattern for event subscription and publishing. All events are defined in [`src/xtalk/serving/events.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py). [`src/xtalk/serving/modules/input_gateway.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/input_gateway.py) and [`src/xtalk/serving/modules/output_gateway.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/output_gateway.py) are special cases responsible for receiving frontend input and sending output back to the frontend.

To invoke models inside a `Manager`, refer to [`src/xtalk/serving/modules/asr_manager.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/asr_manager.py). You can use methods such as `pipeline.get_asr_model`.

The `mode` argument of the event-bus `publish` method controls when the call
returns. `RETURN_AFTER_DISPATCH` schedules listeners in the background,
`WAIT_UNTIL_COMPLETE` waits for every directly triggered listener in priority
order, and `WAIT_UNTIL_COMPLETE_OR_STOPPED` additionally lets a listener stop
lower-priority propagation by returning `EventPropagation.STOP`. The short
strings `dispatch`, `wait`, and `wait_stoppable` are accepted as aliases, while
application code should prefer the enum members. Waiting along an event chain
ensures that every handler in that chain finishes before control returns to the
original event source.

## Introduce a New Model Type

Create the [`src/xtalk/models/<model_type>`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models) folder, define the interface in [`src/xtalk/models/<model_type>/interfaces.py`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models), decorate it with `@model_type(aliases=[...])` imported from `xtalk`, then create the corresponding model files in the same folder. The folder name becomes the primary config key; aliases can be used to define additional config-key aliases.
