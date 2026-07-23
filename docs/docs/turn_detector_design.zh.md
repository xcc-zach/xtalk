```python
class TurnDetectionAction(Enum):
    DO_NOTHING = 1
    STOP_SPEAKING = 2
    START_GENERATION = 3


class TurnDetectionSemantic(Enum):
    IDLE = "idle"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"
    SHOULD_BACKCHANNEL = "should_backchannel"


class TurnVADResult(Enum):
    SPEECH = 1
    SILENCE = 2


@dataclass(frozen=True)
class TurnDetectionResult:
    action: TurnDetectionAction
    semantic: TurnDetectionSemantic
    vad_result: TurnVADResult | None = None


class TurnDetector(ABC):
    """Abstract interface for turn-taking detectors."""

    @property
    def listening(self) -> bool:
        ...

    @listening.setter
    def listening(self, value: bool) -> None:
        ...

    def listening_lock(self, is_async: bool = True):
        ...

    @abstractmethod
    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        ...

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        ...

    @abstractmethod
    def clone(self) -> "TurnDetector":
        ...
```

## `detect`实现的最佳实践

框架中实际调用的是 `async_detect`。因此，实现新的 turn detector 时，最佳实践是首先实现 `async_detect`，再在 `detect` 中对其进行同步包装。

```python
import asyncio

def detect(
    self,
    audio: Optional[bytes] = None,
    text: Optional[str] = None,
    assistant_text: Optional[str] = None,
    speech_start: bool = False,
    speech_pause: Optional[bool] = None,
) -> TurnDetectionResult:
    return asyncio.run(
        self.async_detect(
            audio=audio,
            text=text,
            assistant_text=assistant_text,
            speech_start=speech_start,
            speech_pause=speech_pause,
        )
    )
```

如果底层实现本身已经是同步的，也可以直接实现 `detect`，并复用基类提供的 `async_detect` 默认包装。

## `async_detect`说明

`TurnDetector` 支持同时消费音频信号、ASR 文本、已播放的 AI 回复文本和
VAD 侧信号。每次调用都应返回一个 `TurnDetectionResult`，表示当前时刻的
轮次判断结果。

### 输入参数

- `audio`：当前音频帧，格式为 PCM 16-bit、单声道、16 kHz 字节流。
- `text`：当前轮次截至目前的 ASR 文本。
- `assistant_text`：已经确认播放给用户的累计 AI 回复文本。`None` 表示
  本次调用不携带 AI 回复更新。
- `speech_start`：VAD 刚检测到说话开始时传入的信号。
- `speech_pause`：用户当前可能出现停顿时传入的信号，通常与 `text` 一起使用。

这些参数同时出现的组合如下：

- 仅传入 `audio`，走纯音频判定路径（音频路径）
- 仅传入 `text` 和 `speech_pause`，走文本语义判定路径（文本路径）
- 仅传入 `assistant_text`，使用已播放的 AI 回复文本更新 detector 上下文
- 仅传入 `speech_start=True`，通知 detector 当前说话轮次开始（辅助信号）

当前仓库中的两个典型实现分别代表了两种路径：

- `SoulxDuplug`：以音频路径为主，并在文本停顿信号上提供 fallback
- `LLMTurnDetector`：以文本语义路径为主，主要依赖 `text` 与 `speech_pause`

现有 detector 实现可以接收但不使用 `assistant_text`。需要感知 AI 回复的
实现可以把累计值保存为当前 session 的状态，并在后续轮次判断中使用。

## AI 回复文本更新

`TurnDetectorManager` 同时订阅 `ResponseUpdate` 和 `ResponseFinish`。每收到
一个事件，就单独调用一次
`async_detect(assistant_text=event.text)`。这个纯上下文调用的返回值会被忽略，
不会转换成 `STOP_SPEAKING` 或 `START_GENERATION`。

`ResponseUpdate.text` 是已经确认播放给用户的累计文本前缀，而不是文本增量；
`ResponseFinish.text` 是最终完整回复。因此，detector 保存上下文时应覆盖旧值，
不能追加。最后一次 `ResponseUpdate` 和 `ResponseFinish` 可能包含相同文本，
所以状态更新应当具有幂等性。

### 返回值

返回值为 `TurnDetectionResult`，由三部分组成：

- `action`：服务层应立即执行的动作
- `semantic`：当前会话状态的语义解释
- `vad_result`：可选的 VAD 结果，仅在 detector 需要代理输出 VAD 状态时使用

### `TurnDetectionAction`语义

- `DO_NOTHING`：当前不触发额外动作
- `STOP_SPEAKING`：当前应中断系统正在播放的语音
- `START_GENERATION`：当前应开始生成系统回复

其中：

- `STOP_SPEAKING` 一般用于用户打断系统说话
- `START_GENERATION` 一般用于确认用户已经说完，已可开始回答

### `TurnDetectionSemantic`语义

- `IDLE`：当前没有明确的轮次推进信号
- `INCOMPLETE`：用户仍在继续当前轮次，尚未说完
- `COMPLETE`：用户当前输入在语义上已完整
- `WAIT`：用户明确表达了等待语义
- `BACKCHANNEL`：用户输入属于短促附和，不应作为正式轮次完成
- `SHOULD_BACKCHANNEL`：当前状态提示系统可以产生 backchannel

`semantic` 主要用于表达 detector 的语义判断（未来用于LLM的精细语义控制），而 `action` 决定服务层的即时行为。两者相关但不等价。

### `vad_result`语义

`vad_result` 为可选字段，仅在 turn detector 同时承担 VAD 代理职责时使用。

- `TurnVADResult.SPEECH`：当前处于说话状态
- `TurnVADResult.SILENCE`：当前处于静音状态

当 pipeline 中未配置独立 VAD 时，系统使用该字段触发VAD。
如果前端或后端配置了独立 VAD，该字段无效。

## XTurnix

`XTurnix` 是通过 vLLM 部署 XTurnix 模型的有状态文本 detector。每次收到累计
`assistant_text` 时，它只记录上下文而不执行推理；收到累计用户 `text` 时，
它会更新对话历史，并向 vLLM 请求一个受限动作。

detector 按照下表映射框架状态与模型动作：

| `listening` | XTurnix 状态 | 模型动作 | 框架结果 |
| --- | --- | --- | --- |
| `True` | `<|listening|>` | `<|start|>` | `START_GENERATION`、`COMPLETE` |
| `True` | `<|listening|>` | `<|keep|>` | `DO_NOTHING`、`INCOMPLETE` |
| `False` | `<|speaking|>` | `<|stop|>` | `STOP_SPEAKING`、`INCOMPLETE` |
| `False` | `<|speaking|>` | `<|keep|>` | `DO_NOTHING`、`IDLE` |

`speech_pause=True` 会在当前用户文本位置插入模型的原子 `<|pause|>` 标记。
detector 会记录累计文本的源位置，因此 ASR 回改也会删除被此次回改覆盖的停顿
标记。

vLLM 请求始终使用模型名 `xturnix`，只生成一个 token，并将生成范围限制为
当前状态下合法的两个动作。动作 token ID 会通过 vLLM `/tokenize` 接口动态
解析和校验。长对话会在消息边界截断，同时保留携带当前状态的 system prompt
和最新对话。

部署和配置方式请参阅[支持的模型](supported_models.zh.md#轮次检测)。

## `listening`语义

`TurnDetector` 基类内置了 `listening` 状态及其锁。该状态用于区分 detector 当前是在“监听用户完成输入”，还是在“监听用户是否打断系统输出”。

常见约定如下：

- `listening = True`：系统正在等待用户说完，应判断何时 `START_GENERATION`
- `listening = False`：系统正在播放输出，应判断用户输入是否会触发 `STOP_SPEAKING`

服务层中的 `TurnDetectorManager` 会在 TTS 开始播放时将 `listening` 置为 `False`，并在播放结束或被中断后恢复为 `True`。

## 实现建议

- 每次返回结果时，应保证 `action` 与 `semantic` 语义一致。
- 如果 detector 内部维护会话状态，应确保该状态只属于当前实例，不应跨 session 共享。
- 若实现使用 `speech_pause`，应将其视为“当前出现停顿”的提示，而不是 turn 结束后的重置信号。

## `clone`

请参阅[模型对象的 `clone()` 与 `reset()` 语义](model_clone_reset.zh.md)。
