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
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        ...

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
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
    speech_start: bool = False,
    speech_pause: Optional[bool] = None,
) -> TurnDetectionResult:
    return asyncio.run(
        self.async_detect(audio, text, speech_start, speech_pause)
    )
```

如果底层实现本身已经是同步的，也可以直接实现 `detect`，并复用基类提供的 `async_detect` 默认包装。

## `async_detect`说明

`TurnDetector` 支持同时消费音频信号、ASR 文本和 VAD 侧信号。每次调用都应返回一个 `TurnDetectionResult`，表示当前时刻的轮次判断结果。

### 输入参数

- `audio`：当前音频帧，格式为 PCM 16-bit、单声道、16 kHz 字节流。
- `text`：当前轮次截至目前的 ASR 文本。
- `speech_start`：VAD 刚检测到说话开始时传入的信号。
- `speech_pause`：用户当前可能出现停顿时传入的信号，通常与 `text` 一起使用。

这些参数同时出现的组合如下：

- 仅传入 `audio`，走纯音频判定路径（音频路径）
- 仅传入 `text` 和 `speech_pause`，走文本语义判定路径（文本路径）
- 仅传入 `speech_start=True`，通知 detector 当前说话轮次开始（辅助信号）

当前仓库中的两个典型实现分别代表了两种路径：

- `SoulxDuplug`：以音频路径为主，并在文本停顿信号上提供 fallback
- `LLMTurnDetector`：以文本语义路径为主，主要依赖 `text` 与 `speech_pause`

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
