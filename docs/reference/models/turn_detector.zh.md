# Turn Detector 接口

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
    def listening(self) -> bool: ...

    @listening.setter
    def listening(self, value: bool) -> None: ...

    def listening_lock(self, is_async: bool = True): ...

    @abstractmethod
    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
        assistant_interrupted: bool = False,
    ) -> TurnDetectionResult: ...

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
        assistant_interrupted: bool = False,
    ) -> TurnDetectionResult: ...

    @abstractmethod
    def clone(self) -> "TurnDetector": ...
```

`TurnDetector` 根据音频、ASR 文本、已播放的 AI 回复文本和 VAD 侧信号判断轮次。服务层调用 `async_detect`；其默认实现在线程池中运行 `detect`。

## `async_detect`

返回当前时刻的轮次判断结果。

### 参数

- `audio`：当前音频帧，PCM 16-bit、单声道、16 kHz 字节流。
- `text`：当前轮次截至目前的 ASR 文本。
- `assistant_text`：已确认播放给用户的累计 AI 回复文本。`None` 表示本次调用不携带 AI 回复更新。该值是累计前缀；detector 保存时覆盖旧值，重复收到相同值时状态保持不变。
- `speech_start`：VAD 刚检测到说话开始时为 `True`。
- `speech_pause`：用户当前可能出现停顿时为 `True`，通常与 `text` 一起传入。它只提示“当前出现停顿”，不带重置语义。
- `assistant_interrupted`：AI 语音在正常播放完成前刚被打断时为 `True`。

### 返回值

`TurnDetectionResult`，含三个字段：

- `action`：服务层立即执行的动作。
    - `DO_NOTHING`：不触发额外动作。
    - `STOP_SPEAKING`：中断系统正在播放的语音；用于用户打断系统说话。
    - `START_GENERATION`：开始生成回复；用于确认用户已经说完。
- `semantic`：detector 对会话状态的语义判断。`action` 与 `semantic` 相关且必须一致；服务层的即时行为只由 `action` 决定。
    - `IDLE`：没有明确的轮次推进信号。
    - `INCOMPLETE`：用户仍在继续当前轮次。
    - `COMPLETE`：用户当前输入在语义上已完整。
    - `WAIT`：用户明确表达了等待意图。
    - `BACKCHANNEL`：输入属于短促附和，不计为轮次完成。
    - `SHOULD_BACKCHANNEL`：系统可以产生 backchannel。
- `vad_result`：可选；仅在 detector 同时代理 VAD 状态时使用。`SPEECH` 表示说话，`SILENCE` 表示静音。未配置独立 VAD 时服务层用它触发 VAD 行为；存在前端或后端 VAD 时忽略该字段。

## `listening`

基类内置的状态及其锁，区分 detector 当前是在监听用户完成输入，还是在监听用户是否打断系统输出。

- `listening = True`：系统等待用户说完；detector 决定何时发出 `START_GENERATION`。
- `listening = False`：系统正在播放输出；detector 决定用户输入是否触发 `STOP_SPEAKING`。

`TurnDetectorManager` 在 TTS 开始播放时将 `listening` 置为 `False`，在播放结束或被中断后恢复为 `True`。

## `clone`

detector 持有的会话状态只属于当前实例。参见[模型对象的 `clone()` 与 `reset()`](../model_clone_reset.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[输入组合与判定路径 →](../../explanation/turn_detector.zh.md)** · Explanation

-   **[注册新模型 →](../../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[在 `detect` 中包装异步检测 →](../../how_to/adapt_async_models.zh.md)** · How-to

</div>
