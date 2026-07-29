```python
@dataclass(frozen=True)
class ForcedAlignmentUnit:
    """一个映射到合成音频时间线的文本单元。"""

    text: str
    start_ms: float
    end_ms: float
    char_start: int = -1
    char_end: int = -1


@model_type
class ForcedAligner(ABC):
    """强制对齐模型的抽象接口。"""

    @abstractmethod
    def align(
        self,
        *,
        audio: bytes,
        text: str,
        language: str | None = None,
    ) -> list[ForcedAlignmentUnit]:
        """将文本单元与 48 kHz PCM 音频对齐。"""
        pass

    async def async_align(
        self,
        *,
        audio: bytes,
        text: str,
        language: str | None = None,
    ) -> list[ForcedAlignmentUnit]:
        """异步完成强制对齐。"""
        ...

    @abstractmethod
    def clone(self) -> "ForcedAligner":
        """为新服务会话克隆 aligner。"""
        pass
```

## `ForcedAlignmentUnit`

`text`为当前字或词；`start_ms`和`end_ms`是相对于输入音频起点的毫秒时间戳。

`char_start`和`char_end`是该单元在原始文本中的左闭右开字符区间。模型无法直接提供字符位置时可保留默认值`-1`，播放管理器会根据`text`重新映射。

## `align`

`audio`必须是 48 kHz、单声道、有符号 16 位小端 PCM 原始字节，不包含 WAV 头；`text`应与音频中实际朗读的文本一致；`language`是可选的模型语言提示。

返回值应按时间顺序排列，时间戳不得使用音频采样点或秒作为单位。实现可以执行本地推理，也可以像`Qwen3ForcedAligner`一样请求远程 vLLM 服务。

## `async_align`

框架使用`async_align`避免阻塞事件循环。默认实现在线程池中调用同步`align`；具有原生异步客户端的实现可以覆盖该方法。

## `clone`

`clone`为新 session 返回可独立使用的实例。模型权重、连接配置等不可变资源可以共享，但可变的请求状态不应跨 session 共享。Forced aligner 没有`reset`接口，每次`align`调用应相互独立。

配置`forced_aligner`后，播放校准自动启用，不再需要额外的`enabled`开关。

## 先播放、后校准

Forced aligner 只校准文本播放位置，不应阻塞音频。系统应立即发送最终将要播放的、经过语速处理的 PCM，同时在后台收集相同音频；完整音频可用后再异步调用`async_align`。

不引入`segment_id`，因此必须保证每个 session 内文本边界、音频块和`TTSChunkPlayed`严格 FIFO，且多个句子的音频不能交错输出。

### TTS：两级校准

普通 TTS 在开始合成前已知完整句子。内部队列依次放入句子开始标记、音频块和句子结束标记，播放管理器据此建立边界。

1. **粗略校准**：音频生成期间，根据完整文本、已生成音频时长和已播放时长估算前缀；音频生成完成后，用实际总时长替换预计时长，继续按播放比例估算。
2. **精准校准**：句子音频完整后后台调用 forced aligner，结果返回后改用字/词时间戳。

开放音频期间可采用保守估算：

```text
safe_played_ms = max(0, played_ms - 200)
estimated_total_ms = max(
    text_weight × estimated_ms_per_unit,
    generated_audio_ms + 300,
)
```

中文按字、英文按词计算`text_weight`，标点附着到相邻单元并增加少量停顿权重。句子结束后，粗略比例改为`played_ms / total_audio_ms`。

### StreamingTextTTS：三级校准

StreamingTextTTS 在生成过程中最终文本和总音频时长都未知，但音频块仍应立即播放。每个 session 只维护一个隐式流状态，并在`append_text`成功后累加已被 TTS 接受的文本。

1. **在线估算**：根据当前已接受文本、已生成音频时长和已播放时长使用上述保守公式估算。
2. **总时长比例**：完整音频流结束后，使用完整文本和实际总音频时长计算`played_ms / total_audio_ms`。
3. **精准校准**：同时后台请求 forced aligner，结果返回后改用字/词时间戳。

实际总时长必须从最终发送的 PCM 计算，不能简单使用原始时长除以语速。

### 校准切换与异常处理

`ResponseUpdate`保持单调增长：精准结果领先时立即前进，落后于已显示位置时不回退，而是暂停增长，等待精准时间线追上。

对齐失败时继续使用粗略结果。完整播放不需要等待对齐；提前中断时优先使用已有精准结果，可短暂等待正在执行的对齐任务，超时后回退到粗略前缀。StreamingTextTTS 在完整音频生成前被中断时通常只能使用在线估算，除非服务端能提供已消费文本位置。
