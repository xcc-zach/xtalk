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

Forced aligner 只校准文本播放位置，不应阻塞音频。音频准备与带背压的发送相互独立：生产端先完成语速处理，将最终 PCM 放入发送队列，并在合成结束后立即通过`TTSTextSynthesized`发布完整最终 PCM；播放管理器随即启动`async_align`，发送端则独立地按播放背压消费 PCM，并在该片段发送完成后发布`TTSTextDeliveryFinished`。

不引入`segment_id`。每个 session 内的准备边界、发送边界、音频块和`TTSChunkPlayed`必须严格 FIFO，且不同片段的音频不能交错。播放管理器通过两个 FIFO 队列配对准备流与发送流，并校验两侧文本一致。每个准备完成的片段都会立即创建独立的 forced-alignment 任务，不限制 session 内的对齐并发数；即使结果乱序返回，也会写入各任务捕获的片段对象。

### TTS：两级校准

普通 TTS 在开始合成前已知完整句子。带背压的发送端在开始发送时发布`TTSTextSynthesisStarted`，消费到句尾标记后发布`TTSTextDeliveryFinished`。`TTSTextSynthesized`由生产端直接发布，可能早于或晚于发送开始；严格 FIFO 配对会把两条事件流绑定到同一个播放片段。

| 阶段 | 音频播放 | 当前校准方式 |
| --- | --- | --- |
| 句子仍在合成 | 立即开始，不等待对齐 | 使用保守在线估算的 **L1 粗略校准** |
| 最终 PCM 已可用，forced alignment 正在运行 | 不间断地继续播放 | 使用 `played_ms / total_audio_ms` 的 **L1 粗略校准** |
| Forced alignment 结果可用 | 不受影响 | 使用字/词时间戳的 **L2 精准校准** |
| 播放先于 forced alignment 完成 | 正常完成 | 确认整句并取消该句未完成的对齐任务 |

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

StreamingTextTTS 在生成过程中最终文本和总音频时长都未知，但音频块仍应立即播放。每个 session 只维护一个隐式流状态，并在`append_text`成功后累加已被 TTS 接受的文本。音频读取任务独立于带背压的发送准备语速处理后的 PCM；上游音频流结束后立即通过`TTSTextSynthesized`发布完整 PCM，切换到总时长比例并启动 forced alignment，同时发送队列中的音频继续播放。

| 级别 | 使用阶段 | 所需信息 |
| --- | --- | --- |
| **L1 在线估算** | 文本流和音频流仍在生成 | 已接受文本、已准备音频时长和已确认播放音频时长 |
| **L2 总时长比例** | 完整音频流已可用，forced alignment 正在运行 | 完整已接受文本、最终处理后的 PCM 和实际总音频时长 |
| **L3 精准校准** | Forced alignment 已返回有效单元 | 映射到完整文本的字级或词级时间戳 |

实际总时长必须从最终发送的 PCM 计算，不能简单使用原始时长除以语速。

### 校准切换与异常处理

`ResponseUpdate`保持单调增长：精准结果领先时立即前进，落后于已显示位置时不回退，而是暂停增长，等待精准时间线追上。

对齐失败时继续使用粗略结果。完整播放不需要等待对齐；提前中断时优先使用已有精准结果，可短暂等待正在执行的对齐任务，超时后回退到粗略前缀。StreamingTextTTS 在完整音频生成前被中断时通常只能使用在线估算，除非服务端能提供已消费文本位置。

### 调试前端文本更新

设置 `XTALK_LOG_LEVEL='xtalk.serving.modules.tts_playback_manager=DEBUG'` 后，可以检查每次面向前端的 `ResponseUpdate` 由哪种判定产生。每次更新只输出一条日志，例如：

```text
TTS response update - session: ..., source: regular:L2-precise, mode: regular, level: L2-precise, state: ready, played_ms: 820.0, total_ms: 1600.0, delta: '本次新增文本', text: '完整已显示文本前缀'
```

当已播放片段完成并确认剩余文本时，`source`为`playback-complete`；当前校准等级推进文本前缀时，`source`为`<mode>:<level>`。一次更新跨越片段边界时，两种来源用`+`连接；只有最终兜底更新无法归入上述两类时，才使用防御性的`final-commit`来源。`delta`和`text`使用转义后的字符串表示，因此文本中的换行符不会把日志拆成多行。这些字段可能包含对话内容，只应在需要调试时启用。
