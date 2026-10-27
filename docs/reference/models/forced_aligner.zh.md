# 强制对齐接口

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
    ) -> list[ForcedAlignmentUnit]: ...

    async def async_align(
        self,
        *,
        audio: bytes,
        text: str,
        language: str | None = None,
    ) -> list[ForcedAlignmentUnit]: ...

    @abstractmethod
    def clone(self) -> "ForcedAligner": ...
```

`ForcedAligner` 把文本单元映射到合成音频的时间线上。配置 `forced_aligner` 即启用播放校准，没有额外的 `enabled` 开关。服务层调用 `async_align`；其默认实现在线程池中运行 `align`，具有原生异步客户端的实现可以覆盖它。

## `ForcedAlignmentUnit`

- `text`：当前字或词。
- `start_ms`、`end_ms`：相对于输入音频起点的毫秒时间戳。
- `char_start`、`char_end`：该单元在原始文本中的左闭右开字符区间。模型无法提供字符位置时保留 `-1`，播放管理器根据 `text` 重新映射。

## `align`

### 参数

- `audio`：48 kHz、单声道、有符号 16 位小端 PCM 原始字节，不含 WAV 头。
- `text`：`audio` 中朗读的文本。
- `language`：可选的模型语言提示。

### 返回值

- `list[ForcedAlignmentUnit]`：按时间排序的单元，时间戳单位为毫秒。推理可以在本地执行，也可以像 `Qwen3ForcedAligner` 一样请求远程 vLLM 服务。

## `clone`

为新会话返回可独立使用的实例。模型权重和连接配置可以共享；可变的请求状态只属于一个会话。`ForcedAligner` 没有 `reset`，每次 `align` 调用相互独立。参见[模型对象的 `clone()` 与 `reset()`](../model_clone_reset.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[先播放、后校准 →](../../explanation/forced_aligner.zh.md)** · Explanation

-   **[注册新模型 →](../../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[可用的强制对齐模型 →](../supported_models.zh.md)** · Reference

</div>
