# 适配异步模型

底层客户端是异步或远程服务时，先实现 `async_*` 方法，再在同步抽象方法中包装它；本页假定你已通过[引入新模型](../tutorial/introduce_a_new_model.zh.md)了解 `@model` 注册与配置文件选用模型的方式。

服务层调用每种模型类型的 `async_*` 方法。各基类的 `async_*` 默认在线程池中运行同步方法，因此同步客户端只需实现同步方法。异步客户端反过来：实现 `async_*`，并让同步方法用 `asyncio.run()` 运行它。

下面每个类都是完整的：用 `@model` 注册、实现了全部抽象方法，可在配置文件中按类名选用。示例用 `aiohttp` 访问 `base_url` 处的占位 HTTP 服务；保留方法结构，把请求代码换成你自己的客户端即可。每次调用都新建一个 `aiohttp.ClientSession`，因为 `asyncio.run()` 为每次同步调用创建新的事件循环，绑定到已关闭循环的会话无法复用。

## ASR

服务层调用 `async_recognize_stream` 时传入上次调用之后新到达的音频，并期望得到本轮截至目前的完整文本。本实现把本轮音频保存在内存中，每次调用都把整个缓冲区发给服务。

```python
import asyncio

import aiohttp

from xtalk import model
from xtalk.model_types import ASR


@model
class RemoteASR(ASR):
    def __init__(self, base_url: str, stream_chunk_bytes: int = 6400) -> None:
        self.base_url = base_url
        self._stream_chunk_bytes = stream_chunk_bytes
        self._turn_audio = bytearray()
        self._turn_text = ""

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        del chat_history
        self._turn_audio.extend(audio)
        if not self._turn_audio:
            return self._turn_text
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/recognize",
                data=bytes(self._turn_audio),
                params={"final": "1" if is_final else "0"},
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        self._turn_text = payload["text"]
        return self._turn_text

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        return asyncio.run(
            self.async_recognize_stream(
                audio, is_final=is_final, chat_history=chat_history
            )
        )

    async def async_recognize(self, audio: bytes) -> str:
        self.reset()
        return await self.async_recognize_stream(audio, is_final=True)

    def recognize(self, audio: bytes) -> str:
        return asyncio.run(self.async_recognize(audio))

    def stream_chunk_bytes_hint(self) -> int | None:
        return self._stream_chunk_bytes

    def reset(self) -> None:
        self._turn_audio.clear()
        self._turn_text = ""

    def clone(self) -> "RemoteASR":
        return RemoteASR(
            base_url=self.base_url, stream_chunk_bytes=self._stream_chunk_bytes
        )
```

`stream_chunk_bytes_hint` 告诉服务层每次流式调用前积累多少字节；`6400` 字节对应 200 ms 的 16 kHz PCM 16-bit 音频。

```json
{
    "asr": {
        "type": "RemoteASR",
        "params": {"base_url": "http://127.0.0.1:9000"}
    }
}
```

### 离线 ASR

只能识别完整语句的服务只需实现 `async_recognize` 和 `recognize`；继承的 `recognize_stream` 与 `async_recognize_stream` 使用 [`src/xtalk/models/audio_utils.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/audio_utils.py) 中的 `MockStreamRecognizer`，以 10 个块为窗口在 `async_recognize` 之上构建伪流式行为。

```python
import asyncio

import aiohttp

from xtalk import model
from xtalk.model_types import ASR


@model
class RemoteOfflineASR(ASR):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    async def async_recognize(self, audio: bytes) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/recognize", data=audio
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        return payload["text"]

    def recognize(self, audio: bytes) -> str:
        return asyncio.run(self.async_recognize(audio))

    def reset(self) -> None:
        return

    def clone(self) -> "RemoteOfflineASR":
        return RemoteOfflineASR(base_url=self.base_url)
```

## TTS

服务层调用 `async_synthesize_stream`，音频块一到达就播放。本实现以流式方式读取 HTTP 响应体，其余方法都由这个生成器派生。

```python
import asyncio
from typing import Any, AsyncIterator, Iterable

import aiohttp

from xtalk import model
from xtalk.model_types import TTS


@model
class RemoteTTS(TTS):
    def __init__(
        self, base_url: str, voice: str = "default", chunk_bytes: int = 9600
    ) -> None:
        self.base_url = base_url
        self._voice = voice
        self._chunk_bytes = chunk_bytes

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/synthesize",
                json={"text": text, "voice": self._voice},
            ) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_chunked(self._chunk_bytes):
                    yield chunk

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        chunks: list[bytes] = []
        async for chunk in self.async_synthesize_stream(text, **kwargs):
            chunks.append(chunk)
        return b"".join(chunks)

    def synthesize(self, text: str) -> bytes:
        return asyncio.run(self.async_synthesize(text))

    def synthesize_stream(self, text: str, **kwargs: Any) -> Iterable[bytes]:
        loop = asyncio.new_event_loop()
        try:
            stream = self.async_synthesize_stream(text, **kwargs)
            while True:
                try:
                    yield loop.run_until_complete(stream.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def set_voice(self, voice_names: list[str]) -> None:
        if voice_names:
            self._voice = voice_names[0]

    def clone(self) -> "RemoteTTS":
        return RemoteTTS(
            base_url=self.base_url, voice=self._voice, chunk_bytes=self._chunk_bytes
        )
```

同步的 `synthesize_stream` 在私有事件循环上逐块驱动异步生成器，同步迭代的调用方同样能增量收到音频。底层是同步流式客户端时，直接实现 `synthesize_stream` 和 `synthesize`，保留继承的 `async_*` 包装即可。

```json
{
    "tts": {
        "type": "RemoteTTS",
        "params": {"base_url": "http://127.0.0.1:9001", "voice": "default"}
    }
}
```

## VAD

服务层逐帧调用 `async_is_speech`，或以批量方式调用 `async_detect_frames`；继承的批量方法会顺序调用 `async_is_speech`。`set_threshold` 是抽象方法，因此类中保存阈值并随每次请求发送。

```python
import asyncio

import aiohttp

from xtalk import model
from xtalk.model_types import VAD


@model
class RemoteVAD(VAD):
    def __init__(self, base_url: str, threshold: float = 0.5) -> None:
        self.base_url = base_url
        self._threshold = threshold

    async def async_is_speech(self, frame: bytes) -> bool:
        if not frame:
            return False
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/is_speech",
                data=frame,
                params={"threshold": str(self._threshold)},
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        return bool(payload["speech"])

    def is_speech(self, frame: bytes) -> bool:
        return asyncio.run(self.async_is_speech(frame))

    def set_threshold(self, *, threshold: float) -> None:
        self._threshold = threshold

    def reset(self) -> None:
        return

    def clone(self) -> "RemoteVAD":
        return RemoteVAD(base_url=self.base_url, threshold=self._threshold)
```

服务拒绝阈值更新时，保留 `set_threshold` 并在其中抛出 `NotImplementedError`；Realtime 网关随后以 `error` 事件回应客户端的 `session.update`。

```json
{
    "vad": {
        "type": "RemoteVAD",
        "params": {"base_url": "http://127.0.0.1:9002", "threshold": 0.5}
    }
}
```

## Turn Detector

`TurnDetector.__init__` 创建 `listening` 相关的锁，因此子类要调用 `super().__init__()`。服务层以不同的参数组合调用 `async_detect`：仅携带上下文的调用（单独的 `assistant_text`、`speech_start` 或 `assistant_interrupted`）只更新状态并返回 `DO_NOTHING`；携带 `text` 和 `speech_pause` 的调用向服务询问用户是否说完。

```python
import asyncio
from typing import Optional

import aiohttp

from xtalk import model
from xtalk.model_types import TurnDetector
from xtalk.models.turn_detector.interfaces import (
    TurnDetectionAction,
    TurnDetectionResult,
    TurnDetectionSemantic,
)


@model
class RemoteTurnDetector(TurnDetector):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self._assistant_text = ""

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
        assistant_interrupted: bool = False,
    ) -> TurnDetectionResult:
        if assistant_text is not None:
            self._assistant_text = assistant_text
        if not text or not speech_pause:
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.IDLE,
            )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/turn",
                json={"text": text, "assistant_text": self._assistant_text},
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        if payload["complete"]:
            return TurnDetectionResult(
                action=TurnDetectionAction.START_GENERATION,
                semantic=TurnDetectionSemantic.COMPLETE,
            )
        return TurnDetectionResult(
            action=TurnDetectionAction.DO_NOTHING,
            semantic=TurnDetectionSemantic.INCOMPLETE,
        )

    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
        assistant_interrupted: bool = False,
    ) -> TurnDetectionResult:
        return asyncio.run(
            self.async_detect(
                audio=audio,
                text=text,
                assistant_text=assistant_text,
                speech_start=speech_start,
                speech_pause=speech_pause,
                assistant_interrupted=assistant_interrupted,
            )
        )

    def clone(self) -> "RemoteTurnDetector":
        return RemoteTurnDetector(base_url=self.base_url)
```

`assistant_text` 是累计前缀，因此保存时覆盖旧值而非追加。各 `TurnDetectionAction` 与 `TurnDetectionSemantic` 取值的含义见 [Turn Detector 接口](../reference/models/turn_detector.zh.md)。

```json
{
    "turn_detector": {
        "type": "RemoteTurnDetector",
        "params": {"base_url": "http://127.0.0.1:9003"}
    }
}
```

## 语音增强

服务层调用 `async_enhance` 时传入麦克风帧和等长的远端参考信号，输入结束时调用 `async_flush`。本实现把两段缓冲区放在一个请求中转发；服务逐帧应答，因此 `flush` 不返回尾部音频。

```python
import asyncio

import aiohttp

from xtalk import model
from xtalk.model_types import SpeechEnhancer


@model
class RemoteSpeechEnhancer(SpeechEnhancer):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    async def async_enhance(self, audio: bytes, far: bytes) -> bytes:
        if not audio:
            return b""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/enhance",
                data=aiohttp.FormData(
                    {"audio": audio, "far": far}, quote_fields=False
                ),
            ) as response:
                response.raise_for_status()
                return await response.read()

    def enhance(self, audio: bytes, far: bytes) -> bytes:
        return asyncio.run(self.async_enhance(audio, far))

    def flush(self) -> bytes:
        return b""

    async def async_flush(self) -> bytes:
        return b""

    def reset(self) -> None:
        return

    def clone(self) -> "RemoteSpeechEnhancer":
        return RemoteSpeechEnhancer(base_url=self.base_url)
```

`far` 的字节长度始终与 `audio` 相同；没有播放时服务层用静音填充。内部缓冲音频的服务应在 `async_flush` 中返回剩余采样，并在 `reset` 中清空缓冲。

```json
{
    "speech_enhancer": {
        "type": "RemoteSpeechEnhancer",
        "params": {"base_url": "http://127.0.0.1:9004"}
    }
}
```

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[注册并选用模型 →](../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[ASR 接口 →](../reference/models/asr.zh.md)** · Reference

-   **[模型对象的 `clone()` 与 `reset()` →](../reference/model_clone_reset.zh.md)** · Reference

</div>
