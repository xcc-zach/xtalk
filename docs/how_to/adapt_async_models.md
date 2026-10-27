# Adapt Async Models

Implement a model whose underlying client is asynchronous or remote by writing the `async_*` method first and wrapping it in the synchronous abstract method; this assumes you know how `@model` registration and config selection work from [Introduce a New Model](../tutorial/introduce_a_new_model.md).

The service layer calls the `async_*` method of every model type. Each base class implements `async_*` by running the synchronous method in a thread pool, so a synchronous client only needs the synchronous method. An asynchronous client goes the other way: implement `async_*`, and have the synchronous method run it with `asyncio.run()`.

Every class below is complete: it registers with `@model`, implements all abstract methods, and can be selected from the configuration file by class name. The examples talk to a placeholder HTTP service at `base_url` with `aiohttp`; keep the method structure and replace the request code with your own client. Each call opens its own `aiohttp.ClientSession`, because `asyncio.run()` creates a fresh event loop per synchronous call and a session bound to a closed loop cannot be reused.

## ASR

The service layer calls `async_recognize_stream` with the audio that arrived since the previous call and expects the full text of the turn so far. This implementation keeps the turn audio in memory and sends the whole buffer to the service on every call.

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

`stream_chunk_bytes_hint` tells the service layer how many bytes to collect before each streaming call; `6400` bytes is 200 ms of 16 kHz PCM 16-bit audio.

```json
{
    "asr": {
        "type": "RemoteASR",
        "params": {"base_url": "http://127.0.0.1:9000"}
    }
}
```

### Offline ASR

A service that only recognizes complete utterances needs `async_recognize` and `recognize`; the inherited `recognize_stream` and `async_recognize_stream` build pseudo-streaming on top of `async_recognize` with `MockStreamRecognizer` from [`src/xtalk/models/audio_utils.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/audio_utils.py), using a window of 10 chunks.

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

The service layer calls `async_synthesize_stream` and plays chunks as they arrive. This implementation streams the HTTP response body and derives every other method from that generator.

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

The synchronous `synthesize_stream` drives the async generator one chunk at a time on a private loop, so callers that iterate synchronously still receive audio incrementally. With a synchronous streaming client, implement `synthesize_stream` and `synthesize` directly and keep the inherited `async_*` wrappers instead.

```json
{
    "tts": {
        "type": "RemoteTTS",
        "params": {"base_url": "http://127.0.0.1:9001", "voice": "default"}
    }
}
```

## VAD

The service layer calls `async_is_speech` once per frame, or `async_detect_frames` with a batch; the inherited batch method calls `async_is_speech` sequentially. `set_threshold` is abstract, so the class stores the threshold and sends it with each request.

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

A service that rejects threshold updates keeps `set_threshold` and raises `NotImplementedError` from it; the Realtime gateway then answers the client's `session.update` with an `error` event.

```json
{
    "vad": {
        "type": "RemoteVAD",
        "params": {"base_url": "http://127.0.0.1:9002", "threshold": 0.5}
    }
}
```

## Turn Detector

`TurnDetector.__init__` creates the `listening` locks, so the subclass calls `super().__init__()`. The service layer calls `async_detect` with different argument combinations: context-only calls (`assistant_text`, `speech_start` or `assistant_interrupted` alone) update state and return `DO_NOTHING`; calls carrying `text` and `speech_pause` ask the service whether the user has finished.

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

`assistant_text` is a cumulative prefix, so the stored value is replaced rather than appended. The meaning of each `TurnDetectionAction` and `TurnDetectionSemantic` value is listed in the [Turn Detector interface](../reference/models/turn_detector.md).

```json
{
    "turn_detector": {
        "type": "RemoteTurnDetector",
        "params": {"base_url": "http://127.0.0.1:9003"}
    }
}
```

## Speech Enhancer

The service layer calls `async_enhance` with the microphone frame and a same-length far-end reference, then `async_flush` at the end of input. This implementation forwards both buffers in one request; `flush` returns no tail audio because the service answers frame by frame.

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

`far` always has the same byte length as `audio`; the service layer pads it with silence when nothing is playing. A service that buffers audio internally returns the remaining samples from `async_flush` and clears its buffer in `reset`.

```json
{
    "speech_enhancer": {
        "type": "RemoteSpeechEnhancer",
        "params": {"base_url": "http://127.0.0.1:9004"}
    }
}
```

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Register and select the model →](../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[ASR interface →](../reference/models/asr.md)** · Reference

-   **[Model `clone()` and `reset()` →](../reference/model_clone_reset.md)** · Reference

</div>
