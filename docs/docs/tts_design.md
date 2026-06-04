```python
class TTS(ABC):
    """Abstract base class for text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        ...

    def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]:
        yield self.synthesize(text)

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        ...

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        ...

    @abstractmethod
    def clone(self) -> "TTS":
        ...

    def set_voice(self, voice_names: list[str]) -> None:
        ...

    def set_emotion(self, emotion: str | list[float]) -> None:
        ...
```

## Best Practice for Implementing `synthesize`

`synthesize` is the baseline API that every TTS implementation must provide.

The framework actually prefers `async_synthesize_stream`. Therefore, the best practice when implementing a new TTS is to implement `async_synthesize_stream` first, and then implement `synthesize` like this:

```python
import asyncio

def synthesize(self, text: str) -> bytes:
    return self._run_coro(self._collect_stream(text))

async def _collect_stream(self, text: str) -> bytes:
    chunks: list[bytes] = []
    async for chunk in self.async_synthesize_stream(text):
        chunks.append(chunk)
    return b"".join(chunks)

def _run_coro(self, coro: "asyncio.Future[bytes]") -> bytes:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

If the underlying implementation is already synchronous streaming, you may also implement `synthesize_stream` first and then reuse the default `async_synthesize_stream` wrapper provided by the base class.

## Parameters and Return Values of `synthesize` and `synthesize_stream`

### Input Parameters

- `text`: The text segment to synthesize. It is usually one complete sentence, but it may also be the final residual text flushed by the service layer.
- `**kwargs`: Model-specific extension parameters. The framework currently does not pass extra arguments from `TTSManager`, but implementations may still keep this extension point.

### Return Values

- `synthesize`: returns the complete audio as `bytes`
- `synthesize_stream` / `async_synthesize_stream`: yields audio `bytes` chunk by chunk

The expected audio format is:

- PCM 16-bit
- mono
- 48000 Hz

## How the Service Layer Consumes TTS Output

`TTSManager` does not send the whole LLM response to TTS at once. It first buffers text, splits it into sentences, and then calls TTS sentence by sentence.

At this layer, there are two semantics closely related to model implementations:

- The chunks returned by the model are "synthesis-side chunks", not the same as the final chunks sent to the frontend.
- The service layer will split audio again into fixed `TTSChunkReady` pieces of about 100 ms before sending them out.

Therefore:

- The model does not need to align its chunks with the frontend transport granularity.
- It is enough to ensure that the output PCM audio is continuous and in the correct order.
- Even if the model naturally emits very large chunks, it will not break the frontend playback protocol, because the service layer will split them again.

## `set_voice` and `set_emotion` (Experimental Interfaces)

These two methods are optional control interfaces invoked by `TTSManager` through events:

- `set_voice(voice_names)`: switch the current voice
- `set_emotion(emotion)`: switch the current emotion

Specifically:

- `voice_names` usually contains only one voice name for now, but the interface remains `list[str]` to support future multi-reference voice use cases.
- `emotion` may be either a string label or a model-specific vector representation.

Speed adjustment is not part of the `TTS` interface itself. In the current repository, speed control is handled separately by the service-layer speed controller after TTS audio is produced.

## `clone`

See [Semantics of `clone()` and `reset()` on Model Objects](model_clone_reset.md).
