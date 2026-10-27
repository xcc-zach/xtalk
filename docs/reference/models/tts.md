# TTS Interface

X-Talk defines two TTS interfaces. `TTS` receives complete sentences; `StreamingTextTTS` receives incremental text over a live connection. A model that supports incremental text input inherits both.

## `TTS`

```python
class TTS(ABC):
    """Abstract base class for text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]:
        yield self.synthesize(text)

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes: ...

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]: ...

    @abstractmethod
    def clone(self) -> "TTS": ...

    def set_voice(self, voice_names: list[str]) -> None: ...

    def set_emotion(self, emotion: str | list[float]) -> None: ...
```

`synthesize` is the baseline that every implementation provides. The service layer calls `async_synthesize_stream`; the default implementation wraps `synthesize_stream`, which in turn yields the result of `synthesize`.

### Parameters

- `text`: the text segment to synthesize. It is usually one complete sentence; it may also be the final residual text flushed by the service layer.
- `**kwargs`: model-specific extension parameters. `TTSManager` passes no extra arguments; implementations may keep this extension point.

### Returns

- `synthesize` / `async_synthesize`: the complete audio as `bytes`.
- `synthesize_stream` / `async_synthesize_stream`: audio `bytes` chunk by chunk, continuous and in order. Chunk size is free; the service layer re-splits audio into about 100 ms pieces before sending.

Audio format: PCM 16-bit, mono, 48000 Hz.

### `set_voice` and `set_emotion`

Optional control methods, experimental, invoked by `TTSManager` through events.

- `set_voice(voice_names)`: switch the current voice. `voice_names` usually contains one name; the `list[str]` type reserves room for multi-reference voices.
- `set_emotion(emotion)`: switch the current emotion. `emotion` is a string label or a model-specific vector.

Speed control stays outside the `TTS` interface; the service-layer speed controller processes audio after synthesis.

## `StreamingTextTTS`

```python
class StreamingTextTTS(ABC):
    """Abstract base class for live text-streaming TTS engines."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def append_text(self, text: str) -> None: ...

    @abstractmethod
    async def flush(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def audio_stream(self) -> AsyncIterator[bytes]: ...

    @abstractmethod
    def clone(self) -> "StreamingTextTTS": ...
```

### Methods

- `start()`: start an upstream live TTS session, such as opening a WebSocket connection and sending the provider-specific start event.
- `append_text(text)`: send one LLM text chunk upstream as soon as it arrives, without waiting for a complete sentence.
- `flush()`: request synthesis of text received but not yet emitted. The service layer calls it only on `TurnTTSFlushRequested`; sentence boundaries do not trigger it.
- `stop()`: end the upstream session and release connection resources. `stop()` performs no flush; `TTSManager` calls `flush()` first when residual text should be synthesized.
- `audio_stream()`: yield PCM chunks as soon as the model generates them. `TTSManager` wraps them as `TTSChunkReady` events.
- `clone()`: return an instance with independent upstream connection state, buffers, and background task state. A clone never reuses a live connection from another session.

### Audio format

`audio_stream()` yields continuous, ordered PCM 16-bit, mono, 48000 Hz audio. When the upstream service only supports another sample rate, such as Fish Audio PCM at 44100 Hz, the implementation resamples to 48000 Hz before yielding.

## `clone`

See [Model `clone()` and `reset()`](../model_clone_reset.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[How the service layer consumes TTS audio →](../../explanation/tts.md)** · Explanation

-   **[Register a new model →](../../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Wrap an async synthesizer in `synthesize` →](../../how_to/adapt_async_models.md)** · How-to

-   **[Forced aligner interface →](forced_aligner.md)** · Reference

</div>
