# http TTS

This TTS interface is used when input text is not streaming.

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

# websockets TTS

This TTS interface is used when input text arrives incrementally.

```python
class StreamingTextTTS(ABC):
    """Abstract base class for live text-streaming TTS engines."""

    @abstractmethod
    async def start(self) -> None:
        """Start one live TTS session."""
        ...

    @abstractmethod
    async def append_text(self, text: str) -> None:
        """Append incremental text to the active TTS session."""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Request synthesis of text that has been received but not emitted."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the active TTS session and release connection resources."""
        ...

    @abstractmethod
    def audio_stream(self) -> AsyncIterator[bytes]:
        """Yield PCM audio chunks as soon as the model generates audio."""
        ...

    @abstractmethod
    def clone(self) -> "StreamingTextTTS":
        """Clone an independent streaming TTS instance for a new service session."""
        ...
```

`StreamingTextTTS` is a capability interface separate from `TTS`. Regular
non-live TTS implementations only need to implement `TTS`; models that support
incremental text input may inherit both `TTS` and `StreamingTextTTS`.

## Method Semantics

- `start()`: starts an upstream live TTS session, such as opening a WebSocket
  connection and sending the provider-specific start event.
- `append_text(text)`: called immediately when an LLM text chunk arrives. This
  method sends text upstream and does not wait for a complete sentence.
- `flush()`: called only when the service receives `TurnTTSFlushRequested`.
  The current design does not automatically flush at complete-sentence
  boundaries.
- `stop()`: ends the upstream TTS session and releases connection resources.
  `stop()` does not implicitly flush; `TTSManager` should explicitly call
  `flush()` first when residual text should be synthesized.
- `audio_stream()`: yields audio chunks as soon as the model generates them.
  `TTSManager` wraps those chunks as `TTSChunkReady` events.
- `clone()`: clones an instance for a new service session. The cloned instance
  must have independent upstream connection state, buffers, and background task
  state, and must not reuse a live TTS connection from another session.

## How the Service Layer Consumes StreamingTextTTS

When the active TTS model is a `StreamingTextTTS`, `TTSManager` does not use the
regular `pending_sentence_buffer` and sentence-by-sentence synthesis path. It
uses this event flow instead:

```text
TurnTTSStartRequested
  -> StreamingTextTTS.start()
  -> start a background audio_stream reader task

TurnTTSTextAppendRequested(text)
  -> StreamingTextTTS.append_text(text)

TurnTTSFlushRequested
  -> StreamingTextTTS.flush()
  -> StreamingTextTTS.stop()

TurnTTSStopRequested / shutdown
  -> StreamingTextTTS.stop()
```

The background `audio_stream` reader should publish existing events as soon as
model audio arrives:

```text
StreamingTextTTS.audio_stream() yields PCM
  -> TTSManager splits it into about 100 ms chunks
  -> publish TTSChunkReady(audio_chunk=chunk, sample_rate=...)
  -> OutputGateway sends it to the frontend
```

The core goal of `StreamingTextTTS` is that text is sent upstream via
`append_text` as soon as it arrives, and generated TTS audio is immediately
wrapped by `TTSManager` as `TTSChunkReady`.

## Audio Format

`audio_stream()` should yield continuous, ordered PCM audio. The recommended
output format remains the same as regular `TTS`:

- PCM 16-bit
- mono
- 48000 Hz

If the upstream WebSocket TTS only supports another sample rate, such as Fish
Audio PCM output at 44100 Hz but not 48000 Hz, the model implementation should
resample internally to 48000 Hz before yielding from `audio_stream()`. This
keeps the existing frontend binary audio protocol unchanged.

# TTS Playback Progress Tracking

`TTSPlaybackManager` advances played-audio time from frontend `TTSChunkPlayed` events and publishes `ResponseUpdate`. Configuring `forced_aligner` further calibrates the current played-text position. See [Forced Aligner Design](forced_aligner.md) for the interface and the "play first, calibrate later" design.
