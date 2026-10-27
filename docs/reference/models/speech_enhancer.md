# Speech Enhancer Interface

```python
class SpeechEnhancer(ABC):
    """Abstract base class for speech enhancement engines."""

    @abstractmethod
    def enhance(self, audio: bytes, far: bytes) -> bytes: ...

    def flush(self) -> bytes:
        return b""

    async def async_enhance(self, audio: bytes, far: bytes) -> bytes: ...

    async def async_flush(self) -> bytes: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def clone(self) -> "SpeechEnhancer": ...
```

`SpeechEnhancer` processes near-end microphone audio, optionally against a far-end reference for acoustic echo cancellation. The service layer calls `async_enhance`; the default implementation runs `enhance` in a thread pool.

## Audio format

Input and output use PCM 16-bit, mono, 16000 Hz raw bytes without a WAV header.

## `async_enhance`

### Parameters

- `audio`: the near-end microphone signal.
- `far`: the far-end reference signal, derived from the TTS audio being played to the user. The service layer guarantees `len(far) == len(audio)`, the same audio format, and the same time window; implementations skip this check. A silent buffer of the same length is passed when nothing is playing. Implementations that need no reference ignore the argument and still return output for a silent `far`.

### Returns

- `bytes`: enhanced audio. Keep the output length equal to the input `audio` so downstream VAD and ASR timelines stay aligned.

## `async_flush`

Drain tail audio buffered inside the enhancer. The service layer inserts a flush barrier after `VADSpeechEnd`, so all earlier frames are enhanced and dispatched before the flush runs.

### Returns

- `bytes`: remaining enhanced audio, or `b""`.

## `reset` and `clone`

`reset` clears the current session's streaming state: model caches, input/output buffers, remote connections, and pending audio. Weights stay loaded.

`clone` creates an independent runtime instance for a new session. Clones may share weights, configuration, and read-only resources; they never share streaming buffers, remote connections, pending audio, or session state.

See [Model `clone()` and `reset()`](../model_clone_reset.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[How the far-end reference is built →](../../explanation/speech_enhancer.md)** · Explanation

-   **[Register a new model →](../../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Wrap an async enhancer in `enhance` →](../../how_to/adapt_async_models.md)** · How-to

</div>
