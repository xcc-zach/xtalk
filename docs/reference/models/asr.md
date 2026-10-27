# ASR Interface

```python
class ASR(ABC):
    """Abstract interface for automatic speech recognition."""

    @abstractmethod
    def recognize(self, audio: bytes) -> str: ...

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str: ...

    def stream_chunk_bytes_hint(self) -> int | None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def clone(self) -> "ASR": ...

    async def async_recognize(self, audio: bytes) -> str: ...

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str: ...
```

`ASR` converts PCM 16-bit mono audio into text. The service layer calls `async_recognize_stream`; the default implementation runs `recognize_stream` in a thread pool, and the default `recognize_stream` adapts `async_recognize` into streaming calls with `MockStreamRecognizer`.

## `recognize`

Recognize a full audio buffer.

### Parameters

- `audio`: PCM 16-bit mono audio bytes.

### Returns

- `str`: recognized text.

## `async_recognize_stream`

Recognize incremental audio input.

### Parameters

- `audio`: the newly arrived audio for this call, PCM 16-bit mono bytes.
- `is_final`: whether the current point is an acoustic pause, such as the end of one sentence. It is a decoding hint; implementations may flush buffered tail audio, and the accumulated recognized text must be preserved so later audio continues from it. `reset` is the only call that clears the accumulated text.
- `chat_history`: experimental; serialized chat history for the current session, excluding the in-progress turn when unavailable.

### Returns

- `str`: the complete recognized text accumulated from the last `reset` up to the current moment.

## `stream_chunk_bytes_hint`

Return the preferred streaming chunk size.

### Returns

- `int | None`: the byte count of audio that should trigger one streaming recognition step; `None` means no preference, and the service layer triggers recognition frequently.

## `clone` and `reset`

`reset` clears the recognition state and the accumulated text. `clone` returns an instance with shared weights and independent runtime state. See [Model `clone()` and `reset()`](../model_clone_reset.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Wrap an async recognizer or an offline model →](../../how_to/adapt_async_models.md)** · How-to

-   **[Register a new model →](../../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Available ASR implementations →](../supported_models.md)** · Reference

</div>
