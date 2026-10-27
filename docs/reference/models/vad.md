# VAD Interface

```python
class VAD(ABC):
    """Abstract base class for voice activity detection engines."""

    @abstractmethod
    def set_threshold(self, *, threshold: float) -> None: ...

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool: ...

    async def async_is_speech(self, frame: bytes) -> bool: ...
```

`VAD` classifies one audio frame as speech or non-speech. The service layer calls `async_is_speech` in single-frame mode and `async_detect_frames` in batch mode; the latter calls `async_is_speech` sequentially by default. Turn-level speech start and end are derived by `VADManager` from consecutive frame results.

## `async_is_speech`

### Parameters

- `frame`: the current input audio frame, PCM 16-bit, mono, 16 kHz bytes.

### Returns

- `bool`: `True` when the frame is speech, `False` otherwise. An implementation that keeps internal context still returns the judgment for the latest complete frame.

## `set_threshold`

Update the speech probability cutoff for frame-level decisions.

### Parameters

- `threshold`: a probability cutoff in `[0, 1]`.

### Raises

- `NotImplementedError`: raised explicitly by models that cannot update the threshold. The method is abstract, so every subclass implements it.

Clients drive this method through the Realtime `session.update` event; see [Session settings](../realtime_api.md#session-settings).

## `clone` and `reset`

`VAD` implements `clone`. See [Model `clone()` and `reset()`](../model_clone_reset.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[How frame results become turn events →](../../explanation/vad.md)** · Explanation

-   **[Register a new model →](../../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Wrap an async detector in `is_speech` →](../../how_to/adapt_async_models.md)** · How-to

</div>
