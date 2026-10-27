# Forced Aligner Interface

```python
@dataclass(frozen=True)
class ForcedAlignmentUnit:
    """One text unit aligned onto a synthesized audio timeline."""

    text: str
    start_ms: float
    end_ms: float
    char_start: int = -1
    char_end: int = -1


@model_type
class ForcedAligner(ABC):
    """Abstract interface for forced alignment models."""

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

`ForcedAligner` maps text units onto the timeline of synthesized audio. Configuring `forced_aligner` enables playback calibration; no separate `enabled` switch exists. The service layer calls `async_align`; the default implementation runs `align` in a thread pool, and implementations with native asynchronous clients override it.

## `ForcedAlignmentUnit`

- `text`: the current character or word.
- `start_ms`, `end_ms`: millisecond timestamps relative to the start of the input audio.
- `char_start`, `char_end`: a half-open character range in the original text. A model without character offsets leaves both at `-1`; the playback manager remaps the unit from `text`.

## `align`

### Parameters

- `audio`: raw 48 kHz, mono, signed 16-bit little-endian PCM without a WAV header.
- `text`: the text spoken in `audio`.
- `language`: optional model-specific language hint.

### Returns

- `list[ForcedAlignmentUnit]`: units ordered by time, with timestamps in milliseconds. Inference may run locally or call a remote vLLM service, as `Qwen3ForcedAligner` does.

## `clone`

Return an independently usable instance for a new session. Model weights and connection configuration may be shared; mutable request state stays within one session. `ForcedAligner` has no `reset`; each `align` call is independent. See [Model `clone()` and `reset()`](../model_clone_reset.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Play first, calibrate later →](../../explanation/forced_aligner.md)** · Explanation

-   **[Register a new model →](../../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Available forced aligners →](../supported_models.md)** · Reference

</div>
