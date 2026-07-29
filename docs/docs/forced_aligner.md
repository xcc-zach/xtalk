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
    ) -> list[ForcedAlignmentUnit]:
        """Align text units against 48 kHz PCM audio."""
        pass

    async def async_align(
        self,
        *,
        audio: bytes,
        text: str,
        language: str | None = None,
    ) -> list[ForcedAlignmentUnit]:
        """Asynchronously align text units."""
        ...

    @abstractmethod
    def clone(self) -> "ForcedAligner":
        """Clone the aligner for a new service session."""
        pass
```

## `ForcedAlignmentUnit`

`text` is the current character or word. `start_ms` and `end_ms` are millisecond timestamps relative to the start of the input audio.

`char_start` and `char_end` define a half-open character range in the original text. A model that cannot provide character offsets may leave both at `-1`; the playback manager will remap the unit from `text`.

## `align`

`audio` must contain raw 48 kHz, mono, signed 16-bit little-endian PCM without a WAV header. `text` should match what the audio speaks, while `language` is an optional model-specific hint.

Results must be ordered by time and use milliseconds rather than samples or seconds. An implementation may run inference locally or call a remote vLLM service like `Qwen3ForcedAligner`.

## `async_align`

The framework uses `async_align` to avoid blocking the event loop. Its default implementation calls synchronous `align` in a thread pool; implementations with native asynchronous clients may override it.

## `clone`

`clone` returns an independently usable instance for a new session. Immutable resources such as model weights and connection configuration may be shared, but mutable request state must not cross sessions. A forced aligner has no `reset` method, and each `align` call should be independent.

Configuring `forced_aligner` automatically enables playback calibration; no separate `enabled` switch is required.

## Play First, Calibrate Later

A forced aligner calibrates the played-text position and must not block audio. The service should immediately send the final, speed-processed PCM while collecting the same audio in the background, then call `async_align` once the complete audio is available.

No `segment_id` is introduced. Text boundaries, audio chunks, and `TTSChunkPlayed` events must therefore remain strictly FIFO within each session, and audio from different sentences must not be interleaved.

### TTS: Two Calibration Levels

Regular TTS knows the complete sentence before synthesis starts. Its internal queue places a sentence-start marker, audio chunks, and a sentence-end marker in order so the playback manager can determine boundaries.

1. **Rough calibration**: while audio is being generated, estimate the prefix from the complete text, generated-audio duration, and played-audio duration. Once generation finishes, replace the estimated duration with the actual total duration and continue using the playback ratio.
2. **Precise calibration**: after the complete sentence audio is available, run the forced aligner in the background and switch to character or word timestamps when it returns.

The open-audio estimate can be conservative:

```text
safe_played_ms = max(0, played_ms - 200)
estimated_total_ms = max(
    text_weight × estimated_ms_per_unit,
    generated_audio_ms + 300,
)
```

Count Chinese characters and English words as text units. Attach punctuation to adjacent units and give it a small pause weight. After the sentence ends, use `played_ms / total_audio_ms` for the rough ratio.

### StreamingTextTTS: Three Calibration Levels

While StreamingTextTTS is active, neither the final text nor the total audio duration is known, but audio chunks should still play immediately. Each session keeps one implicit stream state and accumulates text only after `append_text` accepts it.

1. **Online estimate**: estimate from the currently accepted text, generated-audio duration, and played-audio duration using the conservative formula above.
2. **Total-duration ratio**: after the complete audio stream ends, use the complete text and actual total audio duration to calculate `played_ms / total_audio_ms`.
3. **Precise calibration**: request forced alignment in the background at the same time, then switch to character or word timestamps.

The actual total duration must be calculated from the PCM sent to the frontend, not by simply dividing the original duration by playback speed.

### Switching and Failure Handling

`ResponseUpdate` remains monotonic. If precise alignment is ahead, advance immediately. If it is behind the displayed position, do not move backwards; pause progress until the precise timeline catches up.

If alignment fails, continue using the rough result. Complete playback does not need to wait for alignment. On early interruption, prefer an available precise result, briefly wait for an in-flight alignment task, and fall back to the rough prefix on timeout. StreamingTextTTS interrupted before its complete audio is generated normally has only the online estimate unless the service exposes how much input text it has consumed.
