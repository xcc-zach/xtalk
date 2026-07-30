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

A forced aligner calibrates the played-text position and must not block audio. Audio preparation and paced delivery are separate: the producer applies speed processing, enqueues the final PCM for delivery, and publishes `TTSTextSynthesized` with the complete final PCM as soon as synthesis finishes. The playback manager starts `async_align` immediately, while the sender independently drains the queued PCM under playback backpressure and publishes `TTSTextDeliveryFinished` after the segment has been delivered.

No `segment_id` is introduced. Preparation boundaries, delivery boundaries, audio chunks, and `TTSChunkPlayed` events remain strictly FIFO within each session, and audio from different segments cannot be interleaved. The playback manager pairs preparation and delivery through two FIFO queues and validates that their text matches. Each prepared segment starts an independent forced-alignment task immediately; there is no per-session alignment concurrency limit, and out-of-order results remain attached to the segment object captured by each task.

### TTS: Two Calibration Levels

Regular TTS knows the complete sentence before synthesis starts. The paced sender publishes `TTSTextSynthesisStarted` when delivery begins and `TTSTextDeliveryFinished` after its sentence-end marker. `TTSTextSynthesized` comes directly from the producer and may arrive before or after delivery starts; strict FIFO pairing binds both event streams to the same playback segment.

| Stage | Audio playback | Active calibration |
| --- | --- | --- |
| Sentence synthesis is still in progress | Starts immediately and does not wait for alignment | **L1 rough calibration** using a conservative online estimate |
| Final PCM is available and forced alignment is running | Continues without interruption | **L1 rough calibration** using `played_ms / total_audio_ms` |
| Forced alignment is ready | Remains unaffected | **L2 precise calibration** using character or word timestamps |
| Playback completes before forced alignment | Completes normally | Confirm the complete sentence and cancel its unfinished alignment task |

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

While StreamingTextTTS is active, neither the final text nor the total audio duration is known, but audio chunks still play immediately. Each session keeps one implicit stream state and accumulates text only after `append_text` accepts it. Its audio reader prepares speed-processed PCM independently from paced delivery; when the upstream audio stream ends, it publishes `TTSTextSynthesized` with the complete PCM, switches to the total-duration ratio, and starts forced alignment while queued audio continues to play.

| Level | Used while | Required information |
| --- | --- | --- |
| **L1 online estimate** | The text and audio streams are still being generated | Accepted text, prepared-audio duration, and confirmed played-audio duration |
| **L2 total-duration ratio** | The complete audio stream is available and forced alignment is running | Complete accepted text, final processed PCM, and actual total audio duration |
| **L3 precise calibration** | Forced alignment has returned usable units | Character- or word-level timestamps mapped onto the complete text |

The actual total duration must be calculated from the PCM sent to the frontend, not by simply dividing the original duration by playback speed.

### Switching and Failure Handling

`ResponseUpdate` remains monotonic. If precise alignment is ahead, advance immediately. If it is behind the displayed position, do not move backwards; pause progress until the precise timeline catches up.

If alignment fails, continue using the rough result. Complete playback does not need to wait for alignment. On early interruption, prefer an available precise result, briefly wait for an in-flight alignment task, and fall back to the rough prefix on timeout. StreamingTextTTS interrupted before its complete audio is generated normally has only the online estimate unless the service exposes how much input text it has consumed.

### Debugging Frontend Text Updates

Set `XTALK_LOG_LEVEL='xtalk.serving.modules.tts_playback_manager=DEBUG'` to inspect the decision behind every frontend-facing `ResponseUpdate`. Each update is emitted as one log record, for example:

```text
TTS response update - session: ..., source: regular:L2-precise, mode: regular, level: L2-precise, state: ready, played_ms: 820.0, total_ms: 1600.0, delta: 'text added by this update', text: 'complete displayed prefix'
```

`source` is `playback-complete` when completing a played segment confirms its remaining text, or `<mode>:<level>` when the active calibration level advances the prefix. An update that crosses a segment boundary can contain both sources joined by `+`. The defensive `final-commit` source is used only when a final fallback update cannot be attributed to either case. `delta` and `text` use escaped string representations, so embedded line breaks remain within one physical log line. These fields can contain conversation content and should only be enabled when needed for debugging.
