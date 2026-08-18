# Speech Enhancer Design

```python
class SpeechEnhancer(ABC):
    """Abstract base class for speech enhancement engines."""

    @abstractmethod
    def enhance(self, audio: bytes, far: bytes) -> bytes:
        ...

    def flush(self) -> bytes:
        return b""

    async def async_enhance(
        self,
        audio: bytes,
        far: bytes,
    ) -> bytes:
        ...

    async def async_flush(self) -> bytes:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def clone(self) -> "SpeechEnhancer":
        ...
```

## Audio Format

`SpeechEnhancer` input and output use:

- PCM 16-bit
- Mono
- 16000 Hz
- Raw PCM bytes without a WAV header

`audio` is the near-end microphone signal. `far` is the far-end reference signal, usually derived from the TTS audio being played to the user, and is used for acoustic echo cancellation.

The upstream audio pipeline always provides `far` and guarantees that:

- `far` uses the same audio format as `audio`
- `len(far) == len(audio)`
- both buffers describe the near-end input and far-end reference for the same time window

## `enhance` and `async_enhance`

The service layer primarily calls `async_enhance`. If the underlying implementation only has a synchronous API, implement `enhance` and reuse the base class thread-pool wrapper for `async_enhance`.

If the underlying implementation is asynchronous or remote, prefer implementing `async_enhance` directly and wrapping it for `enhance`.

```python
import asyncio

def enhance(self, audio: bytes, far: bytes) -> bytes:
    return self._run_coro(self.async_enhance(audio, far=far))

def _run_coro(self, coro: "asyncio.Future[bytes]") -> bytes:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

## Meaning of `far`

`far` is a required interface argument and enables enhancement engines that need a far-end reference, such as acoustic echo cancellation.

- When nothing is being played, the service layer passes a silent `far` buffer with the same length as `audio`.
- When TTS is being played, the service layer reads a same-length slice from the TTS reference buffer.
- If the reference buffer is short, the service layer pads silence on the right and still preserves equal length.
- FastEnhancer accepts the interface argument but ignores `far` in both local and remote modes.
- A concrete enhancer implementation that targets an echo-cancellation service may use `far`.

Therefore, implementations may assume that `far` has the same byte length as `audio`. The upstream service pipeline owns this invariant, so concrete enhancers do not need to check it again.

## How the Service Layer Builds the Far-End Reference

`EnhancerManager` consumes `AudioFrameReceived` and publishes `EnhancedAudioFrameReceived`. When a `SpeechEnhancer` model is configured, each user audio frame is processed like this:

```python
far = far_reference.take(len(audio))
enhanced = await enhancer.async_enhance(audio, far=far)
```

The far-end reference comes from `TTSChunkReady`. The service layer copies the TTS chunk that is about to be sent to the client, converts it to the speech-enhancer format, and writes it into a reference buffer.

Current conversion rules:

- Treat input TTS chunks as PCM 16-bit mono
- Resample from `TTSChunkReady.sample_rate` to 16000 Hz
- Store the converted audio in a bounded FIFO buffer
- Keep up to 5 seconds by default, configurable with `far_reference_buffer_seconds`

When user audio arrives, the service layer reads a far-end reference buffer with the same byte length as the current `audio`. If no reference is available, it returns same-length silence.

## Role of `TTSChunkPlayed`

`TTSChunkPlayed` means the frontend has confirmed that a TTS chunk finished playback. The current event does not include a `chunk_id`, client playback timestamp, or sample offset, so it cannot provide sample-level alignment.

The service layer uses it to discard already played reference audio that microphone frames did not consume. This prevents stale TTS audio from being used as `far` when the user waits until TTS playback completes before speaking.

More precise alignment should come from future client-side playback reference feedback or timestamped playback events.

## FastEnhancer Behavior

`FastEnhancer` implements the `SpeechEnhancer` interface but does not use `far`.
This is true for both local ONNX mode and remote FastEnhancer WebSocket mode.
Remote FastEnhancer continues to use its legacy binary PCM protocol.

`PyWebRTCAudio` is the concrete adapter for the pywebrtc-audio service. Its
constructor only accepts `base_url`, and it sends the upstream-provided `audio`
and `far` to the service's `/v1/stream` JSON WebSocket endpoint.

## `flush`

`flush` / `async_flush` drains tail audio buffered inside the enhancer. The service layer inserts a flush barrier after `VADSpeechEnd` so all earlier audio frames are enhanced and dispatched downstream before the flush runs.

For remote FastEnhancer mode, `flush` sends any locally pending padded audio and then uses the remote FastEnhancer flush command.

## `reset` and `clone`

`reset` should clear the current session's streaming state, such as model caches, input/output buffers, remote connections, and pending audio. It should not reload model weights.

`clone` should create an independent runtime instance for a new session. Clones may share weights, configuration, or read-only resources, but must not share streaming buffers, remote connections, pending audio, or session state.

See [Semantics of `clone()` and `reset()` on Model Objects](model_clone_reset.md).

## Implementation Suggestions

- Keep output length aligned with input `audio` whenever possible, so downstream VAD and ASR timelines do not drift.
- Do not try to infer TTS alignment inside the model interface; length padding and far selection belong to the service layer.
- If the implementation requires `far`, it may assume the upstream pipeline already matched its length with `audio`.
- If the implementation does not support `far`, ignore the argument but do not fail on a silent reference.
- Remote implementations should clear pending audio on `reset` and connection rebuilds.
