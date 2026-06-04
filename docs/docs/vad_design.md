```python
class VAD(ABC):
    """Abstract base class for voice activity detection engines."""

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        ...

    async def async_is_speech(self, frame: bytes) -> bool:
        ...
```

## Best Practice for Implementing `is_speech`

The framework actually calls `async_is_speech`. Therefore, if the underlying implementation is asynchronous, the best practice is to implement `async_is_speech` first, and then implement `is_speech` like this:

```python
import asyncio

def is_speech(self, frame: bytes) -> bool:
    return self._run_coro(self.async_is_speech(frame))

def _run_coro(self, coro: "asyncio.Future[bool]") -> bool:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

## Parameters and Return Value of `async_is_speech`

Each call should return a boolean indicating whether the current audio frame contains speech.

`frame` is the current input audio frame in PCM 16-bit, mono, 16 kHz bytes. In the returned `bool`, `True` means the current frame is speech, and `False` means the current frame is non-speech.

Notes:

- The current `VADManager` in the service layer further smooths frame-level boolean results into speech start / speech end events.
- If the implementation maintains internal context state, the return value should still represent the speech judgment of the latest complete frame.

## How the Service Layer Consumes VAD Output

`VADManager` consumes `EnhancedAudioFrameReceived` events and, when backend VAD is enabled, performs the following steps:

- buffer the input audio and split it into fixed-length frames
- call `async_is_speech` once for each frame
- advance the state machine based on consecutive speech frames and consecutive silence frames
- publish `VADSpeechStart` / `VADSpeechEnd` when thresholds are met

Current default parameters:

- `vad_sample_rate = 16000`
- `vad_frame_samples = 512`
- approximately `32 ms` per frame
- `vad_min_speech_ms = 250`
- `vad_redemption_ms = 500`

Specifically:

- when consecutive speech frames accumulate beyond `vad_min_speech_ms`, the service layer considers the user to have started speaking
- when consecutive silence frames accumulate beyond `vad_redemption_ms`, the service layer considers the user to have stopped speaking

In other words, the VAD model outputs the raw frame-level judgment, while turn-level start/end semantics are produced by `VADManager`.

## Relationship Between Frontend VAD and Backend VAD

X-Talk supports frontend VAD. Backend `VAD` is mainly for cases where the frontend cannot run VAD, or when you explicitly want VAD to run on the server side.

It is generally not recommended to enable both frontend and backend VAD at the same time, otherwise duplicate turn events may be produced.

## `clone` and `reset`

VAD is expected to implement `clone`.

See [Semantics of `clone()` and `reset()` on Model Objects](model_clone_reset.md).
