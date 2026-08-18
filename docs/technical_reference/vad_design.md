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

## Frontend VAD Logic

The web frontend uses Silero VAD v5 by default. Input audio defaults to 16 kHz, PCM 16-bit mono and is split into frames of 512 samples, or approximately 32 ms per frame. The current defaults are:

- `positiveSpeechThreshold = 0.1`
- `negativeSpeechThreshold = 0.02`
- `minSpeechMs = 250`
- `redemptionMs = 500`

The frontend applies two thresholds to the speech probability produced by the model:

- A probability greater than or equal to `0.1` classifies the current frame as speech. On the first matching frame, `FrameProcessor` immediately produces `SpeechStart`, after which the frontend sends `vad_speech_start`.
- A probability below `0.02` advances the speech-end redemption counter.
- A probability greater than or equal to `0.02` immediately resets the redemption counter. Consequently, only consecutive frames below `0.02` can produce `SpeechEnd`.
- A probability in `[0.02, 0.1)` does not start speech. If speech is already active, it prevents speech from ending.

With the default 16 kHz / 512-sample configuration, `FrameProcessor` converts 500 ms to 15 frames, so `SpeechEnd` requires approximately 480 ms of consecutive low-probability frames. The frontend then sends `vad_speech_end`.

`minSpeechMs` does not delay the `SpeechStart` consumed by the current integration. It is used for `FrameProcessor`'s `SpeechRealStart` and short-speech misfire handling, but the frontend does not currently forward either event.

When frontend VAD is enabled, microphone PCM frames are uploaded immediately without waiting for VAD inference. VAD runs serially on enhanced frames through a separate queue. If inference falls behind, that queue may drop old frames, but it does not block or discard microphone frames uploaded to ASR.

`inputConfig.vadRedemptionMs` can override the default `redemptionMs`. It only changes the speech-end waiting period and does not change either probability threshold.

## Relationship Between Frontend VAD and Backend VAD

X-Talk supports frontend VAD. Backend `VAD` is mainly for cases where the frontend cannot run VAD, or when you explicitly want VAD to run on the server side.

It is generally not recommended to enable both frontend and backend VAD at the same time, otherwise duplicate turn events may be produced.

## `clone` and `reset`

VAD is expected to implement `clone`.

See [Semantics of `clone()` and `reset()` on Model Objects](model_clone_reset.md).
