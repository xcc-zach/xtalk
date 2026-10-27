# VAD Pipeline

A VAD model answers one question per frame: speech or silence. Turn-level meaning, the moments a user starts and stops speaking, is produced by the surrounding pipeline. X-Talk runs that pipeline in the browser by default and offers a backend equivalent for clients that cannot run VAD.

## How the service layer consumes VAD output

`VADManager` consumes `EnhancedAudioFrameReceived` events. When backend VAD is enabled, it buffers input audio and splits it into fixed-length frames, calls `async_is_speech` in single-frame mode or `async_detect_frames` in batch mode to obtain one boolean per frame in input order, advances a state machine on consecutive speech and silence frames, and publishes `VADSpeechStart` / `VADSpeechEnd` when thresholds are met.

Default parameters:

- `vad_sample_rate = 16000`
- `vad_frame_samples = 512`, approximately 32 ms per frame
- `vad_min_speech_ms = 250`: consecutive speech frames beyond this mark the user as speaking
- `vad_redemption_ms = 500`: consecutive silence frames beyond this mark the user as finished

The model therefore stays stateless with respect to turns; smoothing lives in `VADManager`, so every model implementation benefits from the same start/end behavior.

## Frontend VAD logic

The web frontend uses Silero VAD v5 by default on 16 kHz, PCM 16-bit mono audio split into 512-sample frames (about 32 ms). Defaults:

- `positiveSpeechThreshold = 0.1`
- `negativeSpeechThreshold = 0.02`
- `minSpeechMs = 250`
- `redemptionMs = 500`

Two thresholds are applied to the speech probability:

- A probability of `0.1` or above classifies the frame as speech. On the first matching frame, `FrameProcessor` immediately produces `SpeechStart`, and the frontend sends `vad_speech_start`.
- A probability below `0.02` advances the speech-end redemption counter.
- A probability of `0.02` or above resets the redemption counter, so only consecutive frames below `0.02` produce `SpeechEnd`.
- A probability in `[0.02, 0.1)` starts no speech; while speech is active, it prevents speech from ending.

With the default configuration, `FrameProcessor` converts 500 ms to 15 frames, so `SpeechEnd` requires about 480 ms of consecutive low-probability frames before the frontend sends `vad_speech_end`.

`minSpeechMs` leaves `SpeechStart` untouched. It drives `FrameProcessor`'s `SpeechRealStart` and short-speech misfire handling, and the frontend forwards neither event.

Microphone PCM frames upload immediately when frontend VAD is enabled; VAD inference runs serially on enhanced frames through a separate queue. If inference falls behind, that queue may drop old frames, while the frames uploaded to ASR continue unaffected.

`inputConfig.vadRedemptionMs` overrides the default `redemptionMs`. It changes only the speech-end waiting period; both probability thresholds stay fixed.

## Frontend VAD and backend VAD

Backend `VAD` exists for clients that cannot run VAD, and for deployments that deliberately place VAD on the server. Enabling both at the same time produces duplicate turn events, so pick one side per deployment.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[VAD interface →](../reference/models/vad.md)** · Reference

-   **[Register a new model →](../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Set the backend threshold through `session.update` →](../reference/realtime_api.md#session-settings)** · Reference

</div>
