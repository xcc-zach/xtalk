# Speech Enhancer Pipeline

Echo cancellation needs to know what the speaker is playing while the microphone records. X-Talk keeps that knowledge in the service layer: `EnhancerManager` builds the far-end reference and hands every enhancer an `audio` / `far` pair of equal length, so model implementations never reason about TTS timing themselves.

## How the service layer builds the far-end reference

`EnhancerManager` consumes `AudioFrameReceived` and publishes `EnhancedAudioFrameReceived`. With a `SpeechEnhancer` configured, each user audio frame is processed as:

```python
far = far_reference.take(len(audio))
enhanced = await enhancer.async_enhance(audio, far=far)
```

The reference comes from `TTSChunkReady`. The service layer copies each TTS chunk about to be sent to the client, converts it to the enhancer format, and writes it into a reference buffer:

- input TTS chunks are treated as PCM 16-bit mono
- audio is resampled from `TTSChunkReady.sample_rate` to 16000 Hz
- the converted audio is stored in a bounded FIFO buffer
- the buffer keeps up to 5 seconds by default, configurable with `far_reference_buffer_seconds`

When user audio arrives, the service layer reads a same-length slice from the buffer. With nothing playing, or when the buffer runs short, it returns or pads silence so `far` always matches `audio` in length. The pipeline owns this invariant, which is why enhancers may assume it.

## Role of `TTSChunkPlayed`

`TTSChunkPlayed` confirms that the frontend finished playing a TTS chunk. The event carries no `chunk_id`, client playback timestamp, or sample offset, so sample-level alignment is out of reach. The service layer uses it to discard reference audio that was played but never consumed by microphone frames. Without this, a user who waits for playback to finish before speaking would be enhanced against stale TTS audio. Finer alignment depends on future client-side playback feedback with timestamps.

## FastEnhancer behavior

`FastEnhancer` implements `SpeechEnhancer` and ignores `far` in both local ONNX mode and remote WebSocket mode; remote FastEnhancer keeps its legacy binary PCM protocol. In remote mode, `flush` sends locally pending padded audio and then issues the remote flush command.

`PyWebRTCAudio` adapts the pywebrtc-audio service. Its constructor accepts only `base_url`, and it forwards the upstream `audio` and `far` to the service's `/v1/stream` JSON WebSocket endpoint.

## Implementation suggestions

- Keep output length aligned with input `audio`; downstream VAD and ASR timelines depend on it.
- Leave TTS alignment to the service layer; length padding and far selection belong there.
- An implementation that uses `far` may rely on the length invariant; one that ignores `far` still succeeds on a silent reference.
- Remote implementations clear pending audio on `reset` and on connection rebuilds.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Speech enhancer interface →](../reference/models/speech_enhancer.md)** · Reference

-   **[Register a new model →](../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[TTS Pipeline →](tts.md)** · Explanation

</div>
