# Forced Alignment Pipeline

A forced aligner calibrates the played-text position shown to the user and never blocks audio. Audio starts playing as soon as it is synthesized; text progress begins with a rough estimate and is upgraded whenever more precise information arrives.

## Play first, calibrate later

Audio preparation and paced delivery are separate. The producer applies speed processing, enqueues the final PCM for delivery, and publishes `TTSTextSynthesized` with the complete final PCM as soon as synthesis finishes. `TTSPlaybackManager` starts `async_align` immediately, while the sender independently drains the queued PCM under playback backpressure and publishes `TTSTextDeliveryFinished` after the segment has been delivered.

No `segment_id` is introduced. Preparation boundaries, delivery boundaries, audio chunks, and `TTSChunkPlayed` events remain strictly FIFO within each session, and audio from different segments cannot interleave. The playback manager pairs preparation and delivery through two FIFO queues and validates that their text matches. Each prepared segment starts an independent forced-alignment task immediately; there is no per-session alignment concurrency limit, and out-of-order results remain attached to the segment object captured by each task.

## `TTS`: two calibration levels

Regular TTS knows the complete sentence before synthesis starts. The paced sender publishes `TTSTextSynthesisStarted` when delivery begins and `TTSTextDeliveryFinished` after its sentence-end marker. `TTSTextSynthesized` comes directly from the producer and may arrive before or after delivery starts; strict FIFO pairing binds both event streams to the same playback segment.

| Stage | Audio playback | Active calibration |
| --- | --- | --- |
| Sentence synthesis is still in progress | Starts immediately and does not wait for alignment | **L1 rough calibration** using a conservative online estimate |
| Final PCM is available and forced alignment is running | Continues without interruption | **L1 rough calibration** using `played_ms / total_audio_ms` |
| Forced alignment is ready | Remains unaffected | **L2 precise calibration** using character or word timestamps |
| Playback completes before forced alignment | Completes normally | Confirm the complete sentence and cancel its unfinished alignment task |

The open-audio estimate stays conservative:

```text
safe_played_ms = max(0, played_ms - 200)
estimated_total_ms = max(
    text_weight × estimated_ms_per_unit,
    generated_audio_ms + 300,
)
```

Chinese characters and English words count as text units. Punctuation attaches to adjacent units with a small pause weight. After the sentence ends, the rough ratio switches to `played_ms / total_audio_ms`.

## `StreamingTextTTS`: three calibration levels

While StreamingTextTTS is active, neither the final text nor the total audio duration is known, yet audio chunks still play immediately. Each session keeps one implicit stream state and accumulates text only after `append_text` accepts it. Its audio reader prepares speed-processed PCM independently from paced delivery; when the upstream audio stream ends, it publishes `TTSTextSynthesized` with the complete PCM, switches to the total-duration ratio, and starts forced alignment while queued audio continues to play.

Each accepted text increment carries the duration of speed-processed PCM already prepared before that increment is forwarded. L1 converts only the new increment into immutable timing units anchored no earlier than that prepared-audio watermark, the already emitted or played audio, and the previous unit's end. Accepting text alone cannot advance `ResponseUpdate`; a unit becomes visible only after confirmed playback crosses its fixed end time plus the safety lag. Later text therefore cannot redistribute previously played time across an expanded text buffer.

StreamingTextTTS L1 uses fixed conservative estimates of 280 ms per Chinese character, 380 ms per English word, 160 ms per punctuation mark, and a 300 ms playback safety lag.

| Level | Used while | Required information |
| --- | --- | --- |
| **L1 online estimate** | The text and audio streams are still being generated | Accepted text, prepared-audio duration, and confirmed played-audio duration |
| **L2 total-duration ratio** | The complete audio stream is available and forced alignment is running | Complete accepted text, final processed PCM, and actual total audio duration |
| **L3 precise calibration** | Forced alignment has returned usable units | Character- or word-level timestamps mapped onto the complete text |

The actual total duration is calculated from the PCM sent to the frontend; dividing the original duration by playback speed gives a different, wrong number.

## Switching and failure handling

`ResponseUpdate` remains monotonic. If precise alignment is ahead, progress advances immediately. If it is behind the displayed position, progress pauses until the precise timeline catches up.

If alignment fails, the rough result stays in use. Complete playback never waits for alignment. On early interruption, an available precise result is preferred, an in-flight alignment task is awaited briefly, and the rough prefix is used on timeout. StreamingTextTTS interrupted before its complete audio is generated normally has only the online estimate unless the service exposes how much input text it has consumed.

## Debugging frontend text updates

Set `XTALK_LOG_LEVEL='xtalk.serving.modules.tts_playback_manager=DEBUG'` to inspect the decision behind every frontend-facing `ResponseUpdate`. Each update is emitted as one log record:

```text
TTS response update - session: ..., source: regular:L2-precise, mode: regular, level: L2-precise, state: ready, played_ms: 820.0, total_ms: 1600.0, delta: 'text added by this update', text: 'complete displayed prefix'
```

`source` is `playback-complete` when completing a played segment confirms its remaining text, or `<mode>:<level>` when the active calibration level advances the prefix. An update that crosses a segment boundary contains both sources joined by `+`. The defensive `final-commit` source appears only when a final fallback update cannot be attributed to either case. `delta` and `text` use escaped string representations, so embedded line breaks remain within one physical log line. These fields can contain conversation content; enable them only while debugging.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Forced aligner interface →](../reference/models/forced_aligner.md)** · Reference

-   **[TTS Pipeline →](tts.md)** · Explanation

-   **[Register a new model →](../tutorial/introduce_a_new_model.md)** · Tutorial

</div>
