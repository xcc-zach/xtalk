# TTS Pipeline

TTS models produce audio; `TTSManager` decides when to ask for it and how to hand it to the client. Keeping these responsibilities apart lets a model emit audio at whatever granularity its backend prefers while the frontend protocol stays fixed.

## How the service layer consumes `TTS`

`TTSManager` buffers the LLM response, splits it into sentences, and calls TTS sentence by sentence. The chunks a model returns are synthesis-side chunks; the service layer re-splits the audio into `TTSChunkReady` pieces of about 100 ms before sending them out. Models therefore need no alignment with the frontend transport granularity. Continuous, correctly ordered PCM is sufficient, and a model that naturally emits very large chunks still fits the playback protocol.

## How the service layer consumes `StreamingTextTTS`

When the active model is a `StreamingTextTTS`, `TTSManager` bypasses the `pending_sentence_buffer` and sentence-by-sentence path and follows this event flow:

```text
TurnTTSStartRequested
  -> StreamingTextTTS.start()
  -> start a background audio_stream reader task

TurnTTSTextAppendRequested(text)
  -> StreamingTextTTS.append_text(text)

TurnTTSFlushRequested
  -> StreamingTextTTS.flush()
  -> StreamingTextTTS.stop()

TurnTTSStopRequested / shutdown
  -> StreamingTextTTS.stop()
```

The background reader publishes audio as soon as it arrives:

```text
StreamingTextTTS.audio_stream() yields PCM
  -> TTSManager splits it into about 100 ms chunks
  -> publish TTSChunkReady(audio_chunk=chunk, sample_rate=...)
  -> OutputGateway sends it to the frontend
```

Text reaches the upstream service through `append_text` the moment it is generated, and generated audio is wrapped as `TTSChunkReady` immediately. Flushing happens only on `TurnTTSFlushRequested`; sentence boundaries carry no flush semantics, which keeps the upstream session free to decide its own prosody boundaries.

## Playback progress tracking

`TTSPlaybackManager` advances played-audio time from frontend `TTSChunkPlayed` events and publishes `ResponseUpdate`. Configuring `forced_aligner` further calibrates the played-text position; see [Forced Alignment Pipeline](forced_aligner.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[TTS interface →](../reference/models/tts.md)** · Reference

-   **[Register a new model →](../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Forced Alignment Pipeline →](forced_aligner.md)** · Explanation

</div>
