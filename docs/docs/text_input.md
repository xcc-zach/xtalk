# Text Input

## Overview

The current implementation exposes `Session.sendText(text)` from
`xtalk-client`. It accepts text in every connected stream state: `idle`,
`listening`, `processing`, and `speaking`. There is no stream-state guard in
`sendText`; only the realtime connection must be active.

Opening a session still requests microphone access and starts the existing
audio pipeline. Sending text does not enable, disable, mute, or recreate
microphone capture.

## Client Behavior

```ts
const session = createSession(websocketURL, config);

await session.open();
await session.sendText("Set a timer for two seconds.");
```

`sendText` performs the following work:

1. Trims surrounding whitespace.
2. Rejects blank text and normalized strings whose JavaScript `length`
   exceeds 2,048.
3. Rejects the call unless `connectionState` is `connected`.
4. Allows only one text submission awaiting confirmation.
5. Registers the `finish_asr` waiter before sending the request.

The WebSocket request is:

```json
{
  "action": "submit_text",
  "text": "Set a timer for two seconds."
}
```

The client does not insert an optimistic user message. The existing
`finish_asr` handler appends the final user message, so the conversation
contains one server-confirmed copy.

The confirmation timeout is 10 seconds. Disconnecting, closing, reopening, or
switching the session cancels a pending call. Text is not queued or retried
automatically.

## Server Processing

`TextMsgHandler` validates that `text` is a string, trims it, rejects blank
input, and enforces an 8 KiB UTF-8 limit. It ignores any client-provided
`origin` and emits one serialized synthetic speech turn:

```text
submit_text
  -> TurnInputAbortRequested(origin="text")
  -> VADSpeechStart(origin="text")
  -> ASRResultPartial(origin="text")
  -> VADSpeechEnd(origin="text")
  -> ASRResultFinal(origin="text")
  -> existing Agent, Tool, TTS, and persistence pipeline
```

The partial result is inside the VAD boundaries. The final result is published
after VAD end, leaving the frontend in `processing` rather than letting a late
VAD-end action change it back to `idle`.

The managers handle text-origin events as follows:

- `ASRManager` invalidates in-flight recognition, clears acoustic ASR state,
  and drops microphone frames until the synthetic VAD end. A generation token
  prevents a late result from the replaced audio turn from being published.
- `VADManager` clears buffered VAD state and suppresses acoustic VAD processing
  for the same interval.
- `TurnTakingManager` does not start or finalize audio ASR for text. On text
  VAD start, it cancels interruptible LLM generation and requests TTS stop
  with reason `text_input`.
- `TurnDetectorManager` ignores text-origin VAD and ASR partial events because
  pressing Send already establishes the end of the user turn.
- `LatencyManager` clears frontend VAD timestamps left by an earlier audio
  turn before tracking text-turn latency.

Input metadata consistently uses `origin`:

- `client`: client-originated audio VAD;
- `turn_detector`: a boundary emitted by the Turn Detector;
- `asr`: ordinary audio recognition;
- `text`: a submitted text turn.

## State Semantics

| Current stream state | Text submission behavior |
| --- | --- |
| `idle` | Runs the synthetic text turn with no active input to replace. |
| `listening` | Invalidates and clears the unfinished acoustic input first. |
| `processing` | Cancels interruptible generation, then starts the text turn. |
| `speaking` | Requests generation and TTS playback stop, then starts the text turn. |

## Implicit Confirmation

`OutputGateway` includes `origin` in the existing ASR messages. No separate
success acknowledgement or request ID is used. A successful text submission
produces:

```json
{
  "action": "finish_asr",
  "data": {
    "text": "Set a timer for two seconds.",
    "origin": "text"
  }
}
```

The ordinary `finish_asr` handler updates the conversation first. The
`sendText` promise then resolves only when both `origin` is `text` and `text`
equals the normalized submission.

## Current Limits

- The session must be connected; disconnected and reconnecting submissions
  are rejected.
- Only one text submission may await confirmation per client session.
- The client limit is a normalized JavaScript string length of 2,048; the
  backend protocol limit is 8 KiB after UTF-8 encoding.
- Confirmation uses `origin` plus normalized text rather than a request ID.
- The client never retries automatically because a duplicate turn could
  repeat a side-effecting tool call.
- Microphone capture continues normally. Its frames are temporarily discarded
  by VAD and ASR while the synthetic text boundaries are being processed.

## Automated Coverage

`tests/test_text_input.py` covers request validation, event ordering,
per-session serialization, origin echoing, audio-ASR bypass, VAD suppression,
Turn Detector bypass, and latency reset. `tests/test_asr_finalization.py`
covers cancellation of an in-flight final ASR result.

The frontend production build verifies that `sendText` is present in the
published type declarations and browser bundles.
