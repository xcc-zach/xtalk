# Realtime API Protocol Reference

Xtalk exposes an OpenAI Realtime style API with a standard subset and Xtalk extensions. Clients must handle playback acknowledgements and ASR replacement semantics.

## Transport conventions

Connect to `WS /v1/realtime?model=xtalk`, using the configured model alias. Control events and all server output use JSON with `type`. Server events include `event_id`; clients may supply `event_id`. C→S means client to server; S→C means server to client. Each connection owns a temporary Session and Conversation (dialogue context). An Item is a message; a Response represents one reply generation. Output and content indices start at zero. “Content identity” means `response_id`, `item_id`, `output_index`, and `content_index`.

Normal input/response audio is **24 kHz, mono, signed PCM16 little-endian**, without a WAV header. JSON Base64 and raw PCM binary input are both enabled by default; output remains Base64 JSON. The format of `session.audio.mix.delta` is specified by its own metadata.

Each input message is limited to 1 MiB (UTF-8 bytes for JSON, PCM bytes for binary). Both input forms share an audio buffer budget of 60 seconds; committing or completing an audio turn releases the corresponding input budget so more audio can be sent. Odd byte counts, invalid Base64, and oversized input are rejected without changing valid buffering. Send 20–100 ms per audio message for realtime streaming.

## Session settings

`session.update.session` accepts `type` (only `realtime`), `instructions`, `output_modalities` (`["audio"]` or `["text"]`), `audio.input.format/transcription/turn_detection`, and `audio.output.format/voice/speed/emotion`. Both formats are fixed to `{"type":"audio/pcm","rate":24000}`. Transcription defaults to the configured model alias, such as `{"model":"xtalk"}`; setting it to `null` disables external transcription events while internal ASR continues.

`turn_detection` accepts `null` or a `server_vad` object containing `type`, `threshold`, `prefix_padding_ms`, `silence_duration_ms`, `create_response`, and `interrupt_response`. The default is `{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500,"create_response":true,"interrupt_response":true}`. Requested VAD threshold updates require model support. Manual commits retain Xtalk automatic turn responses.

`voice` defaults to `default`; `speed` defaults to `1.0` and accepts `0.25–1.5`. Updating speed during an active response is an Xtalk extension; already processed audio is not processed again. The Xtalk `emotion` extension has the form `{"name":"happy","vector":[]}`; a nonempty name takes precedence over the vector, and updates cannot leave both empty. Updating instructions, voice, or emotion requires model support and otherwise returns an error. Omitted settings retain their values.

## WebSocket error events

Missing fields, incorrect types, out-of-range values, and unsupported operations produce an `error` event over WebSocket. Clients should listen for this event and use `error.code`, `error.message`, and `error.param` to correct the request. The connection remains usable after validation failure, allowing subsequent requests.

- `error.type`: `invalid_request_error` for invalid requests; `server_error` for unexpected server execution failures.
- `error.code` and `error.message`: the specific error code and a readable explanation.
- `error.param`: the related protocol field path, or `null` when no field can be identified.
- `error.event_id`: the client event ID that caused the error, or `null` when absent or unavailable.
- Top-level `event_id`: the independent identifier of the server error event, not the client request ID.

```json
{
  "type": "error",
  "event_id": "evt_server_error",
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_value",
    "message": "Input should be less than or equal to 1",
    "param": "session.audio.input.turn_detection.threshold",
    "event_id": "evt_client_update"
  }
}
```

A failed `session.update` returns `error` without a success acknowledgement through `session.updated`. For other requests, use their corresponding success events to determine the outcome; successfully sending a message does not confirm that the operation succeeded.

## Event table

| Stage | Event or message | Direction | Key fields (excluding envelope) | Behavior and client handling |
| --- | --- | --- | --- | --- |
| Session | `session.queue.updated` / `session.queue.granted` | S→C | updated: `position`, `limit`, `active`; granted: — | Xtalk extension: admission queue status and grant, possibly before session.created. A grant does not mean the session is ready. |
| Session | `session.created` | S→C | `session` | Announces a ready temporary session and its complete effective configuration; no attach handshake. |
| Session | `session.update` | C→S | `session` | Partially updates session settings; omitted fields remain unchanged. Unsupported fields or model capabilities return errors. See Session settings above. |
| Session | `session.updated` | S→C | `session` | Returns complete effective settings after a successful update; do not execute configuration echoes as commands. |
| Audio input | `input_audio_buffer.append` | C→S | `audio` | Appends Base64 PCM audio in order, without per-message acknowledgement or implicit commit. |
| Audio input | Raw PCM binary message | C→S | WebSocket binary | Xtalk extension, enabled by default; equivalent to append. Can be mixed with Base64 and shares resampling, buffering, and turn handling. |
| Audio input | `input_audio_buffer.speech_start` | C→S | — | Xtalk extension: client speech-start notification, without commit. In manual mode starts ASR and replays buffered audio. |
| Audio input | `input_audio_buffer.commit` | C→S | — | Commits current audio; an empty buffer is an error. Retains automatic turn responses without response.create. |
| Audio input | `input_audio_buffer.clear` | C→S | — | Clears uncommitted input and resets resampling and manual speech state. |
| Audio input | `input_audio_buffer.committed` | S→C | `item_id` | Acknowledges an audio turn commit and identifies its input item; transcription and response may still be pending. |
| Audio input | `input_audio_buffer.cleared` | S→C | — | Acknowledges input buffer clearing. |
| Audio input | `input_audio_buffer.speech_started` / `input_audio_buffer.speech_stopped` | S→C | `origin` | Speech boundary notifications retain Xtalk origin semantics; update client state without echoing speech_start or commit. |
| Input items and transcription | `conversation.item.create` | C→S | `item`, `xtalk.generate_response` | Accepts message items with user/system/assistant roles and text or input_text content. Optional IDs must be unique. Context injection is the default; only user items with xtalk.generate_response=true trigger a live text turn, with nonblank text of at most 8192 UTF-8 bytes. |
| Input items and transcription | `conversation.item.added` / `conversation.item.done` | S→C | `item` | Announces input item addition and completion; finalized audio items may contain a transcript. |
| Input items and transcription | `conversation.item.input_audio_transcription.delta` | S→C | `item_id`, `content_index`, `delta`, `xtalk.mode`, `xtalk.revision` | ASR preview extension: append adds text; replace replaces it with the full delta. Revisions increase monotonically; ignore stale previews and previews after completed. |
| Input items and transcription | `conversation.item.input_audio_transcription.completed` | S→C | `item_id`, `content_index`, `transcript` | Final complete transcript; transcription and response progress asynchronously, without a global ordering guarantee. |
| Input items and transcription | `conversation.item.input_audio_transcription.failed` | S→C | `item_id`, `content_index`, `error` | Reports input transcription failure without closing the connection. |
| Response generation | `response.created` | S→C | `response` | Starts generation; response contains id/status/output. |
| Response generation | `response.output_item.added` / `response.output_item.done` | S→C | `response_id`, `output_index`, `item` | Announces output item creation and completion. |
| Response generation | `response.content_part.added` / `response.content_part.done` | S→C | `response_id`, `item_id`, `output_index`, `content_index`, `part` | Announces output content part creation and completion. |
| Response generation | `response.output_text.delta` / `response.output_text.done` | S→C | Content identity; `delta` / `text` | Text-only output: delta appends generated text; done carries the complete text. |
| Response generation | `response.output_audio.delta` / `response.output_audio.done` | S→C | Content identity; `delta` in delta events | Audio output: delta contains Base64 PCM; done ends audio generation, not playback. |
| Response generation | `response.output_audio_transcript.delta` / `response.output_audio_transcript.done` | S→C | Content identity; `delta` / `transcript` | Text of synthesized audio, not unsynthesized LLM text. Successful sentences append deltas; successful whole-turn streaming emits one full delta followed by the full transcript. |
| Response generation | `response.done` | S→C | `response.id`, `response.status`, `response.output` | Generation terminal state: completed/cancelled/failed/incomplete. Does not wait for playback acknowledgements or mean the user has heard all audio. |
| Response generation | `response.cancel` | C→S | `response_id` (optional) | Cancels specified or active generation; errors when nothing can be cancelled. Playback queue handling and acknowledgements remain separate. |
| Playback controls and acknowledgements | `response.output_audio.playback.chunk_played` | C→S | `response_id` | Xtalk extension: acknowledge each original audio event block once after actual playback to release output budget; do not acknowledge resegmented playback blocks. |
| Playback controls and acknowledgements | `response.output_audio.playback.finished` | C→S | `response_id` | Xtalk extension: report completion after all audio actually plays; does not replace block acknowledgements. |
| Playback controls and acknowledgements | `response.output_audio.playback.stopped` | C→S | `response_id`, `played_audio_ms` | Xtalk extension: after stopping, report incremental played milliseconds from unconfirmed audio, not cumulative playback position. |
| Playback controls and acknowledgements | `response.output_audio.playback.pause` / `response.output_audio.playback.resume` | S→C | `response_id` | Xtalk extension: pause or resume playback of the specified response. |
| Playback controls and acknowledgements | `response.output_audio.playback.stop` | S→C | `response_id` | Xtalk extension: stop and flush queued audio, then report playback.stopped with incremental played duration. |
| Playback controls and acknowledgements | `response.output_audio.playback.updated` / `response.output_audio.playback.done` | S→C | `response_id`, `text` | Xtalk extension: full playback-confirmed text snapshots and final settlement for display and history, separate from generated text delta/done. |
| Notifications | `conversation.caption.updated` | S→C | `text`, `is_final`, `reason` | Xtalk extension: complete caption snapshot; replace displayed captions rather than append ASR text. |
| Notifications | `conversation.retrieval.updated` | S→C | `text`, `is_final` | Xtalk extension: full retrieval snapshot replaces displayed results; the host owns knowledge ingestion. |
| Notifications | `conversation.speaker.updated` | S→C | `speaker_id`, `reason` | Xtalk extension: speaker identity notification. |
| Notifications | `response.tool.called` | S→C | `name`, `args` | Xtalk extension: notification of a backend tool call; execution remains server-side, without client function_call_output. |
| Notifications | `session.latency.updated` | S→C | `network_latency_ms`, `asr_latency_ms`, `llm_first_token_ms`, `llm_sentence_ms`, `tts_first_chunk_ms` | Xtalk extension: latency metrics in milliseconds. No clock synchronization is provided; network_latency_ms is not measured RTT. |
| Notifications | `session.audio.mix.delta` | S→C | `delta`, `sample_rate`, `channels`, `format` | Xtalk extension: mixed conversation audio when send_full_audio_to_client is enabled. Base64 delta uses event format metadata; it is separate from individual response audio. The host owns file storage. |
| Errors | `error` | S→C | `error.type`, `error.code`, `error.message`; optional `error.param`, `error.event_id` | Protocol or runtime error; request errors can correlate event_id. Binary errors have error.event_id=null. Invalid input normally preserves the connection; sustained output overload may close it. |

## Generation and playback ordering

For the same output content, ordering is `response.created` → `response.output_item.added` → `response.content_part.added` → corresponding delta/done → `response.content_part.done` → `response.output_item.done` → `response.done`. Generation and playback progress are separate: send chunk_played after each original audio block actually plays, then finished after all playback. On stop, flush the queue and report unconfirmed played duration with stopped. Missing block acknowledgements can stall subsequent audio delivery.

## Supported scope

Unrecognized input events return `unsupported_event`. Built-in handlers do not support `response.create`, `conversation.item.truncate`, `change_tts_model`, `change_llm_model`, `ping`, `pong`, or `clock_sync`. Audio turns or text input with `xtalk.generate_response=true` trigger replies. Models are configured server-side, and interrupted playback is settled with `response.output_audio.playback.stopped`.

## Send examples

These snippets use a connected `websocket` that has received `session.created`. `pcm_bytes` contains raw audio in the format above. Choose one encoding and send each audio chunk once:

```python
import base64
import json

# Option 1: JSON Base64.
await websocket.send(json.dumps({
    "type": "input_audio_buffer.append",
    "audio": base64.b64encode(pcm_bytes).decode("ascii"),
}))
```

Alternatively, send binary audio directly:

```python
# Option 2: raw PCM, replacing the Base64 send above.
await websocket.send(pcm_bytes)
```

Server VAD determines turns by default. For manual control, disable automatic detection, wait for `session.updated`, then send audio. Commit after sending the complete turn:

```python
await websocket.send(json.dumps({
    "type": "session.update",
    "session": {"audio": {"input": {"turn_detection": None}}},
}))
# After the receive loop confirms session.updated, send all audio for this turn.
await websocket.send(pcm_bytes)
await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
```

Send live text separately using `conversation.item.create`:

```python
await websocket.send(json.dumps({
    "type": "conversation.item.create",
    "item": {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello"}],
    },
    "xtalk": {"generate_response": True},
}))
```

Omit `xtalk.generate_response` to add context only, for example when injecting history in order.
