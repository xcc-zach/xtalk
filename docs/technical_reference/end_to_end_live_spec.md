# Single-call GPT-Live specification (protocol 2)

## Scope

One fresh voice call supports start, continuous capture and silence, bidirectional PCM, Live-native interruption, microphone mute, incremental transcripts, stop, and visible errors. The existing cascade graph retains history, tools, recording, speaker processing, and response alignment. Live tools, enhancement, speaker processing, retrieval, recording, word alignment, persistence, recovery, and automatic reconnect are deferred.

## Framework boundaries

```text
Application auth → /v1/realtime → Service/EventBus
capture → InputGateway → EndToEndManager → GPTLive writer → OpenAI
OpenAI → GPTLive reader → normalized events
  ├─ transcripts/errors → OutputGateway → ActionHandler → Conversation/UI
  └─ pending PCM → delivery task → _AudioPlaybackBudget → OutputGateway
       → shared WebOutputAudioSession scheduler → cumulative playback receipt
```

- `EndToEnd`: `clone()`, immutable `audio_profile`, secret-free metadata, `open()`, `append_audio()`, `events()`, `set_input_muted()`, `close()`.
- `GPTLive`: one upstream reader, one paced writer, one bounded audio/control admission queue. Startup, command acknowledgements, and finalization run independently of browser playback.
- `EndToEndManager`: client readiness, current call, normalized event consumption, pending PCM, delivery and stall supervision.
- `InputGateway` / `OutputGateway`: input validation and WebSocket I/O through the existing bounded, ordered ASGI `RealtimeConnection` queue. Gateway owns no playback budget.
- `_AudioPlaybackBudget`: private to `end_to_end_manager.py`, following the original TTS chunking and Condition-based budget approach. It owns Live duration accounting, reservation rollback, stream identity, cumulative sample receipts, and cancellation. `TTSManager` retains its original internal accounting.
- `WebOutputAudioSession`: one PCM scheduler for both wrappers. It retains the original clock, recovery/cache, fade-in, actual start notification, idempotent completion, WebKit audio-clock fallback, partial-stop accounting and cleanup. A temporarily empty Live queue has no turn-completion meaning.
- Existing platform subclasses keep their existing abstract method obligations. Continuous playback is an optional capability.

Latest `dev` layering is preserved: login, history, and browser snapshots belong to sample `host/` and `conversation-store.js`; the SDK remains a pure Realtime client. Authenticated Live skips business conversation creation, history writes, and restoration. Protocol extensions use Service input registration and output subscription overrides, inherited once by each clone.

`AudioStreamTransport`, the independent continuous player, output suppression controls, Live SQL/history projections, duplicate Manager input queue, `DuplexState`, `restore_history()`, `append_context()`, and the second action dispatcher have been removed from the Live path. Existing database tables and rows remain intact.

## Configuration and limits

The sample selects `gpt-live-1`, voice `marin`, explicit Responses backend `gpt-5.6-luna`, `tools: []`, and `tool_choice: "none"`. A backend needs its own model access. Model `sample_rate` sets both capture and playback: mono signed PCM16LE at 16000 or 24000 Hz; default 24000. Conflicting `input_sample_rate` is rejected.

| Setting | Default | Owner |
| --- | --- | --- |
| Delivery chunk | 100 ms | Live Manager budget |
| In-flight audio | 300 ms | Live Manager budget |
| Pending output | 5000 ms | Manager `service_config.end_to_end.pending_audio_ms` |
| No-progress deadline | 15 s | Manager `playback_stall_timeout_s` |
| Client media readiness | 30 s | Manager `client_ready_timeout_s` |
| Unsent input / pending adapter audio | 5000 ms | GPTLive `max_pending_audio_ms` |
| Adapter event/write queue | 256 entries each | GPTLive `max_pending_events` |
| Command acknowledgements / graceful close | 15 s each | GPTLive timeout settings |
| Browser PCM frame / JSON / upstream message | 200 ms / 64 KiB / 1 MiB | gateways / adapter |

The 300 ms window waits for progress. Sustained pending output overflow or missing playback progress ends the call with a visible reason. Closing wakes blocked delivery and discards unsent input. Reservations precede publication. Live audio delivery uses `WAIT_UNTIL_COMPLETE` with `require_success=True`: missing handlers or handler exceptions return `False`, allowing immediate reservation rollback. Default EventBus error isolation is preserved. Failed Live JSON or PCM writes notify the call owner once, including failed startup and error notifications.

Live defaults to `enable_persistence: false`. Explicit persistence, recording, full audio export, multi-speaker policy, or forced alignment is rejected before connecting. Default model assembly rejects cascade models mixed into the Live graph; explicit Manager lists and EventBus subscription overrides remain available.

## Protocol and startup

1. Authenticate through the sample host and connect to `/v1/realtime?model=xtalk`. The pure SDK uses its host-provided URL/API key.
2. Standard `session.created` advertises protocol 2, a fresh `stream_id`, media, capabilities, policy, and metadata in `session.xtalk.end_to_end`.
3. Send `session.update`, selecting `end_to_end_protocol: 2` and `output_audio_encoding` in `session.xtalk`; the server replies with `session.updated`.
4. Construct exactly one input/output pair, initialize media, and send `xtalk.audio_stream.ready` with `stream_id`.
5. The server opens the provider and waits for `session.started`, then sends `xtalk.audio_stream.started`. Only the current uncancelled runtime sets `connected`, enables uploading, and displays readiness.

Version 1 clients receive an explicit rebuild/refresh error. Audio before readiness is discarded locally; users speak after the UI reports readiness. Permission denial, close, and cancellation invalidate waiters. Late permissions stop their media tracks. An old connection cannot restore a new runtime's state.

Client controls:

```json
{"type":"session.update","session":{"xtalk":{"end_to_end_protocol":2,"output_audio_encoding":"base64"}}}
{"type":"xtalk.audio_stream.ready","stream_id":"current-call"}
{"type":"xtalk.audio_stream.progress","stream_id":"current-call","played_samples":2400}
{"type":"xtalk.audio_stream.input_muted","stream_id":"current-call","muted":true}
{"type":"xtalk.audio_stream.close","stream_id":"current-call"}
```

Input uses binary PCM or standard `input_audio_buffer.append`. Output uses `xtalk.audio_stream.delta`: JSON carries Base64 in `delta` by default; negotiated `binary_pcm` uses the framework's four-byte big-endian JSON-header length, UTF-8 JSON header, and PCM payload. Headers contain `version: 1`, `type: "xtalk.audio_stream.delta"`, `event_id`, and `stream_id`. Each envelope occupies one WebSocket message.

Receipts carry only the stream identity and cumulative played sample count. Stale, decreasing, noninteger, or beyond-sent progress is rejected. Actual source completion and stop settlement produce receipts; there is no progress polling or second frontend credit window.

## Input, transcripts and UI

Browser capture retains AEC and disables extra VAD/enhancement for Live. It continuously sends PCM; microphone mute creates equal-duration zero PCM once in the input layer. Provider mute commands retain internal acknowledgement tracking. The existing `Session.muted` preference survives reopening.

`xtalk.transcript.delta` preserves speaker, exact delta text, approximate start/end time, provider event identity, stream identity, and local arrival order. Duplicate identities are removed. These are independent transcript fragments, without ASR-final, ResponseFinish, or heard-text semantics. Timestamps serve display ordering only.

The sample reuses start/stop/mute controls and chat bubbles. Start remains cancellable while connecting. Errors appear in Toast and state text, including the default orb view. Current captions survive stop and clear at next start or reload. Live hides unsupported history, upload, dynamic voice, and debug controls and skips their requests. `SessionConfig.mode: "end_to_end"` lets hosts skip historical browser snapshots before connecting; the sample supplies it from the server-rendered mode.

## Stop and errors

Local capture, scheduled audio, observers and pending PCM stop immediately. The adapter discards queued input, sends `session.close`, retains its reader, and waits up to 15 seconds for `session.closed`. `close()` is idempotent.

`xtalk.audio_stream.ended` includes `outcome`, `reason`, and `finalization` (`complete`, `incomplete`, or `not_started`). These distinguish user stop, graceful upstream completion, transport/protocol failure, and unconfirmed finalization. Provider safety, expiry and connection-loss termination have visible reasons. Rejected commands and audio/backend errors may leave an active call running. Transport failure ends this call and prompts manual restart with fresh context.

## Validation

Run the focused backend cases and [browser acceptance workflow](../tutorial/gpt_live.md). Cover six-chunk bursts, delayed playback, suspension, missing `onended`, stalls, startup cancellation, permission denial, first ready audio, mute preference, ten restarts, errors, 16/24 kHz media, version rejection, and preserved database rows. Cascade response identity, TTS alignment, turn races, ASR finalization, tools' playback barrier, and history remain regression targets.

Real API connectivity and synthetic audio validate the protocol and browser chain. Human listening remains necessary for conversational quality, backchannels, microphone acoustics, and native interruption. The tutorial records the verified boundary.

Official sources: [WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live), [Managing sessions](https://developers.openai.com/api/docs/guides/live-conversations).
