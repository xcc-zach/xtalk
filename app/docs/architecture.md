# XTalk Desktop Phase 0 Architecture

## Boundary

Every desktop-specific source, test, resource, and build script lives in
`app/`. The backend uses the installed package's public runtime builder and
tool APIs, including the documented `xtalk.models.agents.tools` API required
for native tools. The UI imports only from the `xtalk-client` package root.
Neither adapter imports private underscored APIs or copies core implementation
code.

## Startup protocol

1. Tauri loads the persisted path of the user's external model configuration
   without starting user-selected processes during native setup.
2. When no valid path is selected, the window opens without a sidecar and the
   WebView immediately opens the native JSON file picker.
3. After the WebView subscribes to managed-model progress, it asks Tauri to
   ensure the selected backend is running. Tauri reuses a healthy owned
   instance; otherwise it starts every referenced managed service before the
   packaged Python sidecar.
4. Tauri creates an ephemeral launch token and application data directories
   and starts the Python sidecar without placing secrets in argv.
5. Tauri writes one bounded JSON launch message to the sidecar's stdin.
6. The sidecar loads the selected XTalk configuration, binds an OS-assigned
   loopback port, starts FastAPI, and emits one readiness JSON line.
7. Tauri validates the readiness protocol and exposes the endpoint and token
   through a narrow command to the WebView.
8. The UI creates an `xtalk-client` session with frontend VAD and enhancement
   disabled.
9. Tauri supplies the packaged Silero model as a top-level configuration
   fallback, so the Python sidecar produces speech boundaries when the selected
   configuration does not declare its own VAD.
The sidecar disables HTTP access logs so the query capability used by the
public SDK cannot appear in URL logs.

## Local interface

The WebView follows the layout and visual language of `examples/sample_app`
without importing example implementation code. It provides a left-side
conversation history that starts collapsed, a context-sensitive top bar, Orb
and conversation views, a bottom glass control dock, and a right-side
settings-and-diagnostics drawer. The conversation sidebar creates new
conversations and switches among all sessions returned by the public client
API. A same-style Tools action below New chat opens a centered dialog that can
copy user tool directories into AppData, update built-in or user enablement,
delete user tools, and restart the sidecar to apply changes. Built-ins are
visibly grouped and never expose a delete action. The settings drawer organizes
language, external model configuration, runtime status, local-service
diagnostics, and recovery into individually expandable rows. It can replace the
model configuration and rediscover the local service. The language row defaults
to automatic
operating-system locale detection and allows a persisted Simplified Chinese or
English override. Static content, dynamic status, accessibility labels, and
native picker text all use the same resolved language. Light, dark, and
narrow-window layouts share the same desktop adapter and offline state model.
The macOS bundle includes the microphone usage description and audio-input
entitlement required when a user starts a voice conversation.

Conversation history is server-authoritative and stored in
`chat_history.sqlite3` under the application data directory forced into the
sidecar configuration. The private desktop startup protocol carries one fixed
anonymous user identifier separately from public `service_config`; the
launch-token and Origin boundary keeps that single-user identity private to the
app. Sidebar session titles and messages therefore survive application and
sidecar restarts without a second WebView-owned history store.

## Authentication contract

All HTTP requests require the launch token and an approved Origin. App-owned
clients send the token in `X-XTalk-App-Token`; the `xtalk-client` adapter adds it
only to its explicit HTTP service URLs because the public SDK does not expose a
custom-header hook.

The WebSocket URL remains token-free so the SDK does not persist the launch
token as part of its local-storage key. A WebSocket is accepted by the outer
middleware only when it carries either the launch token or the XTalk access
token obtained from the launch-token-protected login route. XTalk then performs
the authoritative access-token validation. Loopback peer and Origin checks are
always required.

Text turns do not use an app-owned HTTP endpoint. The public client sends them
through the already authenticated XTalk WebSocket.

## Text input and timer flow

The desktop adapter calls the public `Session.sendText()` method. The SDK trims
the message, limits it to 2,048 JavaScript characters, sends `submit_text` over
the connected WebSocket, and waits up to ten seconds for a matching
`finish_asr` carrying `origin="text"`. The core backend validates an 8 KiB UTF-8
limit and publishes the synthetic text turn through the ordinary VAD, ASR,
Agent, Tool, TTS, and persistence pipeline. The UI never appends an optimistic
local message.

Only one text submission may await confirmation per client Session. Disconnect,
close, reopen, and session switching cancel that pending submission; the client
does not queue or automatically retry a turn that may execute side-effecting
tools.

The app registers enabled built-in and user tools through the public runtime
builder. The bundled asynchronous `timer`, matching
`examples/sample_app/custom_async_tool.py`, is a normal built-in manifest tool.
An enabled user tool with the same exported name takes precedence. Unit tests
cover `Running`, progress,
`Finished`, stop behavior, developer entrypoint loading, and the public
`ToolEngine` final update. The model smoke independently sends a text request,
observes `tool_called` for `timer`, and acknowledges a real assistant/TTS turn.
It does not require a second proactive LLM report because that model-driven
report can overlap the first response; no timer-specific serving workaround is
added for that timing.

Text input targets an already-open XTalk session. Since the public SDK's
`open()` still initializes microphone capture, starting that session requires
microphone permission even when the user subsequently types.

## Local voice pipeline

The WebView requests browser-native echo cancellation, automatic gain control,
and noise suppression, then sends 16 kHz mono PCM through the public
`xtalk-client` WebSocket with frontend VAD and the custom FastEnhancer disabled.
Tauri resolves the bundled `models/audio/silero_vad.onnx` resource, and the
sidecar loads it through the ordinary public XTalk model configuration.
Server-originated speech boundaries then start and finish the configured remote
ASR turn.

The model is pinned by upstream commit and SHA-256 in
`resources/manifests/audio-models.lock.json`. The resource verifier rejects a
missing or changed file, and the installed application never downloads this
base runtime asset.

## Optional native model runtime

Optional downloadable speech models run in a separate Rust HTTP sidecar under
`app/local-model-runtime`. The first engine implements MOSS-TTS-Nano directly
with SentencePiece and five ONNX sessions: reference-audio codec encode,
prefill, autoregressive decode, sampled local frame, and codec decode. Its
primary `POST /api/generate` contract matches the official Python/FastAPI
service: multipart `text` plus `prompt_audio` in, base64 WAV JSON out.
Reference audio is converted to 48 kHz before encoding, and generated output is
fixed to 48 kHz mono PCM16.

Apple Silicon packages also include the Swift sidecar in
`app/local-model-runtime-mlx`. It uses pinned `mlx-audio-swift` APIs to load
local SenseVoice and MOSS safetensor snapshots. It preserves the same offline
ASR WebSocket packet and MOSS multipart HTTP contracts, so the Python model
clients do not branch on the inference backend. Its MOSS response is likewise
48 kHz mono PCM16.

ONNX Runtime is an application resource, not a user-installed dependency. The
sidecar loads the exact packaged dynamic library passed through `--ort-dylib`;
model weights remain outside the application bundle. A selected
`managed://sensevoice-small` or `managed://moss-tts-nano` URL makes Tauri read
the immutable `managed-models.lock.json`, download only that service's pinned
files into `AppData/models/managed/<id>/<version>/`, verify file sizes and
SHA-256 values, and atomically write the completion marker. Every later launch
revalidates the installed snapshot before using it.

The managed URL accepts `?backend=cpu`, `?backend=cuda`, or `?backend=mlx`.
Without a query, Tauri selects a packaged CUDA provider on an NVIDIA device,
then Apple Silicon MLX, then CPU. An explicitly selected unavailable backend is
an error instead of an implicit fallback. CUDA and CPU share the ONNX snapshot;
MLX selects its separately pinned safetensor snapshot.

After the user selects a configuration, Tauri inspects it before applying it.
Configurations that request managed services open a blocking progress dialog.
Native progress events report model verification, per-file downloaded bytes,
service startup, and readiness. The rest of the interface remains inert until
the Python backend passes its health check, at which point the dialog closes
automatically. A startup failure leaves the dialog open with the error and a
close action.

For ONNX, Tauri starts SenseVoice through the packaged native
`sherpa-onnx-offline-websocket-server` and starts MOSS through the Rust sidecar.
For MLX, it starts one Swift sidecar per requested service.
It waits for TCP/readiness health boundaries, then deep-merges actual ephemeral
loopback URLs and the resolved AppData voice path into the Python startup
overlay. The selected file remains portable and contains no generated ports.
If installation, process startup, or Python startup fails, all newly started
children are stopped and the previous configuration is restored. An unexpected
managed-child exit also makes the backend connection unavailable.

The complete local example is
[`../examples/local_models.json`](../examples/local_models.json). Its LLM
matches `server_configs/sample.json` while intentionally leaving `api_key`
empty. SenseVoice consumes 16 kHz PCM through the existing offline WebSocket
client; MOSS reference audio and generated output use 48 kHz. The companion
[`../examples/local_models_mlx.json`](../examples/local_models_mlx.json)
explicitly selects MLX.

## Configuration

Release bundles contain no default XTalk model configuration. The native picker
accepts one external JSON file; Tauri requires an object root, enforces a 1 MiB
limit, canonicalizes the path, and persists only that path under AppConfig. The
configuration contents and provider credentials remain in the external file.
Replacing the selection first validates the new file, stops the active
sidecar, starts another sidecar, persists the successful selection, and asks
the WebView to rediscover the service.

The launch message selects that JSON configuration and may carry top-level
fallbacks plus a JSON overlay. Fallbacks only fill keys absent from the
selected configuration; an explicit model slot is preserved as a complete
value. The overlay is then deep-merged at the highest precedence. The Python
side remains model-type agnostic and only forces
`service_config.data_dir` into AppData. Missing, unknown, or invalid configured
models remain XTalk configuration errors.

Tests use `server_configs/sample.json` as the base model configuration. Optional
turn-detector tests inject a complete standard `type`/`params` object from an
external temporary configuration. The desktop VAD fallback makes the unchanged
sample configuration voice-capable. Private model repositories and credentials
are test inputs, not committed application configuration.

The freezer collects the installed public `xtalk.models` namespace and package
data because core model discovery is dynamic. Required optional dependency
groups are explicit `--xtalk-extra` build inputs; they do not introduce
model-type branches into application behavior.

## Built-in and user tools

Both sources contain Python files and an `xtalk_tool.json` manifest.
`display_name` may be a string or a language dictionary. `ui` is optional:

```json
{
  "display_name": {
    "zh": "计时器",
    "en": "Timer"
  },
  "entrypoint": "timer_tool:create_tools",
  "ui": {
    "entrypoint": "ui/index.html",
    "update_every_s": 0.5
  }
}
```

For user tools, Tauri assigns an internal identifier, recursively copies the
directory under `AppData/tools/<id>/`, and persists that identifier plus its
enabled state in `AppData/tools/registry.json`. Built-ins remain in the
read-only `resources/tools/` bundle and are indexed by
`resources/tools/builtin_tools.json`; origin, deletion permission, and default
enablement are App metadata rather than manifest fields. Built-in IDs use the
`builtin:<id>` namespace. Their enablement overrides are stored in
`AppData/tool_preferences.json`.

The native delete command resolves the tool source itself and rejects built-in
IDs, so deletion protection does not depend on the WebView. Both sources can
be disabled. The Python sidecar resolves the same `module:factory` entrypoint
for each enabled directory, and user exports take precedence over same-named
built-ins. A factory must return a list accepted by
`XtalkBuilder.add_agent_tools()`.

The configured Agent is built after this registry is loaded, so tool changes
take effect through a controlled sidecar restart. A failed developer factory is
omitted without preventing the remaining local service from starting.

The Codex built-in is represented by one catalog entry and one manifest whose
factory returns five native async tools: search, create, continue, set-model,
and delete. Consequently its enabled state is atomic; the App has no per-export
preference. The bundle is disabled by default. Every business thread and turn
uses the official Python SDK with `Sandbox.full_access`, accepts any existing
local directory as `cwd`, uses the SDK's no-prompt approval mode, and explicitly
reapplies the model and reasoning effort stored for that session. Tool
descriptions contain the conditional routing rules, so no Codex-specific
instruction is injected into the Agent's
developer instructions.

The App-side session index is SQLite at
`AppData/tool-data/codex/codex_sessions.sqlite3`. It stores the SDK thread ID,
compact title/summary, working directory, model, effort, status, and timestamps;
the SDK remains responsible for the actual thread transcript. Natural-language
lookup first bounds the active index to 30 candidates, then gives only that JSON
snapshot to an ephemeral resolver thread with a strict output schema. Returned
IDs must be a subset of the snapshot. Delete archives the SDK thread before
marking it inactive in the App index, and per-session async locks serialize
continue, reconfigure, and archive operations.

Asynchronous tool progress is not fed back into the conversational stream.
The Agent records lifecycle updates in tool history, while only the terminal
update triggers one final natural-language report. This prevents a long Codex
turn from appearing as several truncated assistant messages. The Tool UI
broker retains each running status plus a bounded immutable emit history and
replays both when the App WebSocket binds or reconnects. Consequently live and
history cards remain visible even when execution started before the WebView
channel was ready.

The optional UI entrypoint is self-contained HTML, at most one MiB. It remains
separate from the Python entrypoint and declares capabilities by registering
`window.xtalkToolUI.status(callback)` and/or
`window.xtalkToolUI.emit(callback)`. A missing registration means that the
corresponding live or history UI does not exist. `update_every_s` controls live
status polling, defaults to one second, uses `-1` to disable periodic polling,
and otherwise accepts 0.1 through 3600 seconds.

The App wraps only native `AsyncTool` classes at registry load time. The wrapper
delegates the original lifecycle unchanged, observes `astatus()`, and publishes
read-only events over a launch-token-protected App WebSocket. Live events track
the active call. Every original initial/update emit produces a separate history
event containing the emitted message plus status at that moment. History
snapshots are immutable, bounded to 200 items per session, and stored in WebView
AppData under the persisted session ID. The broker also keeps the latest 200
emits per active-process session for reconnect recovery; stable call/sequence
IDs let the WebView deduplicate those replays. Live state is memory-only.

The conversation top bar does not repeat the XTalk product name. It remains
empty when no live-capable tool is running. Active tools create one compact,
collapsed status bar containing the latest status and running count; the user
can expand it to inspect all current live UI cards. Live cards are not inserted
into the message timeline. History cards remain anchored to their immutable
emit positions.

Each card uses a separate `sandbox="allow-scripts"` iframe. The injected CSP
blocks external resources and network APIs, and the bridge suppresses link and
form actions. The opaque frame origin has no Tauri capability. The frame can
receive status/emit data and report desired height, but it cannot invoke, stop,
or otherwise operate the tool. To keep the App's strict top-level CSP, the host
publishes each prepared document behind a high-entropy, 30-second, one-time
loopback ticket; the launch token never enters the iframe URL or document. The
host owns full available width and clamps height to 120–420 px for live cards
and 80–600 px for history cards, with both also capped at 60% of the window
height.

## Shutdown

The UI closes its XTalk session before requesting shutdown. Tauri asks the
authenticated Python sidecar shutdown endpoint to stop, waits for a bounded
interval, and terminates only the child it started if graceful shutdown does
not finish. Managed model processes are then stopped in reverse startup order.

## Phase 0 limitations

- The external model configuration must remain available at its selected path.
  Sandboxed distribution would require a platform security-scoped bookmark
  before this path-based persistence can be used there.
- PyInstaller `onedir` support files remain beside the sidecar in development
  and ordinary bundles; macOS app bundles place them in `Contents/Frameworks`
  as required by the bootloader. Runtime validation rejects incomplete layouts.
- Additional tool dependency management, local enhancement, and provider
  settings belong to later phases. Developer tool directories can use Python
  packages already present in the frozen sidecar.
