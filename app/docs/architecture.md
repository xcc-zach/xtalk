# XTalk Desktop Phase 0 Architecture

## Boundary

Every desktop-specific source, test, resource, and build script lives in
`app/`. The backend uses the installed package's public runtime builder and
tool APIs, including the documented `xtalk.models.agents.tools` API required
for native tools. The UI imports only from the `xtalk-client` package root.
Neither adapter imports private underscored APIs or copies core implementation
code.

## Startup protocol

1. Tauri loads the persisted path of the user's external model configuration.
2. When no valid path is selected, the window opens without a sidecar and the
   WebView immediately opens the native JSON file picker.
3. Tauri creates an ephemeral launch token and application data directories.
4. Tauri starts the packaged Python sidecar without placing secrets in argv.
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
without importing example implementation code. It provides a centered brand
bar, Orb and conversation views, a bottom glass control dock, and a right-side
settings-and-diagnostics drawer. The drawer shows the selected external model
configuration and can replace it, restart the sidecar, and rediscover the local
service. Light, dark, and narrow-window layouts share the same desktop adapter
and offline state model. The macOS bundle includes the microphone usage
description and audio-input entitlement required when a user starts a voice
conversation.

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

The app registers an asynchronous `timer` through the public runtime builder,
matching `examples/sample_app/custom_async_tool.py`. The example module itself
is not imported because importing it executes CLI and asset-download setup.
Unit tests cover `Running`, progress, `Finished`, stop behavior, and the public
`ToolEngine` final update. The model smoke independently sends a text request,
observes `tool_called` for `timer`, and acknowledges a real assistant/TTS turn.
It does not require a second proactive LLM report because that model-driven
report can overlap the first response; no timer-specific serving workaround is
added for that timing.

Text input targets an already-open XTalk session. Since the public SDK's
`open()` still initializes microphone capture, starting that session requires
microphone permission even when the user subsequently types.

## Local voice pipeline

The WebView sends 16 kHz mono PCM through the public `xtalk-client` WebSocket
with frontend VAD and enhancement disabled. Tauri resolves the bundled
`models/audio/silero_vad.onnx` resource, and the sidecar loads it through the
ordinary public XTalk model configuration. Server-originated speech boundaries
then start and finish the configured remote ASR turn.

The model is pinned by upstream commit and SHA-256 in
`resources/manifests/audio-models.lock.json`. The resource verifier rejects a
missing or changed file, and the installed application never downloads this
base runtime asset.

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

## Shutdown

The UI closes its XTalk session before requesting shutdown. Tauri asks the
authenticated sidecar shutdown endpoint to stop, waits for a bounded interval,
and terminates only the child it started if graceful shutdown does not finish.

## Phase 0 limitations

- The external model configuration must remain available at its selected path.
  Sandboxed distribution would require a platform security-scoped bookmark
  before this path-based persistence can be used there.
- PyInstaller `onedir` support files remain beside the sidecar in development
  and ordinary bundles; macOS app bundles place them in `Contents/Frameworks`
  as required by the bootloader. Runtime validation rejects incomplete layouts.
- General tool management, local enhancement, provider settings, and optional
  component supervision belong to later phases. The backend Silero VAD and
  fixed sample-compatible timer are the first slices implemented beyond the
  Phase 0 shell.
