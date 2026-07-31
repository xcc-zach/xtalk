# XTalk Desktop

This directory contains the isolated XTalk desktop application described in
[`../app_plan.md`](../app_plan.md). The current implementation targets Phase 0:
a Tauri WebView, a Python sidecar, explicit loopback security, and adapters that
consume XTalk's public APIs. It also includes realtime SDK text input, the
asynchronous timer used by the sample-app acceptance flow, and the first Phase
1 local-audio slice: a bundled backend Silero VAD.

## Development prerequisites

- Node.js 20 or newer
- Rust with the Tauri 2 platform prerequisites
- Python 3.10–3.13 for the sidecar build
- A prebuilt `xtalk` wheel
- Either the pinned npm `xtalk-client` package or a verified package artifact

The installed application does not require Python, Node.js, or Rust. They are
build-time dependencies only.

The locked npm dependency resolves the client from
`resources/artifacts/xtalk-client-0.2.6.tgz`. Prepare that ignored artifact from
a client build that exposes `Session.sendText()` before running `npm ci`.

## Local checks

```bash
cd app
python -m pytest
npm ci
npm run check
npm run build
python scripts/verify_boundaries.py
```

Model-backed integration checks intentionally read
`../server_configs/sample.json`; they do not copy its credentials into `app/`.
Pass a detector configuration through `XTALK_TEST_CONFIG_OVERLAY` when running
the generic detector integration check. The implementation does not contain a
detector type whitelist.

The packaged-process variant accepts an explicit frozen sidecar path:

```bash
XTALK_RUN_MODEL_TESTS=1 \
XTALK_SIDECAR_EXECUTABLE=src-tauri/binaries/app-backend-$(rustc --print host-tuple) \
python -m pytest tests/integration/test_sidecar_process.py
```

`XTALK_TEST_CONFIG_OVERLAY` remains an external JSON test input for adding a
turn detector to the sample configuration.

The voice conversation smoke also accepts `XTALK_TEST_VAD_MODEL_PATH` so a
final bundle can be tested against the exact ONNX resource installed inside
that bundle.

The model smoke suite also submits a `submit_text` turn through the authenticated
XTalk WebSocket and checks the timer contract from
`examples/sample_app/custom_async_tool.py`. The app ships an equivalent
app-owned `TimerTool` because importing the example module would execute its CLI
and asset-download side effects.

## Private model snapshots

Use `scripts/fetch_model_snapshot.py` with an explicit repository, immutable
revision, and destination. The script reads `HF_TOKEN` only from the environment
and never accepts it as a command-line argument.

```bash
python scripts/fetch_model_snapshot.py \
  --repo-id OWNER/MODEL \
  --revision FULL_COMMIT_SHA \
  --local-dir /path/to/model-snapshot
```

The fetched model is a test/build input, not a source file. Do not add model
weights or local core model implementations to this directory.

## Artifact preparation

```bash
python scripts/prepare_artifacts.py \
  --xtalk-wheel /path/to/xtalk-VERSION.whl \
  --xtalk-version VERSION \
  --client-package /path/to/xtalk-client-VERSION.tgz \
  --client-version VERSION
python scripts/verify_resources.py
```

This copies immutable artifacts into ignored `app/resources/artifacts/` and
writes an ignored SHA-256 lock manifest. The checked-in
`resources/manifests/audio-models.lock.json` separately pins and verifies the
bundled Silero ONNX resource. No core source directory is modified.

Build the target-specific sidecar from the wheel and name any required optional
dependency groups explicitly. For example, the sample configuration requires
the wheel's `ali` provider dependencies:

```bash
python scripts/build_backend.py \
  --python /path/to/python3.12 \
  --xtalk-wheel /path/to/xtalk-VERSION.whl \
  --xtalk-extra ali \
  --xtalk-extra silero-vad
```

For a build that supports managed SenseVoice and MOSS services, also stage
their target-specific native runtimes:

```bash
python scripts/prepare_managed_runtime.py \
  --sherpa-server /path/to/sherpa-onnx-offline-websocket-server \
  --sherpa-ort-library /path/to/sherpa/libonnxruntime \
  --tts-ort-library /path/to/tts/libonnxruntime
```

On Apple Silicon this also builds and stages the pinned Swift/MLX sidecar and
its Metal shader bundle; install the build-host component with
`xcodebuild -downloadComponent MetalToolchain`. CUDA builds additionally pass
`--sherpa-cuda-runtime-dir` and `--tts-cuda-runtime-dir`, each pointing to the
matching ONNX Runtime GPU library directory.

Optional weights are not bundled. Their immutable revisions, paths, sizes, and
SHA-256 values are pinned in
`resources/manifests/managed-models.lock.json` and downloaded into AppData only
when a selected configuration references the service.

The freezer collects the installed public `xtalk.models` namespace and package
data so model discovery continues to be configuration-driven. Optional
dependency groups are build inputs, not model-type branches in application
code. `silero-vad` is mandatory because every desktop launch provides it as a
top-level fallback when the selected configuration has no explicit `vad`.

## Model configuration

Release bundles contain no default XTalk model configuration or provider
credentials. On first launch the native file picker asks the user to select an
external JSON configuration. Tauri persists only its canonical path in the
application configuration directory. The **Settings and diagnostics** drawer
shows the current path and lets the user choose another file; applying a new
selection restarts the sidecar and rediscovers its loopback service.

The selected file must contain a JSON object and be no larger than 1 MiB. It
must remain available at the selected path for subsequent launches. The
frozen sidecar can instantiate only providers whose dependency groups were
included at build time.

Use [`examples/local_models.json`](examples/local_models.json) for the fully
managed local ASR/TTS configuration. Fill its empty LLM API key before use.
Tauri resolves its `managed://` values without modifying the example file. Add
`?backend=cpu`, `?backend=cuda`, or `?backend=mlx` to force a backend; without
it, selection order is CUDA, MLX, then CPU. The
[`examples/local_models_mlx.json`](examples/local_models_mlx.json) variant
forces MLX.

## Developer tool directories

The **Settings and diagnostics** drawer can install a developer tool by
selecting a directory. Tauri copies that directory into the application's data
directory and stores enablement state separately. Each selected directory must
contain an `xtalk_tool.json` file. `display_name` accepts either one string or a
language dictionary. The optional `ui` object points to self-contained HTML:

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

The entrypoint uses Python `module:factory` syntax. The zero-argument factory
must return a list containing LangChain tools, XTalk `SyncTool` or `AsyncTool`
classes, or zero-argument tool factories accepted by
`XtalkBuilder.add_agent_tools()`. The copied Python files may import packages
already included in the frozen sidecar.

Tool logic remains UI-independent. The optional HTML registers one or both
read-only display hooks:

```html
<script>
  window.xtalkToolUI.status((event) => {
    document.body.textContent = event.status;
  });
  window.xtalkToolUI.emit((event) => {
    document.body.textContent = `${event.message}\n${event.status}`;
  });
</script>
```

Calling `status()` declares a live UI; calling `emit()` declares an immutable
chat-history UI. If the entrypoint never registers one hook, the App does not
render that mode. `update_every_s` defaults to one second, accepts `-1` to
disable periodic live refresh, and is otherwise bounded from 0.1 to 3600
seconds. Each original tool emit captures its message and current status for a
history card. The HTML runs in a script-only opaque-origin sandbox; its CSP
blocks external resources and network APIs, link and form actions are
suppressed, and it has no App command capability. It cannot operate the tool.
Prepared documents use high-entropy, short-lived, one-time loopback URLs so the
App launch token never enters the frame. The App owns card width and clamps
reported height to 120–420 px for live cards and 80–600 px for history cards,
additionally capped at 60% of the window height. See
[`examples/tools/timer`](examples/tools/timer) for a complete example.

Installing, enabling, disabling, or deleting a tool updates the AppData
registry. Select **Apply and restart local service** to rebuild the configured
Agent with the enabled tools. The bundled sample-compatible timer remains a
fallback; an installed enabled tool named `timer` replaces that fallback.

## Local interface

The desktop UI follows the visual hierarchy of `examples/sample_app`: a
left conversation sidebar that starts collapsed, a context-sensitive top bar,
Orb/chat views, a bottom glass control dock, and a right
settings-and-diagnostics drawer. The top bar is empty by default; while tools
with live UI are running it shows a collapsed status summary that expands to
the current live cards. The sidebar uses the public session APIs to start a new
chat or switch among all persisted sessions. Its Tools button opens a centered
configuration dialog for installing, enabling, deleting, and applying tools.
Conversation
data remains in AppData-backed `chat_history.sqlite3`; the WebView does not
maintain a duplicate message store. Immutable custom tool UI snapshots are
stored separately in WebView AppData and keyed by the persisted session ID.
A desktop-only fixed anonymous identity
is passed through the private sidecar startup protocol, outside the public
model and service configuration, and keeps those sessions addressable after
the sidecar restarts. The launch-token and Origin boundary prevents other
clients from using that identity. It preserves the desktop adapter boundary
and provides light, dark, and narrow-window layouts. The chat view accepts text
with Enter to send and Shift+Enter for a newline. The interface automatically
selects Simplified Chinese or English from the operating-system language. The
language row in Settings & diagnostics can override that choice, and the
preference persists locally. Text is sent with the public
`Session.sendText()` API and appears only after XTalk confirms the user turn
with a matching `finish_asr` action over the session WebSocket.

Voice input remains raw PCM in the WebView. Frontend VAD and enhancement stay
disabled; the sidecar runs the packaged Silero model and emits server speech
boundaries before the configured ASR. This lets `server_configs/sample.json`
work unchanged while preserving any explicit user-provided `vad` configuration.

Architecture details are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/architecture.zh.md`](docs/architecture.zh.md).
