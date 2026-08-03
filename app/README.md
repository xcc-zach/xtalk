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
`resources/artifacts/xtalk-client-0.2.8.tgz`. Prepare that ignored artifact from
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
`examples/sample_app/custom_async_tool.py`. The app ships an equivalent timer
under `resources/tools/timer`; it is discovered through the same manifest
protocol as a user-installed tool.

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
data so model discovery continues to be configuration-driven. It also collects
the official `openai-codex` package and its pinned `openai-codex-cli-bin`
runtime for the optional Codex built-in. Optional dependency groups are build
inputs, not model-type branches in application
code. `silero-vad` is mandatory because every desktop launch provides it as a
top-level fallback when the selected configuration has no explicit `vad`.

## Building the desktop app

The following sequence produces a self-contained desktop bundle. Run the
commands from `app/` unless a command explicitly changes directory. Build
inputs must match the target architecture; for example, an Apple Silicon build
needs arm64 executables and libraries.

First build a fresh XTalk wheel from the repository root. Do not reuse an older
wheel: `build_backend.py` rejects wheels that do not contain the managed
SenseVoice and MOSS client modules.

```bash
cd ..
python3 -m build --wheel --outdir /tmp/xtalk-dist
cd app
```

If `python3 -m build` is unavailable, install the Python `build` package in the
build environment. Prepare the immutable core artifacts once per wheel or
client-package update. Replace the paths and version strings with the exact
files being packaged:

```bash
python scripts/prepare_artifacts.py \
  --xtalk-wheel /tmp/xtalk-dist/xtalk-VERSION-py3-none-any.whl \
  --xtalk-version VERSION \
  --client-package /path/to/xtalk-client-VERSION.tgz \
  --client-version VERSION
```

Freeze the Python application backend with every provider dependency required
by the configurations that the release must support. The current sample and
managed-local-model configurations require `ali` and the mandatory
`silero-vad` extra:

```bash
python scripts/build_backend.py \
  --python /path/to/python3.12 \
  --xtalk-wheel /tmp/xtalk-dist/xtalk-VERSION-py3-none-any.whl \
  --xtalk-extra ali \
  --xtalk-extra silero-vad
```

Stage the managed-model runtimes. The sherpa server, sherpa ONNX Runtime
library, and TTS ONNX Runtime library must all target the same platform as the
App. The script builds the Rust MOSS runtime itself. On Apple Silicon it also
builds the Swift/MLX runtime and copies its Metal resources:

```bash
python scripts/prepare_managed_runtime.py \
  --sherpa-server /path/to/sherpa-onnx-offline-websocket-server \
  --sherpa-ort-library /path/to/sherpa/libonnxruntime.dylib \
  --tts-ort-library /path/to/tts/libonnxruntime.dylib
```

Before the first Apple Silicon build, install the Xcode Metal toolchain
component:

```bash
xcodebuild -downloadComponent MetalToolchain
```

CUDA release artifacts additionally pass
`--sherpa-cuda-runtime-dir /path/to/sherpa/onnxruntime-gpu/lib` and
`--tts-cuda-runtime-dir /path/to/tts/onnxruntime-gpu/lib`. Those directories
must contain the matching CUDA and shared ONNX Runtime provider libraries.

Install the pinned frontend dependencies, verify all staged resources, and run
the release checks:

```bash
npm ci
python scripts/verify_resources.py
python -m pytest tests/unit/test_artifact_scripts.py
npm run check
cargo test --manifest-path src-tauri/Cargo.toml managed::tests --lib
```

Build the macOS application with Tauri, then use the repository packager to
normalize the PyInstaller framework layout, sign it, smoke-test the frozen
backend, and create the disk image:

```bash
npm run tauri build -- --bundles app
python scripts/package_macos_dmg.py
```

The Apple Silicon output is
`src-tauri/target/release/bundle/dmg/XTalk_VERSION_aarch64.dmg`. The generated
`.app` is under `src-tauri/target/release/bundle/macos/`. A locally built image
uses an ad-hoc signature. Set `APPLE_SIGNING_IDENTITY` to an Apple Developer
identity before running `package_macos_dmg.py` when producing a distributable
release. Notarization remains a separate release step.

Do not replace all of `Contents/Frameworks` with links to the resource runtime.
The PyInstaller bootloader loads `libpython` from `Frameworks`, and a hardened
sidecar cannot map a differently signed copy through such a link. The packager
links only top-level Python `*.dist-info` metadata directories, applies the
sidecar's library-validation entitlement, explicitly signs both bundled Codex
code-mode hosts with the V8 executable-memory entitlements from
`src-tauri/CodexHostEntitlements.plist`, and verifies that the signed backend
can load before creating the DMG. Do not distribute a `.app` produced by the
raw Tauri command alone: its Codex host will fail when V8 creates the first
isolate.

Verify the final bundle and disk image rather than only the intermediate
sidecars:

```bash
codesign --verify --deep --strict --verbose=2 \
  src-tauri/target/release/bundle/macos/XTalk.app
hdiutil verify \
  src-tauri/target/release/bundle/dmg/XTalk_VERSION_aarch64.dmg
shasum -a 256 \
  src-tauri/target/release/bundle/dmg/XTalk_VERSION_aarch64.dmg
```

The release contains ONNX Runtime, the native service executables, the frozen
Python backend, and the Silero VAD model. SenseVoice Small and MOSS-TTS-Nano
weights are deliberately excluded; they are downloaded and verified in
AppData only after a selected configuration references their `managed://`
URLs. Model configuration files and provider credentials are also external and
must never be copied into the bundle.

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

## Built-in and user tool directories

Built-in tools under `resources/tools/` and user tools installed from
**Settings and diagnostics** use the same `xtalk_tool.json` schema. Tauri copies
only user-selected directories into AppData. `display_name` accepts either one
string or a language dictionary. The optional `ui` object points to
self-contained HTML:

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

User-tool install, enable, disable, and delete operations update
`AppData/tools/registry.json`. Built-ins are indexed by
`resources/tools/builtin_tools.json`; they cannot be deleted, but their enabled
overrides are persisted in `AppData/tool_preferences.json`. Select **Apply and
restart local service** to rebuild the configured Agent with enabled tools.
When a user tool exports the same name as a built-in, the user implementation
takes precedence.

The optional **Codex** built-in is one atomic bundle and is disabled by
default. Its single toggle enables or disables
`codex_session_search`, `codex_session_create`,
`codex_session_continue`, `codex_session_set_model`, and
`codex_session_delete` together. The first real task creates a persistent
official Codex SDK thread; later turns always reapply the session's saved
model, reasoning effort, working directory, and `Sandbox.full_access`. It also
uses the SDK's no-prompt approval mode, so arbitrary existing local directories
are accepted as `cwd` values without an App approval step. Enable this bundle
only when unrestricted local Codex access is intended.

The App keeps only its thread index and compact routing metadata in
`AppData/tool-data/codex/codex_sessions.sqlite3`; the Codex SDK remains the
owner of thread history. Natural-language session lookup sends at most 30
App-indexed candidates to a temporary ephemeral Codex thread and validates
that its structured result contains only candidate IDs. Deletion first calls
the SDK archive operation and then removes the thread from the active App pool.
Its custom HTML is display-only, contains no controls, and deliberately does
not show the full-access label in live UI.

## Local interface

The desktop UI follows the visual hierarchy of `examples/sample_app`: a
left conversation sidebar that starts collapsed, a context-sensitive top bar,
Orb/chat views, a bottom glass control dock, and a right
settings-and-diagnostics drawer. The top bar is empty by default; while tools
with live UI are running it shows a collapsed status summary that expands to
the current live cards. The sidebar uses the public session APIs to start a new
chat or switch among all persisted sessions. Its Tools button opens a centered
configuration dialog that groups built-in and user tools. Both groups can be
enabled or disabled; only user tools can be deleted.
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

The WebView requests browser-native echo cancellation, automatic gain control,
and noise suppression for microphone capture. Frontend VAD and the custom
FastEnhancer stay disabled; the sidecar runs the packaged Silero model and emits
server speech boundaries before the configured ASR. This lets
`server_configs/sample.json` work unchanged while preserving any explicit
user-provided `vad` configuration.

Architecture details are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/architecture.zh.md`](docs/architecture.zh.md).
