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
- Python 3.10 or newer (the locked sidecar dependencies are resolved for
  Python 3.12, which remains the preferred build interpreter)
- Network access to PyPI: the wheel step creates its own isolated build
  environment and installs the pinned `build` package there
- The root `frontend/` development dependencies declared by its lockfile

The installed application does not require Python, Node.js, or Rust. They are
build-time dependencies only.

The App dependency points at the repository `frontend/` package. Release builds
do not consume its existing `dist/`: the Tauri pre-build hook runs a clean
frontend install and build, creates a fresh npm package, and replaces the App's
installed client before Vite runs.

## Local checks

```bash
cd app
python -m pytest
npm ci
npm run check
npm run build
python scripts/verify_boundaries.py
```

Model-backed integration checks read the model configuration pointed to by
`XTALK_TEST_CONFIG_PATH` (defaulting to `../server_configs/sample.json`) and
skip when the file is absent; they do not copy its credentials into `app/`.
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

Normal Tauri release builds prepare these artifacts automatically from the
repository's current `src/` and `frontend/` trees. The commands below remain
available for CI jobs that intentionally supply immutable prebuilt inputs.

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
  --python /path/to/python3 \
  --xtalk-wheel /path/to/xtalk-VERSION.whl \
  --xtalk-extra ali \
  --xtalk-extra silero-vad
```

For a build that supports managed SenseVoice, Matcha, MOSS, and local
background wake, download and stage the locked native runtime for the current
platform:

```bash
python scripts/download_managed_runtime.py \
  --sherpa-keyword-spotter /path/to/sherpa-onnx-keyword-spotter-microphone \
  --sherpa-kws-model-dir /path/to/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
```

The script selects the Rust host triple, downloads the corresponding official
sherpa-onnx shared archive, verifies its locked SHA-256, and uses the Sherpa
server and ONNX Runtime 1.27 from that same archive. It then builds and stages
the App's native sidecars. The supported macOS, Linux, and Windows x64/ARM64
archives are pinned in `resources/manifests/native-runtimes.lock.json`.
The keyword-spotter executable and KWS model are explicit build inputs because
they are not part of that shared runtime archive. Later packaging runs reuse
the validated staged copies.

Optional weights are not bundled. Their immutable revisions, paths, sizes, and
SHA-256 values are pinned in
`resources/manifests/managed-models.lock.json` and downloaded into AppData only
when a selected configuration references the service.

The freezer collects the installed public `xtalk.models` namespace and package
data so model discovery continues to be configuration-driven. All Python
sidecar dependencies are constrained by `requirements/sidecar.lock`, except
`openai-codex` and its matching `openai-codex-cli-bin`; those two intentionally
resolve to the current compatible pair in the isolated build environment. The
freezer excludes `openai-codex-cli-bin` from the App bundle: the optional Codex
tool uses a user-installed CLI instead. Optional dependency groups are build
inputs, not model-type branches in application code. `silero-vad` is mandatory
because every desktop launch provides it as a top-level fallback when the
selected configuration has no explicit `vad`.

## Building the desktop app

Every package build expects a complete repository checkout in which `app/`,
`src/`, and `frontend/` are siblings. `scripts/build_from_source.py` builds a
wheel from root `src/`, runs `npm ci`, build, and pack in root `frontend/`,
updates the artifact lock, installs that fresh client for Vite, and freezes the
sidecar from `requirements/sidecar.lock` (resolved for Python 3.12).

From a clean checkout, install the toolchain prerequisites listed above (none
are installed automatically), then run the single checked entrypoint, which
verifies every prerequisite and aborts with remediation hints when any is
missing:

```bash
./scripts/build_macos_local.sh
```

The script checks the repository layout, Node.js (`^20.19.0 || >=22.12.0`),
npm, Python 3.10+, the Rust toolchain, Xcode Command Line Tools, and (on
Apple Silicon) the Metal toolchain. It never installs any of them; use
`--check-only` to run just the checks:

```bash
./scripts/build_macos_local.sh --check-only
```

When the checks pass it runs `npm ci` and `npm run package:macos:local`,
which downloads and stages the locked native runtimes and builds the local
verification installer. The wheel build runs inside a dedicated virtual
environment, so the base interpreter needs no Python packages preinstalled.
Keep Python 3.12 as the selected interpreter when possible because the
sidecar lock is resolved for it.

Apple Silicon additionally needs Xcode's Metal compiler before the first
package build; if it is missing, the script stops with the exact command to
install it (`xcodebuild -downloadComponent MetalToolchain`).

The packaging entrypoint first downloads and verifies the locked Sherpa/ORT
archive for the build host and stages all managed runtimes. It then invokes
Tauri, whose pre-build hook rebuilds the root Python and frontend sources. No
native runtime path environment variable or manual staging command is needed.

For a distributable Developer ID build, configure external Apple credentials
and run the formal entrypoint:

```bash
APPLE_SIGNING_IDENTITY="Developer ID Application: Example (TEAMID)" \
APPLE_NOTARY_KEYCHAIN_PROFILE="xtalk-release" \
npm run package:macos
```

`package:macos` is the only supported distributable macOS packaging entrypoint.
It runs Tauri, whose `beforeBuildCommand` invokes the complete source-input
build, signs the App and DMG with a Developer ID identity, smoke-tests the
frozen services, submits the installer to Apple's Notary Service, staples the
ticket, and performs final Gatekeeper assessment. It fails before building if
either external credential is absent, and removes incomplete outputs after any
later failure.

Create the notary profile once with `xcrun notarytool store-credentials`; the
profile remains in Keychain and no signing secret enters this repository. The
local command still regenerates all source inputs, seals every nested resource,
smoke-tests the frozen services, creates a signed DMG, and verifies both
outputs. It uses an ad-hoc signature and is therefore not a public release.
Direct `tauri build --bundles app` output is only an intermediate Bundle and
must not be installed or distributed on its own.

`scripts/prepare_managed_runtime.py` remains a lower-level advanced command for
an offline mirror or a custom CUDA distribution. Ordinary package builds must
use the locked automatic download path so Sherpa and ORT stay ABI-compatible.

Install the pinned frontend dependencies, verify all staged resources, and run
the release checks:

```bash
npm ci
python scripts/verify_resources.py
python -m pytest tests/unit/test_artifact_scripts.py
npm run check
cargo test --manifest-path src-tauri/Cargo.toml managed::tests --lib
```

The lower-level `python scripts/package_macos_dmg.py` command still repackages
an existing `.app`; it deliberately does not rebuild sources or protect callers
from partial output after a failure. Use `npm run package:macos` for public
release artifacts or `npm run package:macos:local` for local installation.

The Apple Silicon output is
`src-tauri/target/release/bundle/dmg/XTalk_VERSION_aarch64.dmg`. The generated
`.app` is under `src-tauri/target/release/bundle/macos/`. The local command uses
an ad-hoc signature. The formal `package:macos` result is Developer ID signed,
notarized, and stapled.

The packager rejects broken or external App-bundle symlinks and launches the
signed Matcha sidecar without build-machine environment variables. This
ensures its sherpa and ONNX Runtime dependencies resolve from
`Contents/Resources/managed-runtime/ort` before a DMG is created.

Do not replace all of `Contents/Frameworks` with links to the resource runtime.
The PyInstaller bootloader loads `libpython` from `Frameworks`, and a hardened
sidecar cannot map a differently signed copy through such a link. Before
signing, the packager verifies that the Frameworks and Resources runtime layouts
match, removes duplicate code from Resources, and retains only top-level Python
`*.dist-info` and `*.egg-info` metadata there. Frameworks metadata links to
those retained resource directories. The packager applies the sidecar's
library-validation entitlement and verifies that the signed backend can load
before creating the DMG. User-installed Codex executables remain outside the
App signature. Do not distribute a `.app` produced by the raw Tauri command
alone because it has not completed this runtime normalization and verification.

Verify the final bundle and disk image rather than only the intermediate
sidecars:

```bash
codesign --verify --deep --strict --verbose=2 \
  src-tauri/target/release/bundle/macos/XTalk.app
codesign --verify --strict --verbose=2 \
  src-tauri/target/release/bundle/dmg/XTalk_VERSION_aarch64.dmg
hdiutil verify \
  src-tauri/target/release/bundle/dmg/XTalk_VERSION_aarch64.dmg
shasum -a 256 \
  src-tauri/target/release/bundle/dmg/XTalk_VERSION_aarch64.dmg
```

The release contains ONNX Runtime 1.27, the native service executables, the
frozen Python backend, and the Silero VAD model. SenseVoice Small,
AgenticASR Refiner, matcha-icefall-zh-en, and MOSS-TTS-Nano weights are
deliberately excluded; they
are downloaded and verified in AppData only after a selected configuration
references their `managed://` URLs. Model configuration files and provider
credentials are also external and must never be copied into the bundle.

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

## Tool service credentials

Tool API keys are configured independently from tool bundles and model JSON.
The **Service credentials** section in **Settings and diagnostics** persists a
key only in the operating system credential manager: macOS Keychain, Windows
Credential Manager, or Linux Secret Service. There is deliberately no
session-only credential mode. On platforms without an accessible credential
service, provide the documented environment variable before launching XTalk.

Environment variables have higher precedence than stored credentials and are
shown as read-only in the UI. The Serper-backed Web Search built-in accepts
`SERPER_API_KEY` or `GOOGLE_SERPER_API_KEY`; the resolved value is injected as
`SERPER_API_KEY` only into the managed Python sidecar process. Secret values
never enter `xtalk_tool.json`, `resources/credentials.json`, AppData, the
sidecar startup JSON, command-line arguments, or diagnostic output.

`resources/credentials.json` is App-owned metadata that binds a stable
credential ID to supported environment variables and the built-in tool that
consumes it. Adding a credential-backed built-in requires updating this
registry; tool implementation files remain free of storage and platform
branches. Saving, replacing, or deleting a credential requires **Apply and
restart local service** before a running Agent observes the new environment.

Use
[`examples/local_models_moss_tts.json`](examples/local_models_moss_tts.json)
for the fully managed local ASR/TTS configuration. Fill its empty LLM API key
before use.
Tauri resolves its `managed://` values without modifying the example file. Add
`?backend=cpu`, `?backend=cuda`, or `?backend=mlx` to force a backend; without
it, Apple Silicon macOS selects MLX, CUDA-capable Windows/Linux selects CUDA,
and other systems select CPU. Use
[`examples/local_models_agentic_asr.json`](examples/local_models_agentic_asr.json)
to have Tauri download and start both SenseVoice and the AgenticASR Refiner. The
[`examples/local_models_mlx.json`](examples/local_models_mlx.json) variant
forces MLX. Use
[`examples/local_models_matcha.json`](examples/local_models_matcha.json) to
select Matcha Chinese-English TTS. Matcha accepts CPU and CUDA backends; it does
not select MLX. The
[`examples/local_models_qwen3_asr_0_6b_int8.json`](examples/local_models_qwen3_asr_0_6b_int8.json)
example selects the managed Qwen3-ASR 0.6B INT8 snapshot. Its `auto` backend
prefers Core ML on macOS and CUDA where available, then falls back to the native
ARM/x64 CPU provider. Use `?backend=coreml`, `?backend=cuda`, or `?backend=cpu`
to force a Qwen backend; Qwen does not accept `backend=mlx`.

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
    document.body.textContent = `${event.outcome}\n${event.message}\n${event.status}`;
  });
</script>
```

The App exposes the resolved interface language and frame mode through a
read-only runtime context. Tool HTML should use this value instead of
`navigator.language`, because the user can select a language independently of
the operating system:

```html
<script>
  const { language, mode } = window.xtalkToolUI.context;
  const copy = language.startsWith("zh")
    ? { title: "计时器", running: "运行中" }
    : { title: "Timer", running: "Running" };
</script>
```

`language` is currently `zh-CN` or `en`. Changing the App language recreates
open tool UI frames with the new context. `mode` is `live` for the collapsible
top status area and `history` for an immutable chat-history card.

Calling `status()` declares a live UI; calling `emit()` declares an immutable
chat-history UI. If the entrypoint never registers one hook, the App does not
render that mode. `update_every_s` defaults to one second, accepts `-1` to
disable periodic live refresh, and is otherwise bounded from 0.1 to 3600
seconds. Each original tool emit captures its message and current status for a
history card. A terminal emit replaces earlier history cards from the same
tool call, so completed or cancelled tools do not leave stale Running cards
behind. Emit events include `outcome` as `running`, `complete`, or `cancelled`;
cancelled calls receive one final history emit and are removed from live UI.
The HTML runs in a script-only opaque-origin sandbox; its CSP
blocks external resources and network APIs, link and form actions are
suppressed, and it has no App command capability. It cannot operate the tool.
Prepared documents use high-entropy, runtime-scoped loopback URLs that remain
idempotently readable for WKWebView reloads. The capacity-bounded in-memory
store disappears with the backend process, and the App launch token never
enters the frame. The App owns card width and clamps
reported height to 120–420 px for live cards and 80–600 px for history cards,
additionally capped at 60% of the window height. See
[`examples/tools/timer`](examples/tools/timer) for a complete example.

User-tool install, enable, disable, and delete operations update
`AppData/tools/registry.json`. Built-ins are indexed by
`resources/tools/builtin_tools.json`; they cannot be deleted, and optional
built-ins persist enabled overrides in `AppData/tool_preferences.json`.
The Current Time built-in is required and cannot be disabled. Web Search is an
ordinary optional built-in, disabled by default, and can be enabled only when
its external credential resolves successfully. Select **Apply and restart
local service** to rebuild the configured Agent with enabled tools.
When a user tool exports the same name as a built-in, the user implementation
takes precedence.

The optional **Codex** built-in is one atomic bundle and is disabled by
default. Its single toggle enables or disables
`codex_session_search`, `codex_session_create`,
`codex_session_continue`, `codex_models_list`,
`codex_session_set_model`, and `codex_session_delete` together. The model-list
operation queries the authenticated SDK catalog at call time and returns each
visible model's supported reasoning efforts; model changes must use an ID from
that current result and are validated against the SDK again. The first real task creates a persistent
official Codex SDK thread; later turns always reapply the session's saved
model, reasoning effort, working directory, and `Sandbox.full_access`. It also
uses the SDK's no-prompt approval mode, so arbitrary existing local directories
are accepted as `cwd` values without an App approval step. Enable this bundle
only when unrestricted local Codex access is intended.

The Codex built-in does not ship a CLI. On first use it checks the inherited
`PATH` and common Homebrew, npm, nvm, fnm, Volta, Bun, and system installation
locations. Every candidate must execute `codex --version` successfully before
the SDK receives its absolute path. The first valid location is cached for the
sidecar process lifetime; discovery runs again only after that executable path
disappears or is no longer executable. When none is usable, the tool asks the
user to run
`npm install -g @openai/codex` and restart XTalk. An npm shim's directory is
prepended to the child `PATH` so `/usr/bin/env node` resolves the matching Node
installation even when XTalk was launched from Finder.

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
configuration dialog that groups built-in and user tools. Optional tools in
both groups can be enabled or disabled; only user tools can be deleted.
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

Background voice wake is opt-in. When enabled, closing the main window hides it
to the system tray instead of exiting. Tauri supervises the packaged
`sherpa-onnx-keyword-spotter-microphone` process and listens locally for the
user-editable wake phrase, which defaults to `你好小克`; no wake audio is stored
or uploaded. Its persisted acoustic trigger threshold defaults to `0.05` and is
editable in the same settings section. Chinese phrases are converted to
tone-marked pinyin tokens and English phrases use the packaged `en.phone`
lexicon. The generated keyword file is stored in AppData. A detection stops the
keyword spotter before showing the
window and opening a fresh XTalk session, so the
existing proactive Agent greeting starts the conversation without simultaneous
microphone ownership. Ending the conversation resumes keyword detection. Use
the tray Quit action for a full sidecar and application shutdown.

Architecture details are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/architecture.zh.md`](docs/architecture.zh.md).
