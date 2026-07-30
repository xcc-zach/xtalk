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

## Developer tool directories

The **Settings and diagnostics** drawer can install a developer tool by
selecting a directory. Tauri copies that directory into the application's data
directory and stores enablement state separately. Each selected directory must
contain an `xtalk_tool.json` file with exactly two fields:

```json
{
  "display_name": "Timer",
  "entrypoint": "timer_tool:create_tools"
}
```

The entrypoint uses Python `module:factory` syntax. The zero-argument factory
must return a list containing LangChain tools, XTalk `SyncTool` or `AsyncTool`
classes, or zero-argument tool factories accepted by
`XtalkBuilder.add_agent_tools()`. The copied Python files may import packages
already included in the frozen sidecar.

Installing, enabling, disabling, or deleting a tool updates the AppData
registry. Select **Apply and restart local service** to rebuild the configured
Agent with the enabled tools. The bundled sample-compatible timer remains a
fallback; an installed enabled tool named `timer` replaces that fallback.

## Local interface

The desktop UI follows the visual hierarchy of `examples/sample_app`: a
centered brand bar, Orb/chat views, a bottom glass control dock, and a right
settings-and-diagnostics drawer. It preserves the desktop adapter boundary and
provides light, dark, and narrow-window layouts. The chat view accepts text
with Enter to send and Shift+Enter for a newline. Text is sent with the public
`Session.sendText()` API and appears only after XTalk confirms the user turn
with a matching `finish_asr` action over the session WebSocket.

Voice input remains raw PCM in the WebView. Frontend VAD and enhancement stay
disabled; the sidecar runs the packaged Silero model and emits server speech
boundaries before the configured ASR. This lets `server_configs/sample.json`
work unchanged while preserving any explicit user-provided `vad` configuration.

Architecture details are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/architecture.zh.md`](docs/architecture.zh.md).
