# XTalk Desktop native shell

The Tauri shell requires prepared, target-specific sidecar artifacts before
`cargo check`, `tauri dev`, or `tauri build`:

```text
app/src-tauri/binaries/
├── app-backend-<target-triple>[.exe]
├── local-model-runtime-<target-triple>[.exe]
├── sherpa-onnx-offline-websocket-server-<target-triple>[.exe]
└── app-backend-runtime/
    └── PyInstaller onedir runtime files
```

Do not add a placeholder executable. The app build scripts must produce the
real `app-backend` binary for the active target triple and the complete
PyInstaller `onedir` runtime before invoking Tauri.

Build it from an immutable wheel and list any provider extras required by the
selected acceptance configuration:

```sh
python ../scripts/build_backend.py \
  --python /path/to/python3.12 \
  --xtalk-wheel /path/to/xtalk-VERSION.whl \
  --xtalk-extra ali \
  --xtalk-extra silero-vad
```

Prepare the optional managed native runtimes from explicit platform files:

```sh
python ../scripts/prepare_managed_runtime.py \
  --sherpa-server /path/to/sherpa-onnx-offline-websocket-server \
  --sherpa-ort-library /path/to/sherpa/libonnxruntime \
  --tts-ort-library /path/to/tts/libonnxruntime
```

The script builds the Rust MOSS service, gives the executables Tauri's
target-specific names, and stages the two ABI-specific ONNX Runtime libraries
as ignored resources. Apple Silicon builds also compile the pinned Swift MLX
service through Xcode and stage its Metal resource bundle. CUDA packages pass
the two optional `--*-cuda-runtime-dir` arguments so the matching execution
provider libraries are staged. A release build must run it for every target.
The Apple Silicon build host must have Xcode's Metal Toolchain component
installed (`xcodebuild -downloadComponent MetalToolchain`).

`tauri.conf.json` bundles the runtime directory as
`$RESOURCE/app-backend-runtime` and launches the external binary with
the external-binary directory as its working directory. Tauri places resources
beside the executable on Windows and in Cargo development builds. On macOS,
`bundle.macOS.files` also places the runtime contents directly in
`Contents/Frameworks`, which is where the PyInstaller bootloader resolves them
inside an app bundle. Linux AppImage, Debian, and RPM settings copy the runtime
to `/usr/bin/app-backend-runtime`, beside the external binary.

PyInstaller `onedir` resolves its contents relative to its executable, not only
its working directory. The Rust launcher therefore refuses to start when the
platform runtime layout is incomplete: a sibling `app-backend-runtime` for
ordinary targets, or `Contents/Frameworks` on macOS. Every installer smoke test
must still verify this layout in the final signed package; in particular, the
Linux destination assumes Tauri installs the external sidecar at
`/usr/bin/app-backend` and remains a build acceptance item until each selected
bundle target is inspected.

## User-selected model configuration

Release bundles do not contain a default XTalk model configuration. On first
launch the WebView opens the native JSON file picker. After the user selects a
configuration, Tauri validates that it is a JSON object no larger than 1 MiB,
starts the sidecar, and persists only the canonical external file path under
the application configuration directory. Changing the selection stops the old
sidecar, starts a new one, and lets the WebView rediscover its loopback
endpoint.

Debug builds may use an initial external configuration without going through
the picker:

```sh
XTALK_APP_CONFIG_PATH=/absolute/path/to/server_configs/sample.json npm run tauri dev
```

Every launch also resolves the bundled `models/audio/silero_vad.onnx` resource
and sends a top-level VAD fallback to the sidecar. The selected configuration
wins when it already declares `vad`; otherwise the fallback loads `SileroVAD`
from the packaged absolute model path.

The selected path and launch token are sent to the sidecar only in its first
stdin JSON line. They are never added to process arguments or diagnostic
messages. Configuration contents, including provider credentials, remain in
the user-selected file and are not copied into the application bundle or the
selection record.

The example
[`../examples/local_models.json`](../examples/local_models.json) selects both
optional managed services. `managed://` is a desktop-only locator: Tauri
downloads and verifies the pinned snapshot under AppData, starts the native
service, and replaces the locator with an ephemeral loopback URL in the Python
startup overlay. `?backend=cpu|cuda|mlx` forces a backend; otherwise Tauri
chooses CUDA, then MLX, then CPU. The external JSON is never rewritten.

`Info.plist` and `Entitlements.plist` provide the macOS microphone usage
description and hardened-runtime audio-input entitlement. The WebView requests
microphone access only when the user starts a voice conversation.
