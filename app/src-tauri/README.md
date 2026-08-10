# XTalk Desktop native shell

The Tauri shell requires prepared, target-specific sidecar artifacts before
`cargo check` or `tauri dev`. The supported App packaging entrypoints download
and verify the locked Sherpa/ORT archive before invoking Tauri. A Tauri build
then prepares the Python sidecar automatically from the repository's current
root `src/` tree through the configured `beforeBuildCommand`:

```text
app/src-tauri/binaries/
├── app-backend-<target-triple>[.exe]
├── local-model-runtime-<target-triple>[.exe]
├── sherpa-onnx-offline-websocket-server-<target-triple>[.exe]
├── sherpa-onnx-keyword-spotter-microphone-<target-triple>[.exe]
└── app-backend-runtime/
    └── PyInstaller onedir runtime files
```

Do not add a placeholder executable. The app build scripts must produce the
real `app-backend` binary for the active target triple and the complete
PyInstaller `onedir` runtime before invoking Tauri.

For an ordinary public release, run `npm run package:macos` from `app/` with
`APPLE_SIGNING_IDENTITY` and `APPLE_NOTARY_KEYCHAIN_PROFILE` configured. Use
`npm run package:macos:local` for a fully sealed local installer. A direct
`tauri build --bundles app` is an intermediate Bundle that has not completed
the repository's nested signing, smoke-test, DMG, and notarization pipeline.
Use the lower-level command only when intentionally building a sidecar from an
immutable wheel supplied by CI:

```sh
python ../scripts/build_backend.py \
  --python /path/to/python3.12 \
  --xtalk-wheel /path/to/xtalk-VERSION.whl \
  --xtalk-extra ali \
  --xtalk-extra silero-vad
```

Prepare the optional managed native runtimes for the current platform:

```sh
python ../scripts/download_managed_runtime.py \
  --sherpa-keyword-spotter /path/to/sherpa-onnx-keyword-spotter-microphone \
  --sherpa-kws-model-dir /path/to/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
```

The script selects the Rust host target, downloads its official sherpa shared
distribution, validates the SHA-256 from
`resources/manifests/native-runtimes.lock.json`, and stages the included
Sherpa server and ONNX Runtime 1.27 together. It also builds the Rust runtimes;
Apple Silicon builds compile the pinned Swift MLX service and stage its Metal
resource bundle. The Apple Silicon host must have Xcode's Metal Toolchain
component installed (`xcodebuild -downloadComponent MetalToolchain`).
The keyword spotter and KWS model are supplied explicitly once and remain as
ignored staged artifacts for subsequent packaging runs.

The staging step also packages the local keyword spotter and the fixed
`你好小克` KWS model. On Windows, `tauri.windows.conf.json` places the shared
ONNX Runtime DLL beside the packaged sidecars so it takes precedence over an
incompatible system copy.

`tauri.conf.json` bundles the runtime directory as
`$RESOURCE/app-backend-runtime` and launches the external binary with
the external-binary directory as its working directory. Tauri places resources
beside the executable on Windows and in Cargo development builds. On macOS,
`bundle.macOS.files` also places the runtime contents directly in
`Contents/Frameworks`, which is where the PyInstaller bootloader resolves them
inside an app bundle. The supported macOS packager validates both generated
copies before signing, keeps Frameworks as the sole complete runtime, and
retains only Python package metadata under Resources. Linux AppImage, Debian,
and RPM settings copy the runtime to `/usr/bin/app-backend-runtime`, beside the
external binary.

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

Tool service credentials use a separate App-owned registry at
`resources/credentials.json`. The native layer persists them through macOS
Keychain, Windows Credential Manager, or Linux Secret Service; supported
environment variables take precedence. Only credential status crosses back to
the WebView. Resolved secrets are added to the Python child environment during
start or restart and never enter the startup JSON, AppData, arguments, or logs.
The packaged registry contains bindings and environment names only, never key
values. A platform without an accessible credential service must use the
corresponding environment variable.

The example
[`../examples/local_models_moss_tts.json`](../examples/local_models_moss_tts.json)
selects both optional managed services. `managed://` is a desktop-only locator: Tauri
downloads and verifies the pinned snapshot under AppData, starts the native
service, and replaces the locator with an ephemeral loopback URL in the Python
startup overlay. `?backend=cpu|cuda|mlx` forces a backend; otherwise Tauri
chooses CUDA, then MLX, then CPU. The external JSON is never rewritten.

`Info.plist` and `Entitlements.plist` provide the macOS microphone usage
description and hardened-runtime audio-input entitlement. The WebView requests
microphone access when the user starts a voice conversation; the packaged local
keyword spotter also uses it while the user has explicitly enabled background
voice wake.

## Adding a Tauri command

Every command exposed to the WebView must be registered in **four** places.
Omitting any of them fails at runtime; the most common symptom is:

```text
Command <name> not allowed by ACL
```

1. `src/lib.rs` — add the `#[tauri::command]` function and include its name in
   the `invoke_handler(tauri::generate_handler![...])` list.
2. `build.rs` — add the command name to `APP_COMMANDS` so Tauri generates its
   permission identifiers in `gen/schemas/desktop-schema.json`.
3. `capabilities/main.json` — add `allow-<kebab-case-command-name>` to
   `permissions`. This entry is the ACL gate and is the one most often
   forgotten: the ACL error above still appears when steps 1, 2, and 4 are
   correct.
4. `../ui/adapters/native-capabilities.ts` — add the command constant and the
   invoke wrapper used by the WebView.

After changing any of these files, rebuild with `npm run package:macos:local`
so the generated schema and the installed bundle include the new permission.
