# XTalk Desktop native shell

The Tauri shell requires prepared, target-specific sidecar artifacts before
`cargo check`, `tauri dev`, or `tauri build`:

```text
app/src-tauri/binaries/
├── app-backend-<target-triple>[.exe]
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

## Development config

Release builds always send the bundled `config/default.json` path to the
sidecar. Debug builds can read another config without copying or modifying it:

```sh
XTALK_APP_CONFIG_PATH=/absolute/path/to/server_configs/sample.json npm run tauri dev
```

Every launch also resolves the bundled `models/audio/silero_vad.onnx` resource
and sends a top-level VAD fallback to the sidecar. The selected configuration
wins when it already declares `vad`; otherwise the fallback loads `SileroVAD`
from the packaged absolute model path.

The override path and launch token are sent to the sidecar only in its first
stdin JSON line. They are never added to process arguments or diagnostic
messages.

`Info.plist` and `Entitlements.plist` provide the macOS microphone usage
description and hardened-runtime audio-input entitlement. The WebView requests
microphone access only when the user starts a voice conversation.
