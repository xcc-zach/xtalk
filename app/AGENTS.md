# Incremental macOS App Installation

Use the smallest incremental scope that contains every changed runtime artifact.
These workflows are for local `/Applications/XTalk.app` iteration only. Public
release validation must still use `npm run package:macos` or
`npm run package:macos:local` as appropriate.

## Required Command

Run the incremental installer from `app/`:

```sh
npm run install:macos:incremental -- <scope>
```

The installer builds before stopping XTalk, replaces only the selected App
component, applies the repository's ad-hoc signature, verifies the complete App
seal, and relaunches XTalk. Pass `--no-launch` to leave it stopped.

Never modify a running App bundle manually. Every direct change below
`XTalk.app/Contents` invalidates the outer macOS signature and must be followed
by the repository signing and verification flow.

## Scope Selection

### `ui`

Use for changes limited to `app/ui/**` or the App's Vite presentation layer.
It runs the App Vite build, performs an incremental release Cargo build, and
replaces only `Contents/MacOS/xtalk-desktop`.

```sh
npm run install:macos:incremental -- ui
```

### `frontend`

Use for changes under the repository-level `frontend/**` package. It rebuilds
`xtalk-client`, installs that local package into the App without changing the
lockfile, builds the App UI, performs an incremental release Cargo build, and
replaces only `xtalk-desktop`.

```sh
npm run install:macos:incremental -- frontend
```

### `shell`

Use for Rust-only changes under `app/src-tauri/src/**`, `app/src-tauri/build.rs`,
capabilities, or permissions when the bundle layout is unchanged. It reuses the
existing Vite output and Cargo cache and replaces only `xtalk-desktop`.

```sh
npm run install:macos:incremental -- shell
```

If a Rust change also depends on new UI output, use `ui` instead. If it depends
on a changed repository-level client package, use `frontend`.

### `resources`

Use for static bundled changes under `app/resources/tools/**`, the bundled
licenses, credentials registry, managed-model manifest, Silero VAD model, or
the three model examples declared in `tauri.conf.json`. It performs no compiler
build and replaces only the corresponding `Contents/Resources` files.

```sh
npm run install:macos:incremental -- resources
```

Changing an example does not modify an already selected external model config.
Update or reselect that external JSON separately when runtime behavior must use
the new value.

### `icon`

Use only for the native macOS `app/src-tauri/icons/icon.icns`. It updates both
the conventional resource and the versioned `icon-v3.icns` cache key. Changes
to `app/ui/assets/app-icon.svg` belong to `ui`. Finder or Launchpad may retain
an icon cache after the App bundle itself has been updated correctly.

```sh
npm run install:macos:incremental -- icon
```

### `app-backend`

Use for Python-only changes under `app/backend/**`. It reuses the previously
installed XTalk wheel, cached backend virtual environment, and PyInstaller work
tree when available. It replaces only `Contents/MacOS/app-backend` because the
Python module archive is embedded in that executable.

```sh
npm run install:macos:incremental -- app-backend
```

### `core-backend`

Use for pure Python changes under repository-level `src/xtalk/**` when package
dependencies and collected non-Python data are unchanged. It builds a fresh
wheel, updates the cached PyInstaller environment, incrementally freezes the
backend, and replaces only `app-backend`.

```sh
npm run install:macos:incremental -- core-backend
```

### `backend-runtime`

Use when backend dependencies, `pyproject.toml`,
`app/requirements/sidecar.lock`, native Python extensions, or collected package
data changed. It performs the clean locked backend build and replaces both the
backend executable and the complete PyInstaller runtime under
`Contents/Frameworks` and `Contents/Resources/app-backend-runtime`.

```sh
npm run install:macos:incremental -- backend-runtime
```

Use `--xtalk-wheel /absolute/path/to/xtalk.whl` only when intentionally testing
an immutable wheel. Use repeated `--xtalk-extra NAME` arguments only when the
runtime requires a non-default extra set; the default is `ali` plus
`silero-vad`.

## Cases That Must Use Full Packaging

Do not use the incremental installer for changes to bundle identity, version,
`Info.plist` generation, external binary declarations, resource destinations,
macOS entitlements, installer behavior, signing behavior, or DMG layout. Use:

```sh
npm run package:macos:local
```

Also use the full packaging flow before handing an App to another person. The
incremental installer intentionally does not create or validate a DMG and does
not run every frozen-service smoke test.

## Native Managed Runtimes

Changes under `app/local-model-runtime/**`, `app/matcha-model-runtime/**`,
`app/mtd-model-runtime/**`, or `app/local-model-runtime-mlx/**` are separate
native sidecar changes. The current incremental installer does not rebuild
those specialized targets because Matcha, MTD, and MLX require staged external
native inputs. Rebuild the affected runtime through
`scripts/download_managed_runtime.py`, then use the full local package flow.

Do not use `ui`, `shell`, or either backend scope for a native managed-runtime
change; those scopes cannot update its executable or Metal resources.

## Configuration-Only Changes

External files such as `server_configs/*.json` are not bundled build inputs.
They require no App rebuild. Restart the sidecar or XTalk after updating the
currently selected external configuration.
