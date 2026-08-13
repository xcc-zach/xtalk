#!/usr/bin/env python3
"""Build and install the smallest changed XTalk macOS component."""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from package_macos_dmg import (
    link_python_metadata_to_resources,
    sign_app,
    verify_internal_bundle_links,
)


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parent
FRONTEND_ROOT = REPOSITORY_ROOT / "frontend"
DEFAULT_INSTALLED_APP = Path("/Applications/XTalk.app")
BACKEND_BUILD_ROOT = APP_ROOT / ".build" / "backend"
INCREMENTAL_BUILD_ROOT = APP_ROOT / ".build" / "incremental"
DEFAULT_XTALK_EXTRAS = ("ali", "silero-vad")
SCOPES = (
    "ui",
    "frontend",
    "shell",
    "resources",
    "icon",
    "app-backend",
    "core-backend",
    "backend-runtime",
)
IGNORED_RESOURCE_NAMES = ("__pycache__", "*.pyc", ".DS_Store")
PYTHON_METADATA_SUFFIXES = (".dist-info", ".egg-info")


def run(
    command: list[str],
    *,
    cwd: Path = APP_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one incremental build or installation command.

    Parameters
    ----------
    command : list[str]
        Executable and argument vector.
    cwd : pathlib.Path, optional
        Working directory for the command.
    check : bool, optional
        Whether a non-zero exit status raises an exception.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed child process.
    """

    print("+", " ".join(command))
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def parse_args() -> argparse.Namespace:
    """Parse the incremental installation request.

    Returns
    -------
    argparse.Namespace
        Selected scope and local installation options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument(
        "--app",
        type=Path,
        default=DEFAULT_INSTALLED_APP,
        help="installed XTalk.app to update",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Python 3.12 interpreter used for backend builds",
    )
    parser.add_argument(
        "--xtalk-wheel",
        type=Path,
        help="existing XTalk wheel for app-backend or backend-runtime",
    )
    parser.add_argument(
        "--xtalk-extra",
        action="append",
        default=[],
        help="XTalk wheel extra passed to a clean backend-runtime build",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="leave XTalk stopped after installation",
    )
    return parser.parse_args()


def require_installed_app(app: Path) -> Path:
    """Validate and return the installed App bundle.

    Parameters
    ----------
    app : pathlib.Path
        Requested local App bundle.

    Returns
    -------
    pathlib.Path
        Absolute validated App path.

    Raises
    ------
    FileNotFoundError
        Raised when the App bundle is absent.
    ValueError
        Raised when the path is not an ``.app`` directory.
    """

    resolved = app.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if resolved.suffix != ".app":
        raise ValueError("--app must point to an existing .app bundle")
    return resolved


def resolve_python(explicit: Path | None) -> Path:
    """Resolve a Python interpreter for a backend build.

    Parameters
    ----------
    explicit : pathlib.Path | None
        Optional caller-selected interpreter.

    Returns
    -------
    pathlib.Path
        Existing Python executable, preferring Python 3.12.

    Raises
    ------
    RuntimeError
        Raised when no Python interpreter is available.
    """

    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)
    for name in ("python3.12", "python3"):
        executable = shutil.which(name)
        if executable:
            return Path(executable).resolve()
    raise RuntimeError("backend installation requires Python 3.12 or python3")


def resolve_target_triple() -> str:
    """Return the active Rust host target triple.

    Returns
    -------
    str
        Rust host target used by staged Tauri sidecars.
    """

    result = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        check=True,
        capture_output=True,
        text=True,
    )
    target = result.stdout.strip()
    if not target:
        raise RuntimeError("rustc returned an empty host target")
    return target


def atomic_copy_file(source: Path, destination: Path) -> None:
    """Atomically replace one installed file with a built artifact.

    Parameters
    ----------
    source : pathlib.Path
        Existing artifact to install.
    destination : pathlib.Path
        Exact file path inside the installed App.
    """

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def replace_directory(source: Path, destination: Path) -> None:
    """Replace one installed resource directory through a staged sibling.

    Parameters
    ----------
    source : pathlib.Path
        Source directory to copy.
    destination : pathlib.Path
        Exact directory inside the installed App.
    """

    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    staged = staging_root / destination.name
    backup = destination.parent / f".{destination.name}.incremental-backup"
    try:
        shutil.copytree(
            source,
            staged,
            ignore=shutil.ignore_patterns(*IGNORED_RESOURCE_NAMES),
        )
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        elif backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staged, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def resource_file_mappings() -> tuple[tuple[Path, Path], ...]:
    """Return static source-to-bundle resource mappings.

    Returns
    -------
    tuple[tuple[pathlib.Path, pathlib.Path], ...]
        App-relative source and bundle-relative destination pairs.
    """

    return (
        (Path("resources/credentials.json"), Path("credentials.json")),
        (
            Path("resources/manifests/managed-models.lock.json"),
            Path("manifests/managed-models.lock.json"),
        ),
        (
            Path("resources/licenses/silero-vad-LICENSE.txt"),
            Path("licenses/silero-vad-LICENSE.txt"),
        ),
        (
            Path("resources/licenses/mlx-audio-swift-LICENSE.txt"),
            Path("licenses/mlx-audio-swift-LICENSE.txt"),
        ),
        (
            Path("resources/licenses/mlx-swift-LICENSE.txt"),
            Path("licenses/mlx-swift-LICENSE.txt"),
        ),
        (
            Path("resources/licenses/moss-transcribe-cpp-LICENSE.txt"),
            Path("licenses/moss-transcribe-cpp-LICENSE.txt"),
        ),
        (
            Path("resources/models/audio/silero_vad.onnx"),
            Path("models/audio/silero_vad.onnx"),
        ),
        (
            Path("examples/local_models_matcha.json"),
            Path("examples/local_models_matcha.json"),
        ),
        (
            Path("examples/local_models_qwen3_asr_0_6b_int8.json"),
            Path("examples/local_models_qwen3_asr_0_6b_int8.json"),
        ),
        (
            Path("examples/local_models_campplus.json"),
            Path("examples/local_models_campplus.json"),
        ),
        (
            Path("examples/local_models_mtd.json"),
            Path("examples/local_models_mtd.json"),
        ),
    )


def install_resources(app: Path) -> None:
    """Install static tools, examples, manifests, licenses, and models.

    Parameters
    ----------
    app : pathlib.Path
        Installed App bundle.
    """

    resources = app / "Contents" / "Resources"
    replace_directory(APP_ROOT / "resources" / "tools", resources / "tools")
    for source, destination in resource_file_mappings():
        atomic_copy_file(APP_ROOT / source, resources / destination)


def build_desktop(*, build_ui: bool, build_frontend: bool) -> Path:
    """Build the incremental Tauri desktop executable.

    Parameters
    ----------
    build_ui : bool
        Whether to run the App Vite production build first.
    build_frontend : bool
        Whether to rebuild and reinstall the shared ``xtalk-client`` first.

    Returns
    -------
    pathlib.Path
        Built release-mode ``xtalk-desktop`` executable.
    """

    if build_frontend:
        run(["npm", "run", "build"], cwd=FRONTEND_ROOT)
        run(
            [
                "npm",
                "install",
                "--no-save",
                "--package-lock=false",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
                str(FRONTEND_ROOT),
            ]
        )
        build_ui = True
    if build_ui:
        run(["npm", "run", "build"])
    run(
        [
            "cargo",
            "build",
            "--release",
            "--features",
            "tauri/custom-protocol",
            "--manifest-path",
            str(APP_ROOT / "src-tauri" / "Cargo.toml"),
        ]
    )
    executable = APP_ROOT / "src-tauri" / "target" / "release" / "xtalk-desktop"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return executable


def newest_existing_wheel() -> Path:
    """Return the newest previously built XTalk wheel.

    Returns
    -------
    pathlib.Path
        Most recently modified repository wheel.

    Raises
    ------
    FileNotFoundError
        Raised when no prior source wheel can be reused.
    """

    candidates = [
        *APP_ROOT.glob(".build/source-inputs/wheel/xtalk-*.whl"),
        *APP_ROOT.glob("resources/artifacts/xtalk-*.whl"),
    ]
    if not candidates:
        raise FileNotFoundError(
            "no reusable XTalk wheel exists; use core-backend first"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def ensure_wheel_builder(python: Path) -> Path:
    """Return a reusable virtual environment that can build XTalk wheels.

    Parameters
    ----------
    python : pathlib.Path
        Base interpreter used when the cached environment is absent.

    Returns
    -------
    pathlib.Path
        Python executable with the pinned build frontend installed.
    """

    existing = APP_ROOT / ".build" / "source-inputs" / "wheel-venv" / "bin" / "python"
    if existing.is_file():
        return existing
    environment = INCREMENTAL_BUILD_ROOT / "wheel-venv"
    interpreter = environment / "bin" / "python"
    if not interpreter.is_file():
        run([str(python), "-m", "venv", str(environment)])
        run(
            [
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "build==1.5.0",
            ]
        )
    return interpreter


def build_xtalk_wheel(python: Path) -> Path:
    """Build a fresh root XTalk wheel using a reusable build environment.

    Parameters
    ----------
    python : pathlib.Path
        Base interpreter for the cached wheel environment.

    Returns
    -------
    pathlib.Path
        Newly built XTalk wheel.
    """

    interpreter = ensure_wheel_builder(python)
    output = INCREMENTAL_BUILD_ROOT / "wheel"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    run(
        [
            str(interpreter),
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
    )
    wheels = sorted(output.glob("xtalk-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("incremental source build did not produce one wheel")
    return wheels[0]


def cached_pyinstaller() -> tuple[Path, Path] | None:
    """Return cached PyInstaller inputs when a previous full build exists.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path] | None
        Cached environment interpreter and spec file, when both exist.
    """

    interpreter = BACKEND_BUILD_ROOT / "venv" / "bin" / "python"
    specification = BACKEND_BUILD_ROOT / "spec" / "app-backend.spec"
    if interpreter.is_file() and specification.is_file():
        return interpreter, specification
    return None


def run_incremental_pyinstaller(wheel: Path | None) -> Path | None:
    """Reuse the existing backend environment and PyInstaller analysis.

    Parameters
    ----------
    wheel : pathlib.Path | None
        Fresh wheel to install before freezing root XTalk changes.

    Returns
    -------
    pathlib.Path | None
        Built backend executable, or ``None`` when no cache exists.
    """

    cached = cached_pyinstaller()
    if cached is None:
        return None
    interpreter, specification = cached
    if wheel is not None:
        run(
            [
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ]
        )
    run(
        [
            str(interpreter),
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            str(BACKEND_BUILD_ROOT / "dist"),
            "--workpath",
            str(BACKEND_BUILD_ROOT / "work"),
            str(specification),
        ]
    )
    executable = BACKEND_BUILD_ROOT / "dist" / "app-backend" / "app-backend"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return executable


def run_clean_backend_build(
    *,
    wheel: Path,
    python: Path,
    extras: list[str],
) -> Path:
    """Run the clean locked backend builder and return its executable.

    Parameters
    ----------
    wheel : pathlib.Path
        XTalk wheel installed into the frozen backend.
    python : pathlib.Path
        Python interpreter used to create the clean build environment.
    extras : list[str]
        Required XTalk dependency groups.

    Returns
    -------
    pathlib.Path
        Target-suffixed backend executable staged for Tauri.
    """

    command = [
        sys.executable,
        str(APP_ROOT / "scripts" / "build_backend.py"),
        "--python",
        str(python),
        "--xtalk-wheel",
        str(wheel),
    ]
    for extra in extras:
        command.extend(("--xtalk-extra", extra))
    run(command)
    executable = (
        APP_ROOT
        / "src-tauri"
        / "binaries"
        / f"app-backend-{resolve_target_triple()}"
    )
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return executable


def build_backend(
    *,
    scope: str,
    python: Path,
    wheel_override: Path | None,
    extras: list[str],
) -> tuple[Path, Path | None]:
    """Build the smallest backend artifact required by one scope.

    Parameters
    ----------
    scope : str
        ``app-backend``, ``core-backend``, or ``backend-runtime``.
    python : pathlib.Path
        Backend build interpreter.
    wheel_override : pathlib.Path | None
        Optional immutable wheel supplied by the caller.
    extras : list[str]
        XTalk extras used by a clean runtime build.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path | None]
        Backend executable and optional complete runtime directory.
    """

    if wheel_override is not None:
        wheel = wheel_override.expanduser().resolve()
        if not wheel.is_file():
            raise FileNotFoundError(wheel)
    elif scope == "app-backend":
        wheel = newest_existing_wheel()
    else:
        wheel = build_xtalk_wheel(python)

    if scope != "backend-runtime":
        incremental_wheel = (
            wheel
            if scope == "core-backend" or wheel_override is not None
            else None
        )
        incremental = run_incremental_pyinstaller(
            incremental_wheel
        )
        if incremental is not None:
            return incremental, None

    executable = run_clean_backend_build(
        wheel=wheel,
        python=python,
        extras=extras,
    )
    runtime = (
        APP_ROOT / "src-tauri" / "binaries" / "app-backend-runtime"
        if scope == "backend-runtime"
        else None
    )
    return executable, runtime


def install_backend_runtime(app: Path, executable: Path, runtime: Path) -> None:
    """Install a complete PyInstaller backend runtime into an App bundle.

    Parameters
    ----------
    app : pathlib.Path
        Installed App bundle.
    executable : pathlib.Path
        Newly frozen backend executable.
    runtime : pathlib.Path
        Complete PyInstaller onedir support directory.
    """

    contents = app / "Contents"
    atomic_copy_file(executable, contents / "MacOS" / "app-backend")
    replace_directory(runtime, contents / "Frameworks")
    with tempfile.TemporaryDirectory(
        prefix="xtalk-backend-metadata-"
    ) as temporary_name:
        metadata = Path(temporary_name) / "app-backend-runtime"
        metadata.mkdir()
        for source in sorted(runtime.iterdir()):
            if source.name.endswith(PYTHON_METADATA_SUFFIXES):
                shutil.copytree(source, metadata / source.name)
        replace_directory(
            metadata,
            contents / "Resources" / "app-backend-runtime",
        )
    link_python_metadata_to_resources(app)
    verify_internal_bundle_links(app)


def installed_app_process_ids(app: Path, *, include_desktop: bool = True) -> set[int]:
    """Return processes executing binaries from one installed App bundle.

    Parameters
    ----------
    app : pathlib.Path
        Installed App bundle whose native processes should be inspected.
    include_desktop : bool, optional
        Whether to include the main ``xtalk-desktop`` process.

    Returns
    -------
    set[int]
        Matching process identifiers.
    """

    macos_prefix = f"{app / 'Contents' / 'MacOS'}/"
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    process_ids: set[int] = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid_text, command = fields
        if not command.startswith(macos_prefix):
            continue
        executable_name = command[len(macos_prefix) :].split(maxsplit=1)[0]
        if not include_desktop and executable_name == "xtalk-desktop":
            continue
        process_ids.add(int(pid_text))
    return process_ids


def stop_installed_helpers(app: Path, timeout_s: float = 5.0) -> None:
    """Stop App-bundled helper processes left behind by macOS Quit.

    Parameters
    ----------
    app : pathlib.Path
        Installed App bundle whose helper processes must stop.
    timeout_s : float, optional
        Maximum wait after forcing remaining helpers to exit.

    Raises
    ------
    RuntimeError
        Raised when a bundled helper remains alive after SIGKILL.
    """

    helpers = installed_app_process_ids(app, include_desktop=False)
    for pid in helpers:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    graceful_deadline = time.monotonic() + 1.0
    while helpers and time.monotonic() < graceful_deadline:
        time.sleep(0.05)
        helpers &= installed_app_process_ids(app, include_desktop=False)

    for pid in helpers:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = installed_app_process_ids(app, include_desktop=False)
        if not remaining:
            return
        time.sleep(0.05)
    raise RuntimeError(
        "XTalk helper processes did not stop: "
        + ", ".join(str(pid) for pid in sorted(remaining))
    )


def stop_running_app(app: Path, timeout_s: float = 10.0) -> None:
    """Ask the installed XTalk process to quit before replacing code.

    Parameters
    ----------
    app : pathlib.Path
        Installed App bundle being updated.
    timeout_s : float, optional
        Maximum graceful-shutdown wait.

    Raises
    ------
    RuntimeError
        Raised when the desktop process remains alive.
    """

    initial = installed_app_process_ids(app)
    initial_helpers = installed_app_process_ids(app, include_desktop=False)
    if not initial:
        return
    if initial - initial_helpers:
        run(
            [
                "osascript",
                "-e",
                'tell application id "com.xtalk.desktop" to quit',
            ],
            check=False,
        )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        processes = installed_app_process_ids(app)
        helpers = installed_app_process_ids(app, include_desktop=False)
        desktop_processes = processes - helpers
        if not desktop_processes:
            stop_installed_helpers(app)
            return
        time.sleep(0.1)
    raise RuntimeError("XTalk did not quit; close it and rerun the installer")


def sign_and_verify(app: Path) -> None:
    """Apply the repository's local signature and verify the App seal.

    Parameters
    ----------
    app : pathlib.Path
        Updated installed App bundle.
    """

    sign_app(app, "-")
    run(["codesign", "--verify", "--deep", "--strict", str(app)])


def main() -> int:
    """Build, install, sign, verify, and optionally relaunch one scope.

    Returns
    -------
    int
        Process exit status.
    """

    if sys.platform != "darwin":
        raise RuntimeError("incremental App installation is supported only on macOS")
    args = parse_args()
    app = require_installed_app(args.app)
    desktop_executable: Path | None = None
    backend_executable: Path | None = None
    backend_runtime: Path | None = None

    if args.scope == "ui":
        desktop_executable = build_desktop(build_ui=True, build_frontend=False)
    elif args.scope == "frontend":
        desktop_executable = build_desktop(build_ui=True, build_frontend=True)
    elif args.scope == "shell":
        desktop_executable = build_desktop(build_ui=False, build_frontend=False)
    elif args.scope in {"app-backend", "core-backend", "backend-runtime"}:
        python = resolve_python(args.python)
        extras = list(dict.fromkeys(args.xtalk_extra or DEFAULT_XTALK_EXTRAS))
        backend_executable, backend_runtime = build_backend(
            scope=args.scope,
            python=python,
            wheel_override=args.xtalk_wheel,
            extras=extras,
        )

    stop_running_app(app)
    contents = app / "Contents"
    if desktop_executable is not None:
        atomic_copy_file(
            desktop_executable,
            contents / "MacOS" / "xtalk-desktop",
        )
    elif backend_executable is not None and backend_runtime is not None:
        install_backend_runtime(app, backend_executable, backend_runtime)
    elif backend_executable is not None:
        atomic_copy_file(
            backend_executable,
            contents / "MacOS" / "app-backend",
        )
    elif args.scope == "resources":
        install_resources(app)
    elif args.scope == "icon":
        atomic_copy_file(
            APP_ROOT / "src-tauri" / "icons" / "icon.icns",
            contents / "Resources" / "icon.icns",
        )

    sign_and_verify(app)
    if not args.no_launch:
        run(["open", str(app)])
    print(f"incrementally installed {args.scope}: {app}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
