"""Build desktop core inputs directly from the repository source tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parent
FRONTEND_ROOT = REPOSITORY_ROOT / "frontend"
SOURCE_BUILD_ROOT = APP_ROOT / ".build" / "source-inputs"
DEFAULT_XTALK_EXTRAS = ("ali", "silero-vad")
MINIMUM_PYTHON_VERSION = (3, 10)
BUILD_PACKAGE_REQUIREMENT = "build==1.5.0"


def run(
    command: list[str],
    *,
    cwd: Path,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one source-build command without invoking a shell.

    Parameters
    ----------
    command : list[str]
        Executable and argument vector.
    cwd : pathlib.Path
        Working directory for the command.
    capture_output : bool, optional
        Capture text output for parsing.
    env : dict[str, str] | None, optional
        Complete child-process environment override.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Successful command result.
    """

    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=capture_output,
        text=True,
        env=env,
    )


def npm_environment() -> dict[str, str]:
    """Return an npm environment isolated from the user's global cache.

    Returns
    -------
    dict[str, str]
        Process environment using an App-local npm cache.
    """

    cache = APP_ROOT / ".build" / "npm-cache"
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["npm_config_cache"] = str(cache)
    environment["npm_config_update_notifier"] = "false"
    return environment


def python_version(python: Path) -> tuple[int, int]:
    """Return the selected interpreter's major and minor version.

    Parameters
    ----------
    python : pathlib.Path
        Python executable to inspect.

    Returns
    -------
    tuple[int, int]
        Major and minor version.
    """

    result = run(
        [
            str(python),
            "-c",
            "import json,sys; print(json.dumps(list(sys.version_info[:2])))",
        ],
        cwd=APP_ROOT,
        capture_output=True,
    )
    major, minor = json.loads(result.stdout)
    return int(major), int(minor)


def resolve_sidecar_python(explicit: Path | None) -> Path:
    """Select the interpreter used to build the sidecar inputs.

    The locked sidecar dependencies are resolved for Python 3.12, so that
    interpreter is preferred when present. Any Python 3.10 or newer
    interpreter that can build the XTalk wheel is accepted.

    Parameters
    ----------
    explicit : pathlib.Path | None
        Command-line interpreter override.

    Returns
    -------
    pathlib.Path
        Validated Python interpreter path.

    Raises
    ------
    RuntimeError
        Raised when no supported interpreter is available.
    """

    configured = explicit
    if configured is None:
        environment_value = os.environ.get("XTALK_SIDECAR_PYTHON")
        configured = Path(environment_value) if environment_value else None
    candidates = (
        [configured]
        if configured is not None
        else [
            Path(path)
            for path in (
                shutil.which("python3.12"),
                shutil.which("python3.13"),
                shutil.which("python3.11"),
                shutil.which("python3"),
                sys.executable,
            )
            if path
        ]
    )
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate.expanduser()))
        if (
            absolute.is_file()
            and python_version(absolute) >= MINIMUM_PYTHON_VERSION
        ):
            return absolute
    raise RuntimeError(
        "source packaging requires Python 3.10 or newer; "
        "set XTALK_SIDECAR_PYTHON or pass --python"
    )


def wheel_version(wheel: Path) -> str:
    """Read the package version from a built XTalk wheel.

    Parameters
    ----------
    wheel : pathlib.Path
        Wheel created from the repository root.

    Returns
    -------
    str
        Declared distribution version.
    """

    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("XTalk wheel must contain exactly one METADATA file")
        metadata = Parser().parsestr(
            archive.read(metadata_paths[0]).decode("utf-8")
        )
    version = metadata.get("Version", "").strip()
    if not version:
        raise ValueError("XTalk wheel metadata has no version")
    return version


def venv_python(environment: Path) -> Path:
    """Return the Python executable inside a virtual environment.

    Parameters
    ----------
    environment : pathlib.Path
        Virtual-environment root.

    Returns
    -------
    pathlib.Path
        Interpreter path.
    """

    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def build_xtalk_wheel(python: Path) -> Path:
    """Build a fresh XTalk wheel in an isolated build environment.

    The base interpreter only needs the standard ``venv`` module. This step
    creates a dedicated environment, installs the pinned ``build`` package
    there, and builds the repository wheel through PEP 517 isolation.

    Parameters
    ----------
    python : pathlib.Path
        Python 3.10+ interpreter used to seed the wheel build environment.

    Returns
    -------
    pathlib.Path
        Fresh repository XTalk wheel.
    """

    environment = SOURCE_BUILD_ROOT / "wheel-venv"
    run([str(python), "-m", "venv", str(environment)], cwd=APP_ROOT)
    interpreter = venv_python(environment)
    run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            BUILD_PACKAGE_REQUIREMENT,
        ],
        cwd=APP_ROOT,
    )
    wheel_directory = SOURCE_BUILD_ROOT / "wheel"
    wheel_directory.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(interpreter),
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_directory),
        ],
        cwd=REPOSITORY_ROOT,
    )
    wheels = sorted(wheel_directory.glob("xtalk-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("repository source build did not produce one XTalk wheel")
    return wheels[0]


def parse_npm_pack_path(stdout: str) -> Path:
    """Read the generated package filename from ``npm pack --json`` output.

    Parameters
    ----------
    stdout : str
        Captured npm JSON output.

    Returns
    -------
    pathlib.Path
        Relative package filename reported by npm.
    """

    payload = json.loads(stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("npm pack must return exactly one package record")
    record = payload[0]
    if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
        raise ValueError("npm pack output has no package filename")
    return Path(record["filename"])


def build_frontend_package() -> tuple[Path, str]:
    """Build and pack the current repository ``frontend`` sources."""

    environment = npm_environment()
    run(["npm", "ci"], cwd=FRONTEND_ROOT, env=environment)
    run(["npm", "run", "build"], cwd=FRONTEND_ROOT, env=environment)
    package_directory = SOURCE_BUILD_ROOT / "frontend"
    package_directory.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "npm",
            "pack",
            "--json",
            "--pack-destination",
            str(package_directory),
        ],
        cwd=FRONTEND_ROOT,
        capture_output=True,
        env=environment,
    )
    package = package_directory / parse_npm_pack_path(result.stdout)
    if not package.is_file():
        raise RuntimeError("frontend source build did not produce an npm package")
    package_manifest = json.loads(
        (FRONTEND_ROOT / "package.json").read_text(encoding="utf-8")
    )
    version = str(package_manifest.get("version", "")).strip()
    if not version:
        raise ValueError("frontend package has no version")
    return package, version


def prepare_locked_artifacts(
    wheel: Path,
    client_package: Path,
    client_version: str,
) -> Path:
    """Copy fresh core artifacts into the App build context.

    Parameters
    ----------
    wheel : pathlib.Path
        Fresh XTalk wheel.
    client_package : pathlib.Path
        Fresh xtalk-client npm package.
    client_version : str
        Client package version.

    Returns
    -------
    pathlib.Path
        Client artifact copied under ``app/resources/artifacts``.
    """

    run(
        [
            sys.executable,
            str(APP_ROOT / "scripts" / "prepare_artifacts.py"),
            "--xtalk-wheel",
            str(wheel),
            "--xtalk-version",
            wheel_version(wheel),
            "--client-package",
            str(client_package),
            "--client-version",
            client_version,
        ],
        cwd=APP_ROOT,
    )
    return APP_ROOT / "resources" / "artifacts" / client_package.name


def install_fresh_client(client_package: Path) -> None:
    """Replace the App's installed client with the freshly packed source."""

    installed_client = APP_ROOT / "node_modules" / "xtalk-client"
    if installed_client.is_symlink() or installed_client.is_file():
        installed_client.unlink()
    elif installed_client.is_dir():
        shutil.rmtree(installed_client)
    run(
        [
            "npm",
            "install",
            "--no-save",
            "--package-lock=false",
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            str(client_package),
        ],
        cwd=APP_ROOT,
        env=npm_environment(),
    )


def build_backend(
    wheel: Path,
    python: Path,
    extras: list[str],
    target: str | None,
) -> None:
    """Freeze the App backend from the freshly built XTalk wheel."""

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
    if target:
        command.extend(("--target-triple", target))
    run(command, cwd=APP_ROOT)


def parse_args() -> argparse.Namespace:
    """Parse source packaging overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--xtalk-extra", action="append", default=[])
    parser.add_argument("--target-triple")
    return parser.parse_args()


def main() -> int:
    """Rebuild every repository-owned core input before Tauri packaging."""

    args = parse_args()
    if SOURCE_BUILD_ROOT.exists():
        shutil.rmtree(SOURCE_BUILD_ROOT)
    SOURCE_BUILD_ROOT.mkdir(parents=True)
    sidecar_python = resolve_sidecar_python(args.python)
    wheel = build_xtalk_wheel(sidecar_python)
    client_package, client_version = build_frontend_package()
    locked_client = prepare_locked_artifacts(
        wheel,
        client_package,
        client_version,
    )
    install_fresh_client(locked_client)
    extras = list(dict.fromkeys([*DEFAULT_XTALK_EXTRAS, *args.xtalk_extra]))
    target = args.target_triple or os.environ.get("TAURI_ENV_TARGET_TRIPLE")
    build_backend(wheel, sidecar_python, extras, target)
    print(
        "prepared desktop source inputs: "
        f"xtalk {wheel_version(wheel)}, xtalk-client {client_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
