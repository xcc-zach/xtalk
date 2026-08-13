"""Build the Python sidecar as a target-specific PyInstaller onedir layout."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = APP_ROOT / ".build" / "backend"
TAURI_BINARIES = APP_ROOT / "src-tauri" / "binaries"
SIDECAR_LOCK = APP_ROOT / "requirements" / "sidecar.lock"
MINIMUM_PYTHON_VERSION = (3, 10)
LOCKED_PYTHON_VERSION = (3, 12)
EXTRA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LOCKED_REQUIREMENT_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^=<>!~;\s]+$"
)
UNLOCKED_CODEX_PACKAGES = frozenset(
    {"openai-codex", "openai-codex-cli-bin"}
)
REQUIRED_XTALK_EXTRAS = frozenset({"silero-vad"})
REQUIRED_XTALK_MODULES = (
    "xtalk/models/asr/sherpa_onnx_asr.py",
    "xtalk/models/tts/moss_tts_nano.py",
    "xtalk/models/tts/sherpa_onnx_tts.py",
)


def parse_args() -> argparse.Namespace:
    """Parse explicit immutable build inputs.

    Returns
    -------
    argparse.Namespace
        Parsed build arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtalk-wheel", required=True, type=Path)
    parser.add_argument(
        "--xtalk-extra",
        action="append",
        default=[],
        help="Install one named optional dependency group from the XTalk wheel.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--target-triple")
    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path = APP_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    """Run one build command without invoking a shell.

    Parameters
    ----------
    command : list[str]
        Executable and argument vector.
    cwd : pathlib.Path, optional
        Working directory.
    env : dict[str, str] | None, optional
        Explicit subprocess environment.
    """

    subprocess.run(command, cwd=cwd, env=env, check=True)


def selected_python_version(python: Path) -> tuple[int, int]:
    """Read the selected interpreter's major and minor version.

    Parameters
    ----------
    python : pathlib.Path
        Python executable.

    Returns
    -------
    tuple[int, int]
        Major and minor version.
    """

    result = subprocess.run(
        [
            str(python),
            "-c",
            "import json,sys; print(json.dumps(list(sys.version_info[:2])))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    major, minor = json.loads(result.stdout)
    return int(major), int(minor)


def selected_python_base_prefix(python: Path) -> Path:
    """Return the base installation directory for a Python interpreter.

    Parameters
    ----------
    python : pathlib.Path
        Python executable to inspect.

    Returns
    -------
    pathlib.Path
        Resolved ``sys.base_prefix`` reported by the interpreter.
    """

    result = subprocess.run(
        [str(python), "-c", "import sys; print(sys.base_prefix)"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def resolve_target_triple(explicit: str | None) -> str:
    """Resolve the Rust target triple used by Tauri sidecars.

    Parameters
    ----------
    explicit : str | None
        Caller-provided target triple.

    Returns
    -------
    str
        Non-empty target triple.
    """

    if explicit:
        return explicit
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


def wheel_requirement(wheel: Path, extras: list[str]) -> str:
    """Build a deterministic local-wheel requirement with optional extras.

    Parameters
    ----------
    wheel : pathlib.Path
        Immutable XTalk wheel.
    extras : list[str]
        Optional dependency group names requested by the caller.

    Returns
    -------
    str
        Pip requirement string for the local wheel.

    Raises
    ------
    ValueError
        Raised when an extra name is not a valid package extra identifier.
    """

    unique_extras: list[str] = []
    for extra in extras:
        if not EXTRA_NAME_PATTERN.fullmatch(extra):
            raise ValueError(f"invalid XTalk extra name: {extra!r}")
        if extra not in unique_extras:
            unique_extras.append(extra)
    suffix = f"[{','.join(sorted(unique_extras))}]" if unique_extras else ""
    return f"{wheel}{suffix}"


def validate_required_extras(extras: list[str]) -> None:
    """Require dependency groups needed by the bundled desktop pipeline.

    Parameters
    ----------
    extras : list[str]
        Optional dependency group names requested by the caller.

    Raises
    ------
    ValueError
        Raised when a required local-model dependency group is absent.
    """

    missing = REQUIRED_XTALK_EXTRAS - set(extras)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing required XTalk extras: {missing_text}")


def validate_required_wheel_modules(wheel: Path) -> None:
    """Require client adapters needed by managed local-model services.

    Parameters
    ----------
    wheel : pathlib.Path
        Immutable XTalk wheel used to build the packaged backend.

    Raises
    ------
    ValueError
        Raised when the wheel omits a managed ASR or TTS client module.
    """

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    missing = [
        module_path
        for module_path in REQUIRED_XTALK_MODULES
        if module_path not in members
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "XTalk wheel is missing required managed-model modules: "
            f"{missing_text}"
        )


def validate_sidecar_lock(path: Path = SIDECAR_LOCK) -> list[str]:
    """Validate exact sidecar constraints while keeping Codex floating.

    Parameters
    ----------
    path : pathlib.Path, optional
        Checked-in pip constraints file.

    Returns
    -------
    list[str]
        Validated requirement lines.

    Raises
    ------
    ValueError
        Raised for non-exact constraints, duplicates, or a pinned Codex SDK.
    """

    requirements: list[str] = []
    names: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not LOCKED_REQUIREMENT_PATTERN.fullmatch(line):
            raise ValueError(
                f"sidecar lock line {line_number} is not exact: {line!r}"
            )
        name = line.split("==", maxsplit=1)[0].lower().replace("_", "-")
        if name in UNLOCKED_CODEX_PACKAGES:
            raise ValueError(f"{name} must remain unlocked in the sidecar lock")
        if name in names:
            raise ValueError(f"duplicate sidecar lock package: {name}")
        names.add(name)
        requirements.append(line)
    if "pyinstaller" not in names or "wheel" not in names:
        raise ValueError("sidecar lock must constrain PyInstaller and wheel")
    return requirements


def prepare_environment(
    python: Path,
    wheel: Path,
    extras: list[str],
) -> Path:
    """Create an isolated sidecar build environment and install inputs.

    Parameters
    ----------
    python : pathlib.Path
        Build interpreter.
    wheel : pathlib.Path
        Immutable XTalk wheel.
    extras : list[str]
        Optional dependency groups installed from the XTalk wheel.

    Returns
    -------
    pathlib.Path
        Virtual-environment interpreter.
    """

    validate_sidecar_lock()
    environment = BUILD_ROOT / "venv"
    if environment.exists():
        shutil.rmtree(environment)
    run([str(python), "-m", "venv", str(environment)])
    interpreter = venv_python(environment)
    pip_cache = BUILD_ROOT / "pip-cache"
    pip_cache.mkdir(parents=True, exist_ok=True)
    build_environment = os.environ.copy()
    build_environment["PIP_CACHE_DIR"] = str(pip_cache)
    build_environment["PIP_CONSTRAINT"] = str(SIDECAR_LOCK)
    run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--constraint",
            str(SIDECAR_LOCK),
            "pyinstaller",
            "wheel",
            wheel_requirement(wheel, extras),
            str(APP_ROOT),
        ],
        env=build_environment,
    )
    run(
        [str(interpreter), "-m", "pip", "check"],
        env=build_environment,
    )
    return interpreter


def build_onedir(interpreter: Path, target: str) -> tuple[Path, Path]:
    """Freeze the sidecar and return its executable and support directory.

    Parameters
    ----------
    interpreter : pathlib.Path
        Isolated build interpreter.
    target : str
        Tauri target triple.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Main executable and PyInstaller support directory.
    """

    work_path = BUILD_ROOT / "work"
    dist_path = BUILD_ROOT / "dist"
    spec_path = BUILD_ROOT / "spec"
    for path in (work_path, dist_path, spec_path):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    config_path = BUILD_ROOT / "pyinstaller-config"
    config_path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYINSTALLER_CONFIG_DIR"] = str(config_path)
    if "windows" in target:
        conda_runtime = selected_python_base_prefix(interpreter) / "Library" / "bin"
        if conda_runtime.is_dir():
            environment["PATH"] = os.pathsep.join(
                (str(conda_runtime), environment.get("PATH", ""))
            )

    runtime_dir_name = "app-backend-runtime"
    command = [
        str(interpreter),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "app-backend",
        "--contents-directory",
        runtime_dir_name,
        "--paths",
        str(APP_ROOT),
        "--collect-submodules",
        "xtalk.models",
        "--collect-data",
        "xtalk",
        "--collect-all",
        "openai_codex",
        "--exclude-module",
        "codex_cli_bin",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "transformers",
        "--exclude-module",
        "vllm",
        "--exclude-module",
        "torchvision",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
        str(APP_ROOT / "backend" / "entrypoint.py"),
    ]
    run(command, env=environment)

    extension = ".exe" if "windows" in target else ""
    output_root = dist_path / "app-backend"
    executable = output_root / f"app-backend{extension}"
    support = output_root / runtime_dir_name
    if not executable.is_file() or not support.is_dir():
        raise RuntimeError("PyInstaller did not produce the expected onedir layout")
    return executable, support


def install_tauri_layout(executable: Path, support: Path, target: str) -> Path:
    """Install the local target's complete onedir layout for Tauri.

    Parameters
    ----------
    executable : pathlib.Path
        PyInstaller main executable.
    support : pathlib.Path
        PyInstaller support directory.
    target : str
        Tauri target triple.

    Returns
    -------
    pathlib.Path
        Target-suffixed sidecar executable.
    """

    TAURI_BINARIES.mkdir(parents=True, exist_ok=True)
    extension = ".exe" if "windows" in target else ""
    destination = TAURI_BINARIES / f"app-backend-{target}{extension}"
    runtime_destination = TAURI_BINARIES / "app-backend-runtime"
    if runtime_destination.exists():
        shutil.rmtree(runtime_destination)
    shutil.copytree(support, runtime_destination)
    shutil.copy2(executable, destination)
    if not extension:
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR)
    return destination


def main() -> int:
    """Build and install the target-specific sidecar layout.

    Returns
    -------
    int
        Process exit status.
    """

    args = parse_args()
    wheel = args.xtalk_wheel.expanduser().resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("--xtalk-wheel must point to an existing wheel")
    python = args.python.expanduser().resolve()
    version = selected_python_version(python)
    if version < MINIMUM_PYTHON_VERSION:
        raise RuntimeError(
            "locked sidecar builds require Python "
            f"{MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]} or newer"
        )
    if version != LOCKED_PYTHON_VERSION:
        print(
            "warning: requirements/sidecar.lock is resolved for Python "
            f"{LOCKED_PYTHON_VERSION[0]}.{LOCKED_PYTHON_VERSION[1]}; "
            f"building with Python {version[0]}.{version[1]} may fail to "
            "resolve locked dependencies",
            file=sys.stderr,
        )
    target = resolve_target_triple(args.target_triple)
    validate_required_extras(args.xtalk_extra)
    validate_required_wheel_modules(wheel)
    interpreter = prepare_environment(python, wheel, args.xtalk_extra)
    executable, support = build_onedir(interpreter, target)
    destination = install_tauri_layout(executable, support, target)
    print(destination.relative_to(APP_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
