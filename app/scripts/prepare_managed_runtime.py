"""Build and stage native runtimes used by optional managed models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


APP_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME_MANIFEST = APP_ROOT / "local-model-runtime" / "Cargo.toml"
MATCHA_RUNTIME_MANIFEST = APP_ROOT / "matcha-model-runtime" / "Cargo.toml"
MLX_RUNTIME_PACKAGE = APP_ROOT / "local-model-runtime-mlx"
TAURI_BINARIES = APP_ROOT / "src-tauri" / "binaries"
MANAGED_RESOURCES = APP_ROOT / "resources" / "managed-runtime"
MACOS_MANAGED_RUNTIME_RPATH = (
    "@executable_path/../Resources/managed-runtime/ort"
)


def parse_args() -> argparse.Namespace:
    """Parse explicit platform-runtime inputs.

    Returns
    -------
    argparse.Namespace
        Parsed build arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-triple")
    parser.add_argument("--sherpa-server", required=True, type=Path)
    parser.add_argument("--sherpa-library-dir", required=True, type=Path)
    parser.add_argument("--ort-library", required=True, type=Path)
    parser.add_argument("--cuda-runtime-dir", type=Path)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build the Rust runtime without Cargo's release profile.",
    )
    return parser.parse_args()


def resolve_target_triple(explicit: str | None) -> str:
    """Resolve the target triple used by Tauri external binaries.

    Parameters
    ----------
    explicit : str | None
        Optional caller-provided Rust target triple.

    Returns
    -------
    str
        Resolved non-empty target triple.
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


def require_file(path: Path, label: str) -> Path:
    """Resolve and validate one explicit runtime input.

    Parameters
    ----------
    path : pathlib.Path
        Input path.
    label : str
        Human-readable input name.

    Returns
    -------
    pathlib.Path
        Resolved regular-file path.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    """Resolve and validate one explicit runtime directory.

    Parameters
    ----------
    path : pathlib.Path
        Input path.
    label : str
        Human-readable input name.

    Returns
    -------
    pathlib.Path
        Resolved directory path.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def sherpa_shared_libraries(directory: Path, target: str) -> list[Path]:
    """Find the shared sherpa-onnx libraries needed by Matcha.

    Parameters
    ----------
    directory : pathlib.Path
        Target-specific sherpa library directory.
    target : str
        Rust target triple.

    Returns
    -------
    list[pathlib.Path]
        Shared sherpa libraries to stage beside ONNX Runtime.
    """

    if "windows" in target:
        suffixes = (".dll",)
    elif "apple" in target:
        suffixes = (".dylib",)
    else:
        suffixes = (".so",)
    libraries = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and "sherpa-onnx" in path.name.lower()
        and (
            path.name.lower().endswith(suffixes)
            or ("linux" in target and ".so." in path.name.lower())
        )
    )
    if not any("c-api" in path.name.lower() for path in libraries):
        raise FileNotFoundError(
            f"sherpa-onnx C API shared library is missing below {directory}"
        )
    return libraries


def executable_name(name: str, target: str) -> str:
    """Return a platform executable filename.

    Parameters
    ----------
    name : str
        Base executable name.
    target : str
        Rust target triple.

    Returns
    -------
    str
        Executable filename with the target platform suffix.
    """

    return f"{name}.exe" if "windows" in target else name


def copy_file(source: Path, destination: Path) -> None:
    """Copy one runtime file and create its destination directory.

    Parameters
    ----------
    source : pathlib.Path
        Existing source file.
    destination : pathlib.Path
        Destination file.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def reset_generated_runtime_files(directory: Path) -> None:
    """Remove stale generated libraries while retaining tracked guidance.

    Parameters
    ----------
    directory : pathlib.Path
        Shared managed runtime resource directory.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.name == "README.md":
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def target_supports_mlx(target: str) -> bool:
    """Return whether a target can run the Swift MLX sidecar.

    Parameters
    ----------
    target : str
        Rust target triple.

    Returns
    -------
    bool
        ``True`` only for Apple Silicon macOS.
    """

    return target == "aarch64-apple-darwin"


def build_local_runtime(target: str, debug: bool) -> Path:
    """Build the Rust MOSS runtime for one target.

    Parameters
    ----------
    target : str
        Rust target triple.
    debug : bool
        Whether to use Cargo's debug profile.

    Returns
    -------
    pathlib.Path
        Built executable path.
    """

    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(LOCAL_RUNTIME_MANIFEST),
        "--target",
        target,
    ]
    profile = "debug"
    if not debug:
        command.append("--release")
        profile = "release"
    subprocess.run(command, cwd=APP_ROOT, check=True)
    return (
        APP_ROOT
        / "local-model-runtime"
        / "target"
        / target
        / profile
        / executable_name("xtalk-local-model-runtime", target)
    )


def build_matcha_runtime(
    target: str,
    debug: bool,
    sherpa_library_dir: Path,
) -> Path:
    """Build the Rust Matcha runtime against shared sherpa libraries.

    Parameters
    ----------
    target : str
        Rust target triple.
    debug : bool
        Whether to use Cargo's debug profile.
    sherpa_library_dir : pathlib.Path
        Directory containing shared sherpa-onnx libraries built for the
        target.

    Returns
    -------
    pathlib.Path
        Built Matcha executable path.
    """

    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(MATCHA_RUNTIME_MANIFEST),
        "--target",
        target,
    ]
    profile = "debug"
    if not debug:
        command.append("--release")
        profile = "release"
    environment = dict(os.environ)
    environment["SHERPA_ONNX_LIB_DIR"] = str(sherpa_library_dir)
    if "apple" in target:
        rpath_flag = (
            "-C link-arg=-Wl,-rpath," + MACOS_MANAGED_RUNTIME_RPATH
        )
        existing_rustflags = environment.get("RUSTFLAGS", "").strip()
        environment["RUSTFLAGS"] = " ".join(
            value for value in (existing_rustflags, rpath_flag) if value
        )
    subprocess.run(command, cwd=APP_ROOT, check=True, env=environment)
    return (
        APP_ROOT
        / "matcha-model-runtime"
        / "target"
        / target
        / profile
        / executable_name("xtalk-matcha-model-runtime", target)
    )


def built_rust_runtime(name: str, target: str, debug: bool) -> Path:
    """Return the path of another binary from the local runtime package.

    Parameters
    ----------
    name : str
        Cargo binary name.
    target : str
        Rust target triple.
    debug : bool
        Whether Cargo used its debug profile.

    Returns
    -------
    pathlib.Path
        Expected executable path.
    """

    return (
        APP_ROOT
        / "local-model-runtime"
        / "target"
        / target
        / ("debug" if debug else "release")
        / executable_name(name, target)
    )


def build_mlx_runtime(debug: bool) -> tuple[Path, Path]:
    """Build the Apple Silicon MLX executable and Metal resource bundle.

    Parameters
    ----------
    debug : bool
        Whether to use Xcode's Debug configuration.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Executable and ``mlx-swift_Cmlx.bundle`` paths.
    """

    configuration = "Debug" if debug else "Release"
    derived_data = MLX_RUNTIME_PACKAGE / ".build" / "xcode"
    subprocess.run(
        [
            "xcodebuild",
            "build",
            "-scheme",
            "XTalkMLXRuntime",
            "-destination",
            "platform=macOS,arch=arm64",
            "-configuration",
            configuration,
            "-derivedDataPath",
            str(derived_data),
            "-disableAutomaticPackageResolution",
            "-onlyUsePackageVersionsFromResolvedFile",
            "CODE_SIGNING_ALLOWED=NO",
            "CLANG_ENABLE_CODE_COVERAGE=NO",
        ],
        cwd=MLX_RUNTIME_PACKAGE,
        check=True,
    )
    products = derived_data / "Build" / "Products" / configuration
    executable = require_file(
        products / "xtalk-mlx-model-runtime",
        "MLX model runtime",
    )
    bundles = list(products.rglob("mlx-swift_Cmlx.bundle"))
    if len(bundles) != 1 or not bundles[0].is_dir():
        raise FileNotFoundError(
            f"expected one MLX Metal resource bundle below {products}"
        )
    return executable, bundles[0]


def copy_cuda_runtime(source: Path | None, destination: Path) -> None:
    """Copy optional ONNX Runtime CUDA provider libraries.

    Parameters
    ----------
    source : pathlib.Path | None
        Directory from an ONNX Runtime GPU distribution.
    destination : pathlib.Path
        Managed runtime resource directory.
    """

    if source is None:
        return
    resolved = source.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"CUDA runtime directory is missing: {resolved}")
    providers = [
        path
        for path in resolved.iterdir()
        if path.is_file()
        and (
            "onnxruntime_providers_cuda" in path.name.lower()
            or "onnxruntime_providers_shared" in path.name.lower()
        )
    ]
    if not any("providers_cuda" in path.name.lower() for path in providers):
        raise FileNotFoundError(
            f"CUDA execution-provider library is missing below {resolved}"
        )
    for provider in providers:
        copy_file(provider, destination / provider.name)


def main() -> int:
    """Build and stage all target-specific managed runtime artifacts.

    Returns
    -------
    int
        Process exit status.
    """

    args = parse_args()
    target = resolve_target_triple(args.target_triple)
    sherpa_server = require_file(args.sherpa_server, "sherpa server")
    sherpa_library_dir = require_directory(
        args.sherpa_library_dir,
        "sherpa shared-library directory",
    )
    sherpa_libraries = sherpa_shared_libraries(sherpa_library_dir, target)
    ort_library = require_file(args.ort_library, "ONNX Runtime 1.27")
    if "1.27" not in ort_library.name and ort_library.name != "onnxruntime.dll":
        raise ValueError(
            "--ort-library must point to the ONNX Runtime 1.27 library"
        )
    local_runtime = require_file(
        build_local_runtime(target, args.debug),
        "local model runtime",
    )
    matcha_runtime = require_file(
        build_matcha_runtime(target, args.debug, sherpa_library_dir),
        "Matcha model runtime",
    )
    if target_supports_mlx(target):
        mlx_runtime, mlx_bundle = build_mlx_runtime(args.debug)
    else:
        mlx_runtime = require_file(
            built_rust_runtime(
                "xtalk-mlx-model-runtime-unavailable",
                target,
                args.debug,
            ),
            "unsupported-platform MLX runtime",
        )
        mlx_bundle = None

    copy_file(
        local_runtime,
        TAURI_BINARIES
        / f"local-model-runtime-{target}"
        f"{'.exe' if 'windows' in target else ''}",
    )
    copy_file(
        matcha_runtime,
        TAURI_BINARIES
        / f"matcha-model-runtime-{target}"
        f"{'.exe' if 'windows' in target else ''}",
    )
    copy_file(
        sherpa_server,
        TAURI_BINARIES
        / f"sherpa-onnx-offline-websocket-server-{target}"
        f"{'.exe' if 'windows' in target else ''}",
    )
    copy_file(
        mlx_runtime,
        TAURI_BINARIES
        / f"mlx-model-runtime-{target}"
        f"{'.exe' if 'windows' in target else ''}",
    )
    runtime_directory = MANAGED_RESOURCES / "ort"
    reset_generated_runtime_files(runtime_directory)
    copy_file(ort_library, runtime_directory / ort_library.name)
    for sherpa_library in sherpa_libraries:
        copy_file(sherpa_library, runtime_directory / sherpa_library.name)
    if mlx_bundle is not None:
        destination = MANAGED_RESOURCES / mlx_bundle.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(mlx_bundle, destination)
    copy_cuda_runtime(args.cuda_runtime_dir, runtime_directory)
    print(f"prepared managed runtime artifacts for {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
