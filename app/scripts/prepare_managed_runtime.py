"""Build and stage native runtimes used by optional managed models."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


APP_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME_MANIFEST = APP_ROOT / "local-model-runtime" / "Cargo.toml"
MLX_RUNTIME_PACKAGE = APP_ROOT / "local-model-runtime-mlx"
TAURI_BINARIES = APP_ROOT / "src-tauri" / "binaries"
MANAGED_RESOURCES = APP_ROOT / "resources" / "managed-runtime"
WAKE_WORD_RESOURCES = APP_ROOT / "resources" / "models" / "wake-word"
WAKE_WORD_MODEL_FILES = (
    "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "tokens.txt",
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
    parser.add_argument("--sherpa-keyword-spotter", required=True, type=Path)
    parser.add_argument("--sherpa-kws-model-dir", required=True, type=Path)
    parser.add_argument("--sherpa-ort-library", required=True, type=Path)
    parser.add_argument("--tts-ort-library", required=True, type=Path)
    parser.add_argument("--sherpa-cuda-runtime-dir", type=Path)
    parser.add_argument("--tts-cuda-runtime-dir", type=Path)
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
    sherpa_keyword_spotter = require_file(
        args.sherpa_keyword_spotter,
        "sherpa keyword spotter",
    )
    sherpa_kws_model_dir = require_directory(
        args.sherpa_kws_model_dir,
        "sherpa keyword-spotting model directory",
    )
    sherpa_ort = require_file(args.sherpa_ort_library, "sherpa ONNX Runtime")
    tts_ort = require_file(args.tts_ort_library, "TTS ONNX Runtime")
    local_runtime = require_file(
        build_local_runtime(target, args.debug),
        "local model runtime",
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
        sherpa_server,
        TAURI_BINARIES
        / f"sherpa-onnx-offline-websocket-server-{target}"
        f"{'.exe' if 'windows' in target else ''}",
    )
    copy_file(
        sherpa_keyword_spotter,
        TAURI_BINARIES
        / f"sherpa-onnx-keyword-spotter-microphone-{target}"
        f"{'.exe' if 'windows' in target else ''}",
    )
    for filename in WAKE_WORD_MODEL_FILES:
        copy_file(
            require_file(
                sherpa_kws_model_dir / filename,
                f"sherpa keyword-spotting model file {filename}",
            ),
            WAKE_WORD_RESOURCES / filename,
        )
    copy_file(
        mlx_runtime,
        TAURI_BINARIES
        / f"mlx-model-runtime-{target}"
        f"{'.exe' if 'windows' in target else ''}",
    )
    copy_file(
        tts_ort,
        MANAGED_RESOURCES / "ort-tts" / tts_ort.name,
    )
    copy_file(
        sherpa_ort,
        MANAGED_RESOURCES / "sherpa" / sherpa_ort.name,
    )
    if mlx_bundle is not None:
        destination = MANAGED_RESOURCES / mlx_bundle.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(mlx_bundle, destination)
    copy_cuda_runtime(
        args.tts_cuda_runtime_dir,
        MANAGED_RESOURCES / "ort-tts",
    )
    copy_cuda_runtime(
        args.sherpa_cuda_runtime_dir,
        MANAGED_RESOURCES / "sherpa",
    )
    print(f"prepared managed runtime artifacts for {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
