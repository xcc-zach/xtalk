"""Download, verify, and stage the native managed-model runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import stat
import subprocess
import sys
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = APP_ROOT / "resources" / "manifests" / "native-runtimes.lock.json"
MTD_SOURCE_LOCK_PATH = (
    APP_ROOT
    / "resources"
    / "manifests"
    / "moss-transcribe-runtime.lock.json"
)
DEFAULT_CACHE = APP_ROOT / ".build" / "native-runtime-cache"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_READ_TIMEOUT_SECONDS = 300
MTD_SOURCE_NAMES = {"moss-transcribe.cpp", "ggml"}
MTD_NATIVE_CMAKE_SETTING = 'set(GGML_NATIVE ON CACHE BOOL "" FORCE)'
MTD_LLAMAFILE_CMAKE_SETTING = 'set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)'


def verified_tls_context() -> ssl.SSLContext:
    """Create a verified TLS context for build-time artifact downloads.

    Python.org macOS interpreters do not always inherit the Keychain CA set.
    Prefer an explicit ``SSL_CERT_FILE`` when supplied, then the CA bundle from
    the locked ``certifi`` dependency, while retaining normal certificate and
    hostname verification in every case.

    Returns
    -------
    ssl.SSLContext
        Context that requires a trusted server certificate.
    """

    configured_ca = os.environ.get("SSL_CERT_FILE")
    if configured_ca:
        return ssl.create_default_context(cafile=configured_ca)
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def parse_args() -> argparse.Namespace:
    """Parse native runtime preparation options.

    Returns
    -------
    argparse.Namespace
        Selected target, cache, and Cargo profile.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-triple")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sherpa-keyword-spotter", type=Path)
    parser.add_argument("--sherpa-kws-model-dir", type=Path)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_target_triple(explicit: str | None) -> str:
    """Resolve the Rust target triple used by Tauri sidecars.

    Parameters
    ----------
    explicit : str | None
        Optional caller-provided target triple.

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


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_target_record(target: str) -> dict[str, str]:
    """Load and validate the immutable archive record for one target.

    Parameters
    ----------
    target : str
        Rust target triple.

    Returns
    -------
    dict[str, str]
        Archive filename, URL, and SHA-256.

    Raises
    ------
    ValueError
        If the lock file or target record is invalid.
    """

    payload: Any = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("native runtime lock has an unsupported schema")
    targets = payload.get("targets")
    if not isinstance(targets, dict) or target not in targets:
        supported = ", ".join(sorted(targets or {}))
        raise ValueError(
            f"no native runtime is locked for {target}; supported: {supported}"
        )
    record = targets[target]
    if not isinstance(record, dict):
        raise ValueError(f"native runtime record for {target} must be an object")
    required = ("archive", "url", "sha256")
    if any(not isinstance(record.get(key), str) for key in required):
        raise ValueError(f"native runtime record for {target} is incomplete")
    digest = record["sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"native runtime digest for {target} is invalid")
    if not record["url"].startswith("https://github.com/k2-fsa/sherpa-onnx/"):
        raise ValueError(f"native runtime URL for {target} is not trusted")
    if Path(record["archive"]).name != record["archive"]:
        raise ValueError(f"native runtime archive name for {target} is invalid")
    return {key: record[key] for key in required}


def load_mtd_source_records() -> dict[str, dict[str, str]]:
    """Load immutable moss-transcribe.cpp and ggml source archives.

    Returns
    -------
    dict[str, dict[str, str]]
        Strictly validated source records keyed by project name.

    Raises
    ------
    ValueError
        If the source lock is incomplete or does not use immutable GitHub
        archive URLs.
    """

    payload: Any = json.loads(MTD_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("MTD source lock has an unsupported schema")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != MTD_SOURCE_NAMES:
        raise ValueError("MTD source lock must contain moss-transcribe.cpp and ggml")
    repositories = {
        "moss-transcribe.cpp": "localai-org/moss-transcribe.cpp",
        "ggml": "ggml-org/ggml",
    }
    result: dict[str, dict[str, str]] = {}
    for name, repository in repositories.items():
        record = sources[name]
        required = {"archive", "revision", "sha256", "url"}
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"MTD source record for {name} is invalid")
        if not all(isinstance(record[key], str) and record[key] for key in required):
            raise ValueError(f"MTD source record for {name} is incomplete")
        revision = record["revision"]
        expected_url = f"https://github.com/{repository}/archive/{revision}.tar.gz"
        digest = record["sha256"]
        if (
            len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
            or record["url"] != expected_url
            or Path(record["archive"]).name != record["archive"]
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"MTD source record for {name} is not immutable")
        result[name] = {key: record[key] for key in required}
    return result


def download_verified(record: dict[str, str], cache: Path) -> Path:
    """Download one archive and require its locked SHA-256.

    Parameters
    ----------
    record : dict[str, str]
        Locked archive metadata.
    cache : pathlib.Path
        Persistent build cache directory.

    Returns
    -------
    pathlib.Path
        Verified archive path.
    """

    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / record["archive"]
    if destination.is_file() and sha256_file(destination) == record["sha256"]:
        print(f"using verified native runtime cache: {destination}")
        return destination
    if destination.exists():
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    last_error: OSError | ValueError | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        request = urllib.request.Request(
            record["url"],
            headers={"User-Agent": "XTalk-native-runtime-builder/1"},
        )
        digest = hashlib.sha256()
        downloaded = 0
        expected_length: int | None = None
        try:
            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_READ_TIMEOUT_SECONDS,
                context=verified_tls_context(),
            ) as response:
                length_header = response.headers.get("Content-Length")
                expected_length = int(length_header) if length_header else None
                with partial.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
            if expected_length is not None and downloaded != expected_length:
                raise ValueError(
                    "native runtime download was truncated: "
                    f"expected {expected_length} bytes, got {downloaded}"
                )
            actual = digest.hexdigest()
            if actual != record["sha256"]:
                raise ValueError(
                    "native runtime SHA-256 mismatch: "
                    f"expected {record['sha256']}, got {actual}"
                )
            os.replace(partial, destination)
            return destination
        except (OSError, ValueError) as error:
            last_error = error
            print(
                "native runtime download failed "
                f"({attempt}/{DOWNLOAD_ATTEMPTS}): {error}",
                file=sys.stderr,
            )
        finally:
            if partial.exists():
                partial.unlink()
    raise RuntimeError("could not download a verified native runtime") from last_error


def extract_regular_files(archive_path: Path, destination: Path) -> None:
    """Extract files and materialize only safe internal archive links.

    Parameters
    ----------
    archive_path : pathlib.Path
        Verified tar archive.
    destination : pathlib.Path
        Empty extraction directory.
    """

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    link_members: list[tarfile.TarInfo] = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(
                    "native runtime archive escapes extraction root: "
                    f"{member.name}"
                )
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                link_members.append(member)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read native runtime member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(stat.S_IMODE(member.mode))
    pending = link_members
    while pending:
        unresolved: list[tarfile.TarInfo] = []
        for member in pending:
            target = (destination / member.name).resolve()
            linked = (
                (target.parent / member.linkname).resolve()
                if member.issym()
                else (destination / member.linkname).resolve()
            )
            if not linked.is_relative_to(root):
                continue
            if not linked.is_file():
                unresolved.append(member)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(linked, target)
        if len(unresolved) == len(pending):
            break
        pending = unresolved


def single_extracted_directory(root: Path, label: str) -> Path:
    """Return the only top-level directory from a source archive.

    Parameters
    ----------
    root : pathlib.Path
        Extracted archive directory.
    label : str
        Human-readable source name.

    Returns
    -------
    pathlib.Path
        Unique top-level source directory.
    """

    directories = sorted(path for path in root.iterdir() if path.is_dir())
    files = [path for path in root.iterdir() if path.is_file()]
    if len(directories) != 1 or files:
        raise ValueError(f"{label} archive must contain one source directory")
    return directories[0]


def assemble_mtd_source(cache: Path) -> Path:
    """Download, verify, and assemble the pinned MTD source tree.

    The upstream project keeps ggml as a Git submodule. Release builds fetch
    both immutable archives, place the locked ggml revision in the submodule
    directory, and make the upstream native-tuning default overridable so
    Cargo can produce binaries compatible with older CPUs.

    Parameters
    ----------
    cache : pathlib.Path
        Persistent native-runtime build cache.

    Returns
    -------
    pathlib.Path
        Assembled moss-transcribe.cpp source directory.
    """

    records = load_mtd_source_records()
    archives = {
        name: download_verified(record, cache)
        for name, record in records.items()
    }
    cache_key = "-".join(
        records[name]["sha256"][:12] for name in sorted(records)
    )
    destination = cache / f"mtd-source-{cache_key}"
    expected_marker: dict[str, Any] = {
        "schema_version": 1,
        "sources": {
            name: records[name]["revision"] for name in sorted(records)
        },
    }

    extracted_roots: dict[str, Path] = {}
    for name, archive in archives.items():
        extracted = cache / f"mtd-extracted-{name.replace('.', '-')}-{cache_key}"
        extract_regular_files(archive, extracted)
        extracted_roots[name] = single_extracted_directory(extracted, name)

    staging = cache / f"mtd-source-staging-{cache_key}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(extracted_roots["moss-transcribe.cpp"], staging)
    ggml_destination = staging / "third_party" / "ggml"
    if ggml_destination.exists():
        shutil.rmtree(ggml_destination)
    ggml_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(extracted_roots["ggml"], ggml_destination)

    cmake_path = staging / "CMakeLists.txt"
    cmake = cmake_path.read_text(encoding="utf-8")
    if cmake.count(MTD_NATIVE_CMAKE_SETTING) != 1:
        raise ValueError("locked moss-transcribe.cpp native setting changed")
    cmake = cmake.replace(
        MTD_NATIVE_CMAKE_SETTING,
        "if(NOT DEFINED GGML_NATIVE)\n"
        "  set(GGML_NATIVE ON CACHE BOOL \"\")\n"
        "endif()",
    )
    if cmake.count(MTD_LLAMAFILE_CMAKE_SETTING) != 1:
        raise ValueError("locked moss-transcribe.cpp llamafile setting changed")
    cmake = cmake.replace(
        MTD_LLAMAFILE_CMAKE_SETTING,
        "if(NOT DEFINED GGML_LLAMAFILE)\n"
        "  set(GGML_LLAMAFILE ON CACHE BOOL \"\")\n"
        "endif()",
    )
    cmake_path.write_text(cmake, encoding="utf-8")
    marker = staging / ".xtalk-source-lock.json"
    marker.write_text(
        json.dumps(expected_marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)
    return destination


def unique_match(
    root: Path,
    predicate: Callable[[Path], bool],
    label: str,
) -> Path:
    """Find exactly one extracted file matching a predicate.

    Parameters
    ----------
    root : pathlib.Path
        Extracted archive root.
    predicate : Callable[[pathlib.Path], bool]
        File-selection predicate.
    label : str
        Human-readable artifact name.

    Returns
    -------
    pathlib.Path
        The unique matching file.
    """

    matches = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and predicate(path)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {label} below {root}, found {len(matches)}"
        )
    return matches[0]


def locate_runtime_inputs(root: Path, target: str) -> tuple[Path, Path, Path]:
    """Locate server, sherpa library directory, and versioned ORT library.

    Parameters
    ----------
    root : pathlib.Path
        Extracted official sherpa distribution.
    target : str
        Rust target triple.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        WebSocket server, shared-library directory, and ONNX Runtime 1.27.
    """

    server_name = (
        "sherpa-onnx-offline-websocket-server.exe"
        if "windows" in target
        else "sherpa-onnx-offline-websocket-server"
    )
    server = unique_match(root, lambda path: path.name == server_name, "sherpa server")
    if "windows" in target:
        c_api_name = "sherpa-onnx-c-api.dll"
        ort_name = "onnxruntime.dll"
        c_api = unique_match(root, lambda path: path.name == c_api_name, "sherpa C API")
        ort = unique_match(root, lambda path: path.name == ort_name, "ONNX Runtime")
    elif "apple" in target:
        c_api = unique_match(
            root,
            lambda path: path.name == "libsherpa-onnx-c-api.dylib",
            "sherpa C API",
        )
        ort = unique_match(
            root,
            lambda path: path.name.startswith("libonnxruntime.1.27.")
            and path.name.endswith(".dylib"),
            "ONNX Runtime 1.27",
        )
    else:
        c_api = unique_match(
            root,
            lambda path: path.name == "libsherpa-onnx-c-api.so",
            "sherpa C API",
        )
        ort = unique_match(
            root,
            lambda path: path.name.startswith("libonnxruntime.so.1.27."),
            "ONNX Runtime 1.27",
        )
    if ort.parent != c_api.parent:
        raise ValueError("sherpa and ONNX Runtime libraries are not colocated")
    return server, c_api.parent, ort


def prepare_runtime(
    target: str,
    cache: Path,
    debug: bool,
    sherpa_keyword_spotter: Path | None = None,
    sherpa_kws_model_dir: Path | None = None,
) -> None:
    """Download the locked archive and invoke the native staging build.

    Parameters
    ----------
    target : str
        Rust target triple.
    cache : pathlib.Path
        Persistent build cache directory.
    debug : bool
        Whether to build Rust sidecars without the release profile.
    sherpa_keyword_spotter : pathlib.Path | None, optional
        Target-specific microphone keyword-spotter executable to stage.
    sherpa_kws_model_dir : pathlib.Path | None, optional
        Directory containing the fixed wake-word model files.
    """

    record = load_target_record(target)
    archive = download_verified(record, cache)
    extracted = cache / f"extracted-{target}-{record['sha256'][:12]}"
    extract_regular_files(archive, extracted)
    server, sherpa_directory, ort = locate_runtime_inputs(extracted, target)
    mtd_source = assemble_mtd_source(cache)
    command = [
        sys.executable,
        str(APP_ROOT / "scripts" / "prepare_managed_runtime.py"),
        "--target-triple",
        target,
        "--sherpa-server",
        str(server),
        "--sherpa-library-dir",
        str(sherpa_directory),
        "--ort-library",
        str(ort),
        "--mtd-source-dir",
        str(mtd_source),
    ]
    if sherpa_keyword_spotter is not None:
        command.extend(
            ["--sherpa-keyword-spotter", str(sherpa_keyword_spotter)]
        )
    if sherpa_kws_model_dir is not None:
        command.extend(["--sherpa-kws-model-dir", str(sherpa_kws_model_dir)])
    if debug:
        command.append("--debug")
    subprocess.run(command, cwd=APP_ROOT, check=True)


def main() -> int:
    """Prepare the locked managed runtime for the selected host target.

    Returns
    -------
    int
        Process exit status.
    """

    args = parse_args()
    target = resolve_target_triple(args.target_triple)
    prepare_runtime(
        target,
        args.cache_dir.expanduser().resolve(),
        args.debug,
        args.sherpa_keyword_spotter,
        args.sherpa_kws_model_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
