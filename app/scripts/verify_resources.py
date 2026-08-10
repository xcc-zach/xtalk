"""Verify immutable desktop build artifacts and packaged configuration."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_artifacts import APP_ROOT, MANIFEST_PATH, sha256_file


REQUIRED_KINDS = {"xtalk-wheel", "xtalk-client"}
AUDIO_MODEL_MANIFEST_PATH = (
    APP_ROOT / "resources" / "manifests" / "audio-models.lock.json"
)
REQUIRED_AUDIO_MODEL_IDS = {"silero-vad"}
MANAGED_MODEL_MANIFEST_PATH = (
    APP_ROOT / "resources" / "manifests" / "managed-models.lock.json"
)
NATIVE_RUNTIME_MANIFEST_PATH = (
    APP_ROOT / "resources" / "manifests" / "native-runtimes.lock.json"
)
MTD_SOURCE_MANIFEST_PATH = (
    APP_ROOT
    / "resources"
    / "manifests"
    / "moss-transcribe-runtime.lock.json"
)
MTD_LICENSE_PATH = (
    APP_ROOT / "resources" / "licenses" / "moss-transcribe-cpp-LICENSE.txt"
)
MTD_EXAMPLE_PATH = APP_ROOT / "examples" / "local_models_mtd.json"
REQUIRED_MANAGED_MODEL_IDS = {
    "agentic-asr-refiner",
    "agentic-asr-refiner-mlx",
    "matcha-icefall-zh-en",
    "sensevoice-small",
    "sensevoice-small-mlx",
    "moss-tts-nano",
    "moss-tts-nano-mlx",
    "moss-transcribe-diarize",
}
REQUIRED_NATIVE_RUNTIME_TARGETS = {
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-gnu",
    "aarch64-pc-windows-msvc",
    "x86_64-pc-windows-msvc",
}
REQUIRED_BUILTIN_TOOL_IDS = {"current_time", "web_search"}
BUILTIN_TOOLS_PATH = APP_ROOT / "resources" / "tools"
CREDENTIALS_PATH = APP_ROOT / "resources" / "credentials.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, object]:
    """Load a core artifact lock manifest.

    Parameters
    ----------
    path : pathlib.Path, optional
        Manifest path.

    Returns
    -------
    dict[str, object]
        Parsed manifest.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest root must be an object")
    return payload


def resolve_app_relative(value: str) -> Path:
    """Resolve a manifest path while preventing directory escape.

    Parameters
    ----------
    value : str
        POSIX-style app-relative path.

    Returns
    -------
    pathlib.Path
        Resolved path inside ``app/``.
    """

    candidate = (APP_ROOT / value).resolve()
    if not candidate.is_relative_to(APP_ROOT):
        raise ValueError(f"artifact path escapes app/: {value}")
    return candidate


def verify_manifest(payload: dict[str, object]) -> None:
    """Validate artifact records and file hashes.

    Parameters
    ----------
    payload : dict[str, object]
        Parsed lock manifest.
    """

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported artifact manifest schema")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be an array")

    seen: set[str] = set()
    for raw_record in artifacts:
        if not isinstance(raw_record, dict):
            raise ValueError("artifact record must be an object")
        kind = raw_record.get("kind")
        filename = raw_record.get("filename")
        expected_hash = raw_record.get("sha256")
        version = raw_record.get("version")
        if not all(
            isinstance(value, str) and value
            for value in (kind, filename, expected_hash, version)
        ):
            raise ValueError(
                "artifact records require kind, filename, sha256, and version"
            )
        path = resolve_app_relative(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {filename}")
        seen.add(kind)

    missing = REQUIRED_KINDS - seen
    if missing:
        raise ValueError(f"missing required artifacts: {', '.join(sorted(missing))}")


def verify_audio_model_manifest(
    path: Path = AUDIO_MODEL_MANIFEST_PATH,
) -> None:
    """Validate immutable local audio-model records and file hashes.

    Parameters
    ----------
    path : pathlib.Path, optional
        Audio-model lock manifest path.

    Raises
    ------
    ValueError
        Raised when the manifest shape, required records, or model hash is
        invalid.
    FileNotFoundError
        Raised when a model or its license file is absent.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported audio-model manifest schema")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("audio models must be an array")

    seen: set[str] = set()
    for raw_record in models:
        if not isinstance(raw_record, dict):
            raise ValueError("audio-model record must be an object")
        model_id = raw_record.get("id")
        filename = raw_record.get("filename")
        expected_hash = raw_record.get("sha256")
        license_name = raw_record.get("license")
        license_filename = raw_record.get("license_file")
        source = raw_record.get("source")
        version = raw_record.get("version")
        if not all(
            isinstance(value, str) and value
            for value in (
                model_id,
                filename,
                expected_hash,
                license_name,
                license_filename,
                source,
                version,
            )
        ):
            raise ValueError(
                "audio-model records require id, filename, sha256, license, "
                "license_file, source, and version"
            )
        model_path = resolve_app_relative(filename)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if sha256_file(model_path) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {filename}")
        license_path = resolve_app_relative(license_filename)
        if not license_path.is_file():
            raise FileNotFoundError(license_path)
        seen.add(model_id)

    missing = REQUIRED_AUDIO_MODEL_IDS - seen
    if missing:
        raise ValueError(
            f"missing required audio models: {', '.join(sorted(missing))}"
        )


def verify_managed_model_manifest(
    path: Path = MANAGED_MODEL_MANIFEST_PATH,
) -> None:
    """Validate optional runtime-download records without downloading weights.

    Parameters
    ----------
    path : pathlib.Path, optional
        Managed-model lock manifest path.

    Raises
    ------
    ValueError
        Raised when the schema, identifiers, file paths, or integrity metadata
        is invalid.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported managed-model manifest schema")
    services = payload.get("services")
    if not isinstance(services, list):
        raise ValueError("managed model services must be an array")

    seen: set[str] = set()
    for raw_service in services:
        if not isinstance(raw_service, dict):
            raise ValueError("managed service record must be an object")
        service_id = raw_service.get("id")
        version = raw_service.get("version")
        files = raw_service.get("files")
        if (
            not isinstance(service_id, str)
            or not service_id
            or not isinstance(version, str)
            or not version
            or not isinstance(files, list)
            or not files
        ):
            raise ValueError(
                "managed service records require id, version, and files"
            )
        if service_id in seen:
            raise ValueError(f"duplicate managed service: {service_id}")
        seen.add(service_id)
        service_paths: set[str] = set()
        for raw_file in files:
            if not isinstance(raw_file, dict):
                raise ValueError("managed model file record must be an object")
            relative_path = raw_file.get("path")
            source = raw_file.get("url")
            expected_hash = raw_file.get("sha256")
            size = raw_file.get("size")
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
            ):
                raise ValueError("managed model file path is unsafe")
            if relative_path in service_paths:
                raise ValueError(
                    f"duplicate managed model file path: {relative_path}"
                )
            service_paths.add(relative_path)
            if not isinstance(source, str) or not source.startswith("https://"):
                raise ValueError("managed model sources must use HTTPS")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_hash
                )
            ):
                raise ValueError("managed model SHA-256 is invalid")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError("managed model file size is invalid")

        archives = raw_service.get("archives", [])
        if not isinstance(archives, list):
            raise ValueError("managed service archives must be an array")
        for raw_archive in archives:
            if not isinstance(raw_archive, dict):
                raise ValueError("managed archive record must be an object")
            archive_path = raw_archive.get("path")
            if archive_path not in service_paths:
                raise ValueError(
                    "managed archive must reference a downloaded service file"
                )
            if raw_archive.get("format") != "tar-bz2":
                raise ValueError("managed archive format is unsupported")

        required_paths = raw_service.get("required_paths", [])
        if not isinstance(required_paths, list):
            raise ValueError("managed required_paths must be an array")
        for relative_path in required_paths:
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
            ):
                raise ValueError("managed required path is unsafe")

    missing = REQUIRED_MANAGED_MODEL_IDS - seen
    if missing:
        raise ValueError(
            f"missing managed models: {', '.join(sorted(missing))}"
        )


def verify_native_runtime_manifest(
    path: Path = NATIVE_RUNTIME_MANIFEST_PATH,
) -> None:
    """Validate locked Sherpa/ORT archives for desktop build targets.

    Parameters
    ----------
    path : pathlib.Path, optional
        Native runtime lock manifest path.

    Raises
    ------
    ValueError
        Raised when a target, immutable URL, or SHA-256 is invalid.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported native-runtime manifest schema")
    version = payload.get("sherpa_onnx_version")
    targets = payload.get("targets")
    if not isinstance(version, str) or not version or not isinstance(targets, dict):
        raise ValueError("native-runtime manifest requires a version and targets")
    missing = REQUIRED_NATIVE_RUNTIME_TARGETS - set(targets)
    if missing:
        raise ValueError(
            f"missing native runtime targets: {', '.join(sorted(missing))}"
        )
    release_prefix = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        f"v{version}/"
    )
    for target, raw_record in targets.items():
        if (
            not isinstance(target, str)
            or not target
            or not isinstance(raw_record, dict)
            or set(raw_record) != {"archive", "sha256", "url"}
        ):
            raise ValueError("native-runtime target record is invalid")
        archive = raw_record["archive"]
        source = raw_record["url"]
        expected_hash = raw_record["sha256"]
        if (
            not isinstance(archive, str)
            or Path(archive).name != archive
            or not isinstance(source, str)
            or source != release_prefix + archive
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash
            )
        ):
            raise ValueError(f"native-runtime record for {target} is invalid")


def verify_mtd_source_manifest(
    path: Path = MTD_SOURCE_MANIFEST_PATH,
) -> None:
    """Validate immutable moss-transcribe.cpp and ggml source revisions.

    Parameters
    ----------
    path : pathlib.Path, optional
        MTD native source lock path.

    Raises
    ------
    ValueError
        Raised when either source record is missing, mutable, or malformed.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources") if isinstance(payload, dict) else None
    repositories = {
        "moss-transcribe.cpp": "localai-org/moss-transcribe.cpp",
        "ggml": "ggml-org/ggml",
    }
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(sources, dict)
    ):
        raise ValueError("unsupported MTD source manifest schema")
    if set(sources) != set(repositories):
        raise ValueError("MTD source manifest is incomplete")
    for name, repository in repositories.items():
        record = sources[name]
        if not isinstance(record, dict) or set(record) != {
            "archive",
            "revision",
            "sha256",
            "url",
        }:
            raise ValueError(f"MTD source record for {name} is invalid")


def verify_mtd_packaging() -> None:
    """Require the MTD sidecar, license, and example in Tauri bundles.

    Raises
    ------
    FileNotFoundError
        Raised when the tracked license or example is absent.
    ValueError
        Raised when Tauri does not declare the MTD external binary and
        resource destinations.
    """

    for path in (MTD_LICENSE_PATH, MTD_EXAMPLE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    tauri_config = json.loads(
        (APP_ROOT / "src-tauri" / "tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = tauri_config.get("bundle", {})
    external_binaries = bundle.get("externalBin", [])
    resources = bundle.get("resources", {})
    if "binaries/mtd-model-runtime" not in external_binaries:
        raise ValueError("Tauri bundle must package the MTD sidecar")
    if (
        resources.get("../resources/licenses/moss-transcribe-cpp-LICENSE.txt")
        != "licenses/moss-transcribe-cpp-LICENSE.txt"
        or resources.get("../examples/local_models_mtd.json")
        != "examples/local_models_mtd.json"
    ):
        raise ValueError("Tauri bundle must package MTD resources")
        revision = record["revision"]
        archive = record["archive"]
        source = record["url"]
        expected_hash = record["sha256"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
            or not isinstance(archive, str)
            or Path(archive).name != archive
            or source
            != f"https://github.com/{repository}/archive/{revision}.tar.gz"
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash
            )
        ):
            raise ValueError(f"MTD source record for {name} is invalid")


def verify_no_bundled_default_config() -> None:
    """Reject a release bundle that contains a default model configuration."""

    config_path = APP_ROOT / "resources" / "config" / "default.json"
    if config_path.exists():
        raise ValueError("release resources must not contain config/default.json")

    tauri_config_path = APP_ROOT / "src-tauri" / "tauri.conf.json"
    tauri_config = json.loads(tauri_config_path.read_text(encoding="utf-8"))
    resources = tauri_config.get("bundle", {}).get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("Tauri bundle resources must be an object")
    if any(
        "config/default.json" in str(source)
        or "config/default.json" in str(destination)
        for source, destination in resources.items()
    ):
        raise ValueError("Tauri bundle must not package config/default.json")


def verify_builtin_tools_and_credentials() -> None:
    """Validate the publishable built-in tool and credential registries.

    Raises
    ------
    ValueError
        Raised when a registry is malformed, required resources are absent,
        or Tauri does not bundle them at their runtime paths.
    FileNotFoundError
        Raised when a declared built-in tool file is absent.
    """

    catalog = json.loads(
        (BUILTIN_TOOLS_PATH / "builtin_tools.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(catalog, dict)
        or set(catalog) != {"version", "tools"}
        or catalog["version"] != 1
        or not isinstance(catalog["tools"], list)
    ):
        raise ValueError("built-in tool catalog is invalid")
    seen_tools: set[str] = set()
    for entry in catalog["tools"]:
        if (
            not isinstance(entry, dict)
            or not {"id", "path", "enabled_by_default"}.issubset(entry)
            or not set(entry).issubset(
                {"id", "path", "enabled_by_default", "can_disable"}
            )
        ):
            raise ValueError("built-in tool catalog entry is invalid")
        identifier = entry["id"]
        relative_path = entry["path"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in seen_tools
            or not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(entry["enabled_by_default"], bool)
            or not isinstance(entry.get("can_disable", True), bool)
        ):
            raise ValueError("built-in tool catalog entry is invalid")
        seen_tools.add(identifier)
        tool_directory = BUILTIN_TOOLS_PATH / relative_path
        manifest_path = tool_directory / "xtalk_tool.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entrypoint = (
            manifest.get("entrypoint")
            if isinstance(manifest, dict)
            else None
        )
        if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
            raise ValueError("built-in tool entrypoint is invalid")
        module_name, factory_name = entrypoint.split(":")
        if not module_name or not factory_name:
            raise ValueError("built-in tool entrypoint is invalid")
        module_path = tool_directory / f"{module_name}.py"
        if not module_path.is_file():
            raise FileNotFoundError(module_path)

    missing_tools = REQUIRED_BUILTIN_TOOL_IDS - seen_tools
    if missing_tools:
        raise ValueError(
            f"missing required built-in tools: {', '.join(sorted(missing_tools))}"
        )
    current_time = next(
        entry for entry in catalog["tools"] if entry["id"] == "current_time"
    )
    if current_time.get("can_disable", True):
        raise ValueError("the current-time built-in must remain enabled")

    credentials = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(credentials, dict)
        or set(credentials) != {"version", "credentials", "bindings"}
        or credentials["version"] != 1
        or not isinstance(credentials["credentials"], list)
        or not isinstance(credentials["bindings"], list)
    ):
        raise ValueError("credential registry is invalid")
    serialized_credentials = json.dumps(credentials).lower()
    if any(name in serialized_credentials for name in ('"secret"', '"value"')):
        raise ValueError("credential registry must not contain secret values")
    credential_ids: set[str] = set()
    for definition in credentials["credentials"]:
        allowed_keys = {"id", "display_name", "environment", "inject_environment"}
        inject_environment = definition.get("inject_environment")
        if (
            not isinstance(definition, dict)
            or set(definition) - allowed_keys
            or not isinstance(definition["id"], str)
            or not definition["id"]
            or definition["id"] in credential_ids
            or not isinstance(definition["environment"], list)
            or not definition["environment"]
            or not all(
                isinstance(name, str) and name
                for name in definition["environment"]
            )
            or (
                inject_environment is not None
                and (
                    not isinstance(inject_environment, str)
                    or not inject_environment
                    or inject_environment not in definition["environment"]
                )
            )
        ):
            raise ValueError("credential registry definition is invalid")
        credential_ids.add(definition["id"])
    builtin_ids = {f"builtin:{identifier}" for identifier in seen_tools}
    for binding in credentials["bindings"]:
        if (
            not isinstance(binding, dict)
            or set(binding)
            != {"tool_id", "credential_id", "inject_environment"}
            or binding["tool_id"] not in builtin_ids
            or binding["credential_id"] not in credential_ids
            or not isinstance(binding["inject_environment"], str)
            or not binding["inject_environment"]
        ):
            raise ValueError("credential registry binding is invalid")

    tauri_config = json.loads(
        (APP_ROOT / "src-tauri" / "tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    resources = tauri_config.get("bundle", {}).get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("Tauri bundle resources must be an object")
    if resources.get("../resources/tools/") != "tools/":
        raise ValueError("Tauri bundle must package the built-in tool directory")
    if resources.get("../resources/credentials.json") != "credentials.json":
        raise ValueError("Tauri bundle must package the credential registry")


def main() -> int:
    """Run all resource checks.

    Returns
    -------
    int
        Process exit status.
    """

    verify_no_bundled_default_config()
    verify_manifest(load_manifest())
    verify_audio_model_manifest()
    verify_managed_model_manifest()
    verify_native_runtime_manifest()
    verify_mtd_source_manifest()
    verify_mtd_packaging()
    verify_builtin_tools_and_credentials()
    print("resource verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
