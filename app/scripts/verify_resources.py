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
REQUIRED_MANAGED_MODEL_IDS = {
    "matcha-icefall-zh-en",
    "sensevoice-small",
    "sensevoice-small-mlx",
    "moss-tts-nano",
    "moss-tts-nano-mlx",
}


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
    print("resource verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
