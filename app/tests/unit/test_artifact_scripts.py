"""Unit tests for immutable artifact preparation and verification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    """Load an app script module directly from its source path.

    Parameters
    ----------
    name : str
        Script stem.

    Returns
    -------
    types.ModuleType
        Imported module.
    """

    path = APP_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"app_script_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    """Hash artifacts deterministically."""

    module = load_script("prepare_artifacts")
    path = tmp_path / "artifact.whl"
    path.write_bytes(b"xtalk")
    assert (
        module.sha256_file(path)
        == "703c8acbc22fd912ea01711027576b481da9e1890aeab4b3ed6b6077c640e6c8"
    )


def test_resolve_app_relative_rejects_escape() -> None:
    """Reject manifest paths that escape the app directory."""

    module = load_script("verify_resources")
    with pytest.raises(ValueError, match="escapes app"):
        module.resolve_app_relative("../server_configs/sample.json")


def test_checked_in_audio_model_manifest_matches_packaged_model() -> None:
    """Verify the pinned Silero model record and packaged file digest."""

    module = load_script("verify_resources")

    module.verify_audio_model_manifest()


def test_release_has_no_bundled_default_model_config() -> None:
    """Keep provider configuration external to the release bundle."""

    module = load_script("verify_resources")

    module.verify_no_bundled_default_config()


def test_wheel_requirement_adds_sorted_unique_extras() -> None:
    """Compose optional wheel dependencies without model-specific logic."""

    module = load_script("build_backend")
    requirement = module.wheel_requirement(
        Path("/tmp/xtalk.whl"),
        ["ali", "testing", "ali"],
    )
    assert requirement == "/tmp/xtalk.whl[ali,testing]"


def test_wheel_requirement_rejects_invalid_extra() -> None:
    """Reject requirement injection through an invalid extra name."""

    module = load_script("build_backend")
    with pytest.raises(ValueError, match="invalid XTalk extra"):
        module.wheel_requirement(Path("/tmp/xtalk.whl"), ["ali] --index-url"])


def test_backend_build_requires_silero_vad_dependencies() -> None:
    """Reject a sidecar build that cannot execute the bundled VAD model."""

    module = load_script("build_backend")

    with pytest.raises(ValueError, match="silero-vad"):
        module.validate_required_extras(["ali"])
    module.validate_required_extras(["silero-vad", "ali"])
