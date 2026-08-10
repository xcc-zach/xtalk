"""Unit tests for immutable artifact preparation and verification."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import sys
import zipfile
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


def test_managed_model_manifest_pins_optional_downloads() -> None:
    """Validate managed model URLs, sizes, and SHA-256 records."""

    module = load_script("verify_resources")

    module.verify_managed_model_manifest()


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
    assert requirement == f"{Path('/tmp/xtalk.whl')}[ali,testing]"


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


def test_backend_build_requires_managed_model_client_modules(
    tmp_path: Path,
) -> None:
    """Reject a stale wheel that omits a managed-model client adapter."""

    module = load_script("build_backend")
    wheel = tmp_path / "xtalk.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("xtalk/models/asr/sherpa_onnx_asr.py", "")

    with pytest.raises(ValueError, match="moss_tts_nano"):
        module.validate_required_wheel_modules(wheel)

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("xtalk/models/tts/moss_tts_nano.py", "")
    module.validate_required_wheel_modules(wheel)


def test_macos_packager_links_framework_metadata_to_resources(
    tmp_path: Path,
) -> None:
    """Keep Python package metadata available without nested signing bundles."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    frameworks = app / "Contents" / "Frameworks"
    resources = app / "Contents" / "Resources" / "app-backend-runtime"
    metadata_name = "example-1.0.dist-info"
    (frameworks / metadata_name).mkdir(parents=True)
    (resources / metadata_name).mkdir(parents=True)
    (frameworks / metadata_name / "METADATA").write_text(
        "duplicate",
        encoding="utf-8",
    )
    (resources / metadata_name / "METADATA").write_text(
        "canonical",
        encoding="utf-8",
    )

    linked = module.link_python_metadata_to_resources(app)

    assert linked == [frameworks / metadata_name]
    assert (frameworks / metadata_name).is_symlink()
    assert (frameworks / metadata_name / "METADATA").read_text(
        encoding="utf-8"
    ) == "canonical"


def test_macos_packager_signs_both_codex_hosts_with_v8_entitlements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve V8 executable-memory rights in both runtime layouts."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    relative_host = Path("codex_cli_bin/bin/codex-code-mode-host")
    hosts = [
        app / "Contents" / "Frameworks" / relative_host,
        app
        / "Contents"
        / "Resources"
        / "app-backend-runtime"
        / relative_host,
    ]
    for host in hosts:
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_bytes(b"host")
    backend = app / "Contents" / "MacOS" / "app-backend"
    backend.parent.mkdir(parents=True)
    backend.write_bytes(b"backend")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", commands.append)

    module.sign_app(app, "-")

    codex_commands = [
        command
        for command in commands
        if str(module.CODEX_HOST_ENTITLEMENTS) in command
    ]
    assert [Path(command[-1]) for command in codex_commands] == hosts
    assert all(
        "--options" in command and "runtime" in command
        for command in codex_commands
    )


def test_codex_host_entitlements_allow_v8_executable_memory() -> None:
    """Keep every hardened-runtime entitlement required by the V8 host."""

    payload = plistlib.loads(
        (APP_ROOT / "src-tauri" / "CodexHostEntitlements.plist").read_bytes()
    )
    assert payload == {
        "com.apple.security.cs.allow-jit": True,
        "com.apple.security.cs.allow-unsigned-executable-memory": True,
        "com.apple.security.cs.disable-library-validation": True,
    }


def test_managed_runtime_uses_target_specific_binary_names() -> None:
    """Name managed sidecars according to Tauri external-bin conventions."""

    module = load_script("prepare_managed_runtime")

    assert module.executable_name("runtime", "aarch64-apple-darwin") == "runtime"
    assert (
        module.executable_name("runtime", "x86_64-pc-windows-msvc")
        == "runtime.exe"
    )
    assert module.target_supports_mlx("aarch64-apple-darwin")
    assert not module.target_supports_mlx("x86_64-unknown-linux-gnu")


def test_managed_runtime_stages_only_onnx_cuda_provider_files(
    tmp_path: Path,
) -> None:
    """Stage the CUDA and shared providers without copying unrelated files."""

    module = load_script("prepare_managed_runtime")
    source = tmp_path / "gpu"
    destination = tmp_path / "staged"
    source.mkdir()
    (source / "libonnxruntime_providers_cuda.so").write_bytes(b"cuda")
    (source / "libonnxruntime_providers_shared.so").write_bytes(b"shared")
    (source / "unrelated.txt").write_text("skip", encoding="utf-8")

    module.copy_cuda_runtime(source, destination)

    assert (destination / "libonnxruntime_providers_cuda.so").is_file()
    assert (destination / "libonnxruntime_providers_shared.so").is_file()
    assert not (destination / "unrelated.txt").exists()


def test_managed_runtime_declares_complete_wake_word_model_layout() -> None:
    """Stage every model file required by the sherpa keyword spotter."""

    module = load_script("prepare_managed_runtime")

    assert module.WAKE_WORD_MODEL_FILES == (
        "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
        "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
        "tokens.txt",
    )
    assert (
        APP_ROOT / "resources" / "models" / "wake-word" / "keywords.txt"
    ).read_text(encoding="utf-8").strip() == (
        "n ǐ h ǎo x iǎo k è :3.0 #0.25 @你好小克"
    )


def test_windows_stages_sherpa_onnx_runtime_beside_sidecars() -> None:
    """Override a conflicting system ONNX Runtime during Windows startup."""

    config = json.loads(
        (APP_ROOT / "src-tauri" / "tauri.windows.conf.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["bundle"]["resources"] == {
        "../resources/managed-runtime/sherpa/onnxruntime.dll": "onnxruntime.dll"
    }


def test_local_models_example_uses_managed_speech_and_shared_llm() -> None:
    """Keep managed examples aligned on their shared LLM configuration."""

    example = json.loads(
        (APP_ROOT / "examples" / "local_models.json").read_text(
            encoding="utf-8"
        )
    )
    mlx_example = json.loads(
        (APP_ROOT / "examples" / "local_models_mlx.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        example["asr"]["params"]["base_url"]
        == "managed://sensevoice-small"
    )
    assert (
        example["tts"]["params"]["base_url"]
        == "managed://moss-tts-nano"
    )
    assert example["llm_agent"] == mlx_example["llm_agent"]


def test_mlx_local_models_example_selects_native_managed_engine() -> None:
    """Keep the Apple Silicon MLX example on the same client protocols."""

    example = json.loads(
        (APP_ROOT / "examples" / "local_models_mlx.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        example["asr"]["params"]["base_url"]
        == "managed://sensevoice-small?backend=mlx"
    )
    assert (
        example["tts"]["params"]["base_url"]
        == "managed://moss-tts-nano?backend=mlx"
    )
    assert example["tts"]["params"]["voices"][0]["path"].startswith(
        "managed://moss-tts-nano/"
    )
