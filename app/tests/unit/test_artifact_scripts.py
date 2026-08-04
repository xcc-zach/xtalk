"""Unit tests for immutable artifact preparation and verification."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import subprocess
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
    with pytest.raises(ValueError, match="sherpa_onnx_tts"):
        module.validate_required_wheel_modules(wheel)

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("xtalk/models/tts/sherpa_onnx_tts.py", "")
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


def test_macos_packager_rejects_external_bundle_links(
    tmp_path: Path,
) -> None:
    """Prevent release bundles from linking back to the build machine."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    external = tmp_path / "build-only-runtime"
    external.write_bytes(b"runtime")
    link = app / "Contents" / "Frameworks" / "runtime"
    link.parent.mkdir(parents=True)
    link.symlink_to(external)

    with pytest.raises(ValueError, match="external or broken link"):
        module.verify_internal_bundle_links(app)


def test_macos_packager_smoke_tests_matcha_without_external_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch Matcha exactly as a published signed App will launch it."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    runtime = app / "Contents" / "MacOS" / "matcha-model-runtime"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run_runtime(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="help", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run_runtime)

    module.smoke_test_matcha_runtime(app)

    assert calls == [
        (
            [str(runtime), "--help"],
            {
                "capture_output": True,
                "text": True,
                "check": False,
            },
        )
    ]


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


def test_managed_runtime_finds_shared_sherpa_libraries(
    tmp_path: Path,
) -> None:
    """Stage sherpa's shared API without treating ONNX Runtime as sherpa."""

    module = load_script("prepare_managed_runtime")
    (tmp_path / "libsherpa-onnx-c-api.dylib").write_bytes(b"c-api")
    (tmp_path / "libsherpa-onnx-cxx-api.dylib").write_bytes(b"cxx-api")
    (tmp_path / "libonnxruntime.1.27.0.dylib").write_bytes(b"ort")

    libraries = module.sherpa_shared_libraries(
        tmp_path,
        "aarch64-apple-darwin",
    )

    assert [path.name for path in libraries] == [
        "libsherpa-onnx-c-api.dylib",
        "libsherpa-onnx-cxx-api.dylib",
    ]


def test_matcha_macos_build_embeds_packaged_runtime_rpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let the signed Matcha sidecar resolve packaged sherpa libraries."""

    module = load_script("prepare_managed_runtime")
    commands: list[tuple[list[str], dict[str, str]]] = []

    def record_command(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> None:
        del cwd, check
        commands.append((command, env))

    monkeypatch.setattr(module.subprocess, "run", record_command)
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=apple-m1")

    module.build_matcha_runtime(
        "aarch64-apple-darwin",
        False,
        tmp_path,
    )

    assert len(commands) == 1
    assert commands[0][1]["RUSTFLAGS"] == (
        "-C target-cpu=apple-m1 "
        "-C link-arg=-Wl,-rpath,"
        "@executable_path/../Resources/managed-runtime/ort"
    )


def test_managed_runtime_reset_preserves_readme(tmp_path: Path) -> None:
    """Remove stale generated runtimes without deleting tracked guidance."""

    module = load_script("prepare_managed_runtime")
    (tmp_path / "README.md").write_text("tracked", encoding="utf-8")
    (tmp_path / "libonnxruntime.1.28.0.dylib").write_bytes(b"stale")

    module.reset_generated_runtime_files(tmp_path)

    assert (tmp_path / "README.md").is_file()
    assert not (tmp_path / "libonnxruntime.1.28.0.dylib").exists()


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


def test_matcha_local_models_example_uses_sherpa_tts_client() -> None:
    """Keep the Matcha managed example aligned with its HTTP client."""

    example = json.loads(
        (APP_ROOT / "examples" / "local_models_matcha.json").read_text(
            encoding="utf-8"
        )
    )

    assert example["tts"] == {
        "type": "SherpaOnnxTTS",
        "params": {"base_url": "managed://matcha-icefall-zh-en"},
    }
