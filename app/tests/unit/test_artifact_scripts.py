"""Unit tests for immutable artifact preparation and verification."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
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


def test_source_build_reads_versions_from_fresh_artifacts(tmp_path: Path) -> None:
    """Derive lock metadata from newly built source packages."""

    module = load_script("build_from_source")
    wheel = tmp_path / "xtalk-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "xtalk-9.8.7.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: xtalk\nVersion: 9.8.7\n",
        )

    assert module.wheel_version(wheel) == "9.8.7"
    assert module.parse_npm_pack_path(
        json.dumps([{"filename": "xtalk-client-1.2.3.tgz"}])
    ) == Path("xtalk-client-1.2.3.tgz")


def test_source_build_uses_an_app_local_npm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid user-global npm cache permissions during release builds."""

    module = load_script("build_from_source")
    monkeypatch.setattr(module, "APP_ROOT", tmp_path)

    environment = module.npm_environment()

    assert environment["npm_config_cache"] == str(
        tmp_path / ".build" / "npm-cache"
    )
    assert (tmp_path / ".build" / "npm-cache").is_dir()


def test_source_build_preserves_a_virtual_environment_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the selected virtual environment and its installed packages."""

    module = load_script("build_from_source")
    base_python = tmp_path / "base" / "python3.12"
    environment_python = tmp_path / "venv" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"python")
    environment_python.parent.mkdir(parents=True)
    environment_python.symlink_to(base_python)
    monkeypatch.setattr(module, "python_version", lambda _path: (3, 12))

    assert module.resolve_sidecar_python(environment_python) == environment_python


def test_tauri_release_build_is_wired_to_repository_sources() -> None:
    """Rebuild root Python and frontend inputs before each Tauri package."""

    package = json.loads((APP_ROOT / "package.json").read_text(encoding="utf-8"))
    tauri = json.loads(
        (APP_ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )

    assert package["dependencies"]["xtalk-client"] == "file:../frontend"
    assert package["scripts"]["build:source"].startswith(
        "python3 scripts/build_from_source.py"
    )
    assert package["scripts"]["package:macos"] == (
        "python3 scripts/package_macos_release.py"
    )
    assert package["scripts"]["package:macos:local"] == (
        "python3 scripts/package_macos_release.py --local"
    )
    assert tauri["build"]["beforeBuildCommand"] == "npm run build:source"


def test_macos_local_build_entrypoint_checks_prerequisites() -> None:
    """Require prerequisite validation before the packaging entrypoint."""

    entrypoint = APP_ROOT / "scripts" / "build_macos_local.sh"
    logic = entrypoint.read_text(encoding="utf-8")

    assert logic.startswith("#!/usr/bin/env bash")
    assert "--check-only" in logic
    assert "npm run package:macos:local" in logic
    assert "Toolchain components are never installed automatically" in logic
    assert entrypoint.stat().st_mode & 0o111


def test_distribution_packaging_requires_external_apple_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a public release that would silently fall back to ad-hoc signing."""

    module = load_script("package_macos_release")
    monkeypatch.delenv("APPLE_SIGNING_IDENTITY", raising=False)
    monkeypatch.delenv("APPLE_NOTARY_KEYCHAIN_PROFILE", raising=False)

    with pytest.raises(ValueError, match="APPLE_SIGNING_IDENTITY"):
        module.distribution_credentials()

    monkeypatch.setenv(
        "APPLE_SIGNING_IDENTITY",
        "Developer ID Application: XTalk (TEAMID)",
    )
    with pytest.raises(ValueError, match="APPLE_NOTARY_KEYCHAIN_PROFILE"):
        module.distribution_credentials()


def test_failed_macos_release_removes_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never leave an invalid App or DMG at a publishable output path."""

    module = load_script("package_macos_release")
    app = tmp_path / "XTalk.app"
    output = tmp_path / "XTalk.dmg"
    monkeypatch.setattr(module, "APP_OUTPUT", app)

    def fail_build(*args: object, **kwargs: object) -> None:
        del args, kwargs
        app.mkdir(parents=True)
        output.write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, ["npm"])

    monkeypatch.setattr(module.subprocess, "run", fail_build)

    with pytest.raises(subprocess.CalledProcessError):
        module.run_release(
            local=True,
            output=output,
            identity="-",
            notary_profile=None,
        )

    assert not app.exists()
    assert not output.exists()


def test_macos_release_prepares_runtime_before_tauri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make native runtime preparation part of the packaging entrypoint."""

    module = load_script("package_macos_release")
    app = tmp_path / "XTalk.app"
    output = tmp_path / "XTalk.dmg"
    events: list[str] = []
    monkeypatch.setattr(module, "APP_OUTPUT", app)
    monkeypatch.setattr(
        module,
        "prepare_native_runtime",
        lambda: events.append("runtime"),
    )

    def complete_command(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        events.append(Path(command[0]).name)
        if command[0] == "npm":
            app.mkdir(parents=True)
        else:
            output.write_bytes(b"dmg")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", complete_command)

    module.run_release(
        local=True,
        output=output,
        identity="-",
        notary_profile=None,
    )

    assert events[:2] == ["runtime", "npm"]
    assert app.is_dir()
    assert output.is_file()


def test_mlx_runtime_uses_the_top_level_product_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore duplicate MLX resource bundles nested in test products."""

    module = load_script("prepare_managed_runtime")
    package = tmp_path / "mlx-runtime"
    products = package / ".build" / "xcode" / "Build" / "Products" / "Release"
    executable = products / "xtalk-mlx-model-runtime"
    bundle = products / "mlx-swift_Cmlx.bundle"
    nested_bundle = (
        products
        / "XTalkMLXRuntimeTests.xctest"
        / "Contents"
        / "Resources"
        / "mlx-swift_Cmlx.bundle"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    bundle.mkdir()
    nested_bundle.mkdir(parents=True)
    monkeypatch.setattr(module, "MLX_RUNTIME_PACKAGE", package)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
    )

    assert module.build_mlx_runtime(debug=False) == (executable, bundle)


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


def test_release_bundles_builtin_tools_and_secret_free_credential_registry() -> None:
    """Keep required built-ins and credential metadata in every package."""

    module = load_script("verify_resources")

    module.verify_builtin_tools_and_credentials()


def test_native_credentials_pin_each_supported_platform_backend() -> None:
    """Pin the credential abstraction to each operating-system store backend."""

    manifest = (APP_ROOT / "src-tauri" / "Cargo.toml").read_text(
        encoding="utf-8"
    )

    assert "features = [\"apple-native\"]" in manifest
    assert "features = [\"windows-native\"]" in manifest
    assert (
        "features = [\"crypto-rust\", \"sync-secret-service\"]"
        in manifest
    )
    assert manifest.count('keyring = { version = "=3.6.3"') == 3


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


def test_sidecar_lock_is_exact_except_for_codex() -> None:
    """Lock the frozen Python graph while selecting Codex at build time."""

    module = load_script("build_backend")

    requirements = module.validate_sidecar_lock()

    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert not any(
        requirement.lower().startswith("openai-codex")
        for requirement in requirements
    )
    pyproject = (APP_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"openai-codex"' in pyproject
    assert "openai-codex==" not in pyproject


def test_backend_freezer_excludes_bundled_codex_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package the Codex SDK while requiring a user-installed CLI."""

    module = load_script("build_backend")
    build_root = tmp_path / "backend"
    monkeypatch.setattr(module, "BUILD_ROOT", build_root)
    commands: list[list[str]] = []

    def build(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        output = build_root / "dist" / "app-backend"
        (output / "app-backend-runtime").mkdir(parents=True)
        (output / "app-backend").write_bytes(b"backend")

    monkeypatch.setattr(module, "run", build)

    module.build_onedir(Path("/python3.12"), "aarch64-apple-darwin")

    command = commands[0]
    assert "openai_codex" in command
    assert "codex_cli_bin" in command
    codex_index = command.index("codex_cli_bin")
    assert command[codex_index - 1] == "--exclude-module"


def test_native_runtime_lock_supports_common_desktop_targets() -> None:
    """Pin official verified Sherpa/ORT bundles for desktop platforms."""

    module = load_script("download_managed_runtime")
    targets = {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "aarch64-unknown-linux-gnu",
        "x86_64-unknown-linux-gnu",
        "aarch64-pc-windows-msvc",
        "x86_64-pc-windows-msvc",
    }

    for target in targets:
        record = module.load_target_record(target)
        assert len(record["sha256"]) == 64
        assert record["archive"] in record["url"]
    verifier = load_script("verify_resources")
    verifier.verify_native_runtime_manifest()


def test_mtd_runtime_locks_both_native_source_revisions() -> None:
    """Pin moss-transcribe.cpp and its ggml submodule independently."""

    module = load_script("download_managed_runtime")
    records = module.load_mtd_source_records()

    assert set(records) == {"moss-transcribe.cpp", "ggml"}
    assert all(len(record["revision"]) == 40 for record in records.values())
    verifier = load_script("verify_resources")
    verifier.verify_mtd_source_manifest()
    verifier.verify_mtd_packaging()
    verifier.verify_campplus_packaging()


def test_native_runtime_download_context_keeps_tls_verification() -> None:
    """Use a trusted CA bundle without disabling certificate verification."""

    module = load_script("download_managed_runtime")

    context = module.verified_tls_context()

    assert context.check_hostname is True
    assert context.verify_mode.name == "CERT_REQUIRED"


def test_native_runtime_archive_materializes_only_internal_links(
    tmp_path: Path,
) -> None:
    """Materialize loader aliases without trusting external symlinks."""

    module = load_script("download_managed_runtime")
    source = tmp_path / "server"
    source.write_bytes(b"runtime")
    archive_path = tmp_path / "runtime.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        archive.add(source, arcname="runtime/bin/server")
        archive.add(source, arcname="runtime/lib/libonnxruntime.so.1.27.0")
        internal_link = tarfile.TarInfo("runtime/lib/libonnxruntime.so.1")
        internal_link.type = tarfile.SYMTYPE
        internal_link.linkname = "libonnxruntime.so.1.27.0"
        archive.addfile(internal_link)
        link = tarfile.TarInfo("runtime/lib/external")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/external"
        archive.addfile(link)

    destination = tmp_path / "extracted"
    module.extract_regular_files(archive_path, destination)

    assert (destination / "runtime" / "bin" / "server").read_bytes() == b"runtime"
    assert (
        destination / "runtime" / "lib" / "libonnxruntime.so.1"
    ).read_bytes() == b"runtime"
    assert not (destination / "runtime" / "lib" / "external").exists()


def test_mtd_source_assembly_patches_portable_cpu_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assemble the ggml submodule and disable build-machine CPU tuning."""

    module = load_script("download_managed_runtime")
    moss_root = tmp_path / "moss-source"
    ggml_root = tmp_path / "ggml-source"
    moss_root.mkdir()
    ggml_root.mkdir()
    (moss_root / "CMakeLists.txt").write_text(
        'set(GGML_NATIVE ON CACHE BOOL "" FORCE)\n'
        'set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)\n',
        encoding="utf-8",
    )
    (ggml_root / "CMakeLists.txt").write_text("project(ggml)\n", encoding="utf-8")
    archives: dict[str, Path] = {}
    for name, source in (
        ("moss.tar.gz", moss_root),
        ("ggml.tar.gz", ggml_root),
    ):
        archive_path = tmp_path / name
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(source, arcname=source.name)
        archives[name] = archive_path
    records = {
        "moss-transcribe.cpp": {
            "archive": "moss.tar.gz",
            "revision": "1" * 40,
            "sha256": "a" * 64,
            "url": "https://example.invalid/moss.tar.gz",
        },
        "ggml": {
            "archive": "ggml.tar.gz",
            "revision": "2" * 40,
            "sha256": "b" * 64,
            "url": "https://example.invalid/ggml.tar.gz",
        },
    }
    monkeypatch.setattr(module, "load_mtd_source_records", lambda: records)
    monkeypatch.setattr(
        module,
        "download_verified",
        lambda record, _cache: archives[record["archive"]],
    )

    source = module.assemble_mtd_source(tmp_path / "cache")

    cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "if(NOT DEFINED GGML_NATIVE)" in cmake
    assert "if(NOT DEFINED GGML_LLAMAFILE)" in cmake
    assert "FORCE" not in cmake
    assert (source / "third_party" / "ggml" / "CMakeLists.txt").is_file()


def test_native_runtime_locates_colocated_macos_sherpa_and_ort(
    tmp_path: Path,
) -> None:
    """Use the server and ORT from the same verified Sherpa archive."""

    module = load_script("download_managed_runtime")
    server = tmp_path / "bin" / "sherpa-onnx-offline-websocket-server"
    library = tmp_path / "lib"
    server.parent.mkdir()
    library.mkdir()
    server.write_bytes(b"server")
    (library / "libsherpa-onnx-c-api.dylib").write_bytes(b"sherpa")
    ort = library / "libonnxruntime.1.27.0.dylib"
    ort.write_bytes(b"ort")

    actual_server, actual_library, actual_ort = module.locate_runtime_inputs(
        tmp_path,
        "aarch64-apple-darwin",
    )

    assert actual_server == server
    assert actual_library == library
    assert actual_ort == ort


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


def test_macos_packager_prunes_duplicate_resource_runtime(
    tmp_path: Path,
) -> None:
    """Keep only metadata in Resources after validating both layouts."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    frameworks = app / "Contents" / "Frameworks"
    resources = app / "Contents" / "Resources" / "app-backend-runtime"
    metadata_name = "example-1.0.dist-info"
    for root in (frameworks, resources):
        (root / metadata_name).mkdir(parents=True)
        (root / metadata_name / "METADATA").write_text(
            "metadata",
            encoding="utf-8",
        )
        (root / "package").mkdir()
        (root / "package" / "module.py").write_text(
            "value = 1\n",
            encoding="utf-8",
        )
        (root / "libpython3.12.dylib").write_bytes(b"python")

    removed_count, removed_bytes = module.prune_duplicate_resource_runtime(app)

    assert removed_count == 2
    assert removed_bytes > 0
    assert (frameworks / "package" / "module.py").is_file()
    assert (frameworks / "libpython3.12.dylib").is_file()
    assert sorted(path.name for path in resources.iterdir()) == [metadata_name]


def test_macos_packager_rejects_different_runtime_layouts(
    tmp_path: Path,
) -> None:
    """Do not remove a Resources runtime that contains a unique entry."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    frameworks = app / "Contents" / "Frameworks"
    resources = app / "Contents" / "Resources" / "app-backend-runtime"
    frameworks.mkdir(parents=True)
    resources.mkdir(parents=True)
    (frameworks / "shared").write_bytes(b"framework")
    (resources / "shared").write_bytes(b"resource")
    (resources / "unique").write_bytes(b"resource-only")

    with pytest.raises(ValueError, match="runtime layouts differ"):
        module.prune_duplicate_resource_runtime(app)


def test_macos_packager_does_not_require_a_bundled_codex_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sign only code owned by the App when Codex is user-installed."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    backend = app / "Contents" / "MacOS" / "app-backend"
    backend.parent.mkdir(parents=True)
    backend.write_bytes(b"backend")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", commands.append)

    module.sign_app(app, "-")

    assert all("codex_cli_bin" not in " ".join(command) for command in commands)
    assert any(command[-1] == str(backend) for command in commands)


def test_macos_packager_explicitly_signs_managed_runtime_libraries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair invalid upstream dylib signatures before signing the App."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    backend = app / "Contents" / "MacOS" / "app-backend"
    runtime = app / "Contents" / "Resources" / "managed-runtime" / "ort"
    library = runtime / "libonnxruntime.1.27.0.dylib"
    backend.parent.mkdir(parents=True)
    backend.write_bytes(b"backend")
    runtime.mkdir(parents=True)
    library.write_bytes(b"runtime")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", commands.append)

    module.sign_app(app, "-")

    library_sign = next(
        index
        for index, command in enumerate(commands)
        if command[-1] == str(library) and "--sign" in command
    )
    app_sign = next(
        index
        for index, command in enumerate(commands)
        if command[-1] == str(app) and "--sign" in command
    )
    assert library_sign < app_sign
    assert ["codesign", "--verify", "--strict", str(library)] in commands


def test_macos_packager_timestamps_developer_id_signatures() -> None:
    """Use secure timestamps for artifacts submitted to notarization."""

    module = load_script("package_macos_dmg")

    assert module.signing_options("-") == ["--options", "runtime"]
    assert module.signing_options("Developer ID Application: XTalk") == [
        "--options",
        "runtime",
        "--timestamp",
    ]


def test_macos_packager_notarizes_and_assesses_the_dmg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require signing, notarization, stapling, and Gatekeeper assessment."""

    module = load_script("package_macos_dmg")
    output = tmp_path / "XTalk.dmg"
    output.write_bytes(b"dmg")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", commands.append)

    module.sign_and_notarize_dmg(
        output,
        "Developer ID Application: XTalk (TEAMID)",
        "xtalk-release",
    )

    assert any("notarytool" in command for command in commands)
    assert any(command[1:3] == ["stapler", "staple"] for command in commands)
    assert any(command[0] == "spctl" for command in commands)


def test_macos_packager_signs_local_dmg_ad_hoc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give local CI disk images a verifiable ad-hoc signature."""

    module = load_script("package_macos_dmg")
    output = tmp_path / "XTalk.dmg"
    output.write_bytes(b"dmg")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", commands.append)

    module.sign_dmg(output, "-")

    assert commands == [
        ["codesign", "--force", "--sign", "-", str(output)],
        ["codesign", "--verify", "--strict", str(output)],
    ]


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


def test_macos_packager_smoke_tests_mtd_without_external_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch the MTD sidecar exactly as a signed App will launch it."""

    module = load_script("package_macos_dmg")
    app = tmp_path / "XTalk.app"
    runtime = app / "Contents" / "MacOS" / "mtd-model-runtime"
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

    module.smoke_test_mtd_runtime(app)

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
    assert module.target_supports_mtd_metal("aarch64-apple-darwin")
    assert not module.target_supports_mtd_metal("x86_64-apple-darwin")


def test_sherpa_macos_sidecar_embeds_packaged_runtime_rpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let dyld find the managed ONNX Runtime before sidecar startup."""

    module = load_script("prepare_managed_runtime")
    executable = tmp_path / "sherpa-onnx-offline-websocket-server"
    executable.write_bytes(b"mach-o")
    commands: list[list[str]] = []

    def record_command(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(module.subprocess, "run", record_command)

    module.add_macos_managed_runtime_rpath(
        executable,
        "aarch64-apple-darwin",
    )

    assert commands == [
        ["otool", "-l", str(executable)],
        [
            "install_name_tool",
            "-add_rpath",
            "@executable_path/../Resources/managed-runtime/ort",
            str(executable),
        ],
    ]


def test_sherpa_macos_sidecar_reuses_existing_packaged_runtime_rpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep repeated native runtime staging idempotent on macOS."""

    module = load_script("prepare_managed_runtime")
    executable = tmp_path / "sherpa-onnx-keyword-spotter-microphone"
    executable.write_bytes(b"mach-o")
    commands: list[list[str]] = []

    def record_command(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="path @executable_path/../Resources/managed-runtime/ort",
        )

    monkeypatch.setattr(module.subprocess, "run", record_command)

    module.add_macos_managed_runtime_rpath(
        executable,
        "aarch64-apple-darwin",
    )

    assert commands == [["otool", "-l", str(executable)]]


def test_sherpa_non_macos_sidecar_skips_macos_rpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not invoke macOS tooling while staging another platform."""

    module = load_script("prepare_managed_runtime")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )

    module.add_macos_managed_runtime_rpath(
        tmp_path / "sherpa-onnx-offline-websocket-server",
        "x86_64-unknown-linux-gnu",
    )

    assert calls == []


def test_windows_local_runtime_uses_one_static_crt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Align native tokenizer libraries on the Windows static CRT."""

    module = load_script("prepare_managed_runtime")
    calls: list[dict[str, object]] = []

    def record_command(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> None:
        del command, cwd, check
        calls.append(env)

    monkeypatch.setattr(module.subprocess, "run", record_command)
    monkeypatch.setenv("RUSTFLAGS", "-C debuginfo=1")

    module.build_local_runtime("x86_64-pc-windows-msvc", True)

    assert calls[0]["RUSTFLAGS"] == (
        "-C debuginfo=1 -C target-feature=+crt-static"
    )


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
        "en.phone",
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
        "../resources/managed-runtime/ort/onnxruntime.dll": "onnxruntime.dll"
    }


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


def test_mtd_macos_build_uses_pinned_source_and_metal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile the Apple Silicon MTD sidecar from the assembled source."""

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

    module.build_mtd_runtime("aarch64-apple-darwin", False, tmp_path)

    assert len(commands) == 1
    assert "--release" in commands[0][0]
    assert commands[0][1]["MOSS_TRANSCRIBE_CPP_DIR"] == str(tmp_path)
    assert commands[0][1]["MOSS_TRANSCRIBE_METAL"] == "1"


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
        (APP_ROOT / "examples" / "local_models_moss_tts.json").read_text(
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


def test_qwen3_asr_example_selects_only_the_int8_managed_model() -> None:
    """Keep the first Qwen3-ASR example limited to the 0.6B INT8 model."""

    example = json.loads(
        (
            APP_ROOT
            / "examples"
            / "local_models_qwen3_asr_0_6b_int8.json"
        ).read_text(encoding="utf-8")
    )

    assert example["asr"] == {
        "type": "SherpaOnnxASR",
        "params": {
            "base_url": "managed://qwen3-asr-0.6b-int8",
            "mode": "offline",
        },
    }


def test_mtd_local_models_example_uses_managed_diarization() -> None:
    """Keep the managed MTD example aligned with the official client protocol."""

    example = json.loads(
        (APP_ROOT / "examples" / "local_models_mtd.json").read_text(
            encoding="utf-8"
        )
    )

    assert example["speaker_diarization"] == {
        "type": "MossTranscribeDiarize",
        "params": {
            "base_url": "managed://moss-transcribe-diarize",
            "request_timeout_s": 30.0,
            "temperature": 0.0,
            "max_tokens": 2048,
        },
    }


def test_campplus_example_uses_only_base_url_with_sherpa_speech() -> None:
    """Keep CAM++ client parameters minimal and speech on sherpa-onnx."""

    example = json.loads(
        (APP_ROOT / "examples" / "local_models_campplus.json").read_text(
            encoding="utf-8"
        )
    )

    assert example["asr"]["type"] == "SherpaOnnxASR"
    assert example["speaker_diarization"] == {
        "type": "CampPlusDiarization",
        "params": {"base_url": "managed://campplus"},
    }
    assert example["tts"]["type"] == "SherpaOnnxTTS"
