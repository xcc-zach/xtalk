"""Unit tests for component-scoped macOS App installation."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]


def load_incremental_installer():
    """Load the incremental installer from its script path.

    Returns
    -------
    types.ModuleType
        Imported installer module.
    """

    path = APP_ROOT / "scripts" / "install_macos_incremental.py"
    spec = importlib.util.spec_from_file_location(
        "app_script_install_macos_incremental",
        path,
    )
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


def test_incremental_entrypoint_is_exposed_by_npm() -> None:
    """Expose one memorable command for local component installation."""

    package = (APP_ROOT / "package.json").read_text(encoding="utf-8")
    assert '"install:macos:incremental"' in package
    assert "scripts/install_macos_incremental.py" in package


def test_resource_scope_matches_declared_static_bundle_inputs() -> None:
    """Keep incremental static resources within the Tauri bundle mapping."""

    module = load_incremental_installer()
    mappings = dict(module.resource_file_mappings())

    assert mappings[Path("examples/local_models_campplus.json")] == Path(
        "examples/local_models_campplus.json"
    )
    assert mappings[Path("examples/local_models_qwen3_asr_0_6b_int8.json")] == Path(
        "examples/local_models_qwen3_asr_0_6b_int8.json"
    )
    assert mappings[Path("resources/credentials.json")] == Path(
        "credentials.json"
    )
    assert all("app-backend-runtime" not in str(path) for path in mappings)
    assert all("managed-runtime/ort" not in str(path) for path in mappings)


def test_atomic_copy_file_preserves_executable_mode(tmp_path: Path) -> None:
    """Install a new executable without exposing a partial destination."""

    module = load_incremental_installer()
    source = tmp_path / "built" / "xtalk-desktop"
    destination = tmp_path / "XTalk.app" / "Contents" / "MacOS" / "xtalk-desktop"
    source.parent.mkdir()
    source.write_bytes(b"new executable")
    source.chmod(0o755)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old executable")

    module.atomic_copy_file(source, destination)

    assert destination.read_bytes() == b"new executable"
    assert destination.stat().st_mode & 0o111


def test_replace_directory_excludes_local_python_caches(tmp_path: Path) -> None:
    """Do not copy developer caches into bundled tool resources."""

    module = load_incremental_installer()
    source = tmp_path / "tools"
    destination = tmp_path / "XTalk.app" / "Contents" / "Resources" / "tools"
    source.mkdir()
    (source / "tool.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "tool.pyc").write_bytes(b"cache")
    destination.mkdir(parents=True)
    (destination / "stale.py").write_text("stale\n", encoding="utf-8")

    module.replace_directory(source, destination)

    assert (destination / "tool.py").is_file()
    assert not (destination / "stale.py").exists()
    assert not (destination / "__pycache__").exists()
    assert not any(
        path.name.endswith("incremental-backup")
        for path in tmp_path.rglob("*")
    )


def test_newest_wheel_prefers_latest_build(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Reuse the latest immutable wheel for App-owned backend code."""

    module = load_incremental_installer()
    source_wheels = tmp_path / ".build" / "source-inputs" / "wheel"
    artifact_wheels = tmp_path / "resources" / "artifacts"
    source_wheels.mkdir(parents=True)
    artifact_wheels.mkdir(parents=True)
    older = artifact_wheels / "xtalk-1.0-py3-none-any.whl"
    newer = source_wheels / "xtalk-2.0-py3-none-any.whl"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, ns=(1, 1))
    os.utime(newer, ns=(2, 2))
    monkeypatch.setattr(module, "APP_ROOT", tmp_path)

    assert module.newest_existing_wheel() == newer


def test_backend_runtime_keeps_only_metadata_in_resources(tmp_path: Path) -> None:
    """Install one runtime copy and preserve metadata for package discovery."""

    module = load_incremental_installer()
    app = tmp_path / "XTalk.app"
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources" / "app-backend-runtime"
    frameworks = app / "Contents" / "Frameworks"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    frameworks.mkdir(parents=True)
    executable = tmp_path / "app-backend"
    executable.write_bytes(b"backend")
    executable.chmod(0o755)
    runtime = tmp_path / "runtime"
    metadata = runtime / "xtalk-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: xtalk\n", encoding="utf-8")
    (runtime / "libpython3.12.dylib").write_bytes(b"runtime")

    module.install_backend_runtime(app, executable, runtime)

    assert (macos / "app-backend").read_bytes() == b"backend"
    assert (frameworks / "libpython3.12.dylib").is_file()
    assert (frameworks / "xtalk-1.0.dist-info").is_symlink()
    assert (resources / "xtalk-1.0.dist-info" / "METADATA").is_file()
    assert not (resources / "libpython3.12.dylib").exists()
