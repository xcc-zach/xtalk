"""Unit tests for sidecar startup and generic configuration composition."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from backend.config import (
    MAX_STARTUP_LINE_BYTES,
    PROTOCOL_VERSION,
    StartupConfig,
    build_effective_config,
    deep_merge_config,
    normalize_origin,
    read_startup_config,
)


TOKEN = "t" * 32


def _startup(
    tmp_path: Path,
    *,
    config: dict[str, object],
    overlay: dict[str, object] | None = None,
    fallbacks: dict[str, object] | None = None,
) -> StartupConfig:
    """Create a launch configuration backed by a temporary JSON file."""

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return StartupConfig.from_mapping(
        {
            "protocol_version": PROTOCOL_VERSION,
            "token": TOKEN,
            "config_path": str(config_path),
            "data_dir": str(tmp_path / "data"),
            "origins": ["tauri://localhost"],
            "config_overlay": overlay or {},
            "config_fallbacks": fallbacks or {},
        }
    )


def test_read_startup_config_consumes_one_json_line(tmp_path: Path) -> None:
    """Parse the exact parent-to-sidecar launch protocol."""

    payload = {
        "protocol_version": 1,
        "token": TOKEN,
        "config_path": str(tmp_path / "base.json"),
        "data_dir": str(tmp_path / "data"),
        "builtin_tools_root": str(tmp_path / "resources" / "tools"),
        "origins": [
            "TAURI://LOCALHOST/",
            "tauri://localhost",
            "http://localhost:1420",
        ],
        "config_overlay": {"arbitrary_section": {"enabled": True}},
        "anonymous_user_id": " desktop-user ",
    }
    stream = io.StringIO(json.dumps(payload) + "\nignored second line\n")

    startup = read_startup_config(stream)

    assert startup.protocol_version == 1
    assert startup.token == TOKEN
    assert startup.origins == (
        "tauri://localhost",
        "http://localhost:1420",
    )
    assert startup.config_overlay == payload["config_overlay"]
    assert startup.builtin_tools_root == (
        tmp_path / "resources" / "tools"
    ).resolve()
    assert startup.anonymous_user_id == "desktop-user"
    assert startup.config_fallbacks == {}
    assert stream.readline() == "ignored second line\n"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"token": "", "config_path": "a", "data_dir": "b"},
        {"token": "x", "config_path": "", "data_dir": "b"},
        {"token": "x", "config_path": "a", "data_dir": ""},
        {
            "protocol_version": 2,
            "token": "x",
            "config_path": "a",
            "data_dir": "b",
        },
    ],
)
def test_read_startup_config_rejects_invalid_payloads(payload: object) -> None:
    """Reject malformed or incompatible launch messages."""

    with pytest.raises(ValueError):
        read_startup_config(io.StringIO(json.dumps(payload) + "\n"))


def test_read_startup_config_enforces_line_and_token_limits(
    tmp_path: Path,
) -> None:
    """Bound parent input and require a launch secret of useful length."""

    short_token_payload = {
        "protocol_version": 1,
        "token": "too-short",
        "config_path": str(tmp_path / "base.json"),
        "data_dir": str(tmp_path / "data"),
    }
    with pytest.raises(ValueError, match="at least 32 bytes"):
        read_startup_config(
            io.StringIO(json.dumps(short_token_payload) + "\n")
        )

    with pytest.raises(ValueError, match="exceeds"):
        read_startup_config(
            io.StringIO(" " * (MAX_STARTUP_LINE_BYTES + 1) + "\n")
        )


def test_read_startup_config_rejects_non_object_fallbacks(
    tmp_path: Path,
) -> None:
    """Reject fallback configuration that is not a JSON object."""

    payload = {
        "token": TOKEN,
        "config_path": str(tmp_path / "base.json"),
        "data_dir": str(tmp_path / "data"),
        "config_fallbacks": ["invalid"],
    }

    with pytest.raises(ValueError, match="config_fallbacks"):
        read_startup_config(io.StringIO(json.dumps(payload) + "\n"))


@pytest.mark.parametrize("anonymous_user_id", ["", "   ", 42, False])
def test_read_startup_config_rejects_invalid_anonymous_user_id(
    tmp_path: Path,
    anonymous_user_id: object,
) -> None:
    """Require a non-empty string when the app supplies a stable identity."""

    payload = {
        "token": TOKEN,
        "config_path": str(tmp_path / "base.json"),
        "data_dir": str(tmp_path / "data"),
        "anonymous_user_id": anonymous_user_id,
    }

    with pytest.raises(ValueError, match="anonymous_user_id"):
        read_startup_config(io.StringIO(json.dumps(payload) + "\n"))


def test_normalize_origin_rejects_paths_and_accepts_explicit_null() -> None:
    """Keep the Origin allow-list exact and path-free."""

    assert normalize_origin("null") == "null"
    assert normalize_origin("HTTPS://EXAMPLE.TEST/") == "https://example.test"
    with pytest.raises(ValueError):
        normalize_origin("https://example.test/not-an-origin")


def test_deep_merge_is_generic_and_does_not_mutate_inputs() -> None:
    """Deep-merge arbitrary mappings while replacing lists and scalar values."""

    base = {
        "model_slot": {
            "type": "RegisteredType",
            "params": {
                "endpoint": "https://old.example",
                "options": ["old"],
                "nested": {"left": 1},
            },
        },
        "unchanged": {"value": True},
    }
    overlay = {
        "model_slot": {
            "params": {
                "endpoint": "https://new.example",
                "options": ["new"],
                "nested": {"right": 2},
            }
        }
    }

    merged = deep_merge_config(base, overlay)

    assert merged == {
        "model_slot": {
            "type": "RegisteredType",
            "params": {
                "endpoint": "https://new.example",
                "options": ["new"],
                "nested": {"left": 1, "right": 2},
            },
        },
        "unchanged": {"value": True},
    }
    assert base["model_slot"]["params"]["endpoint"] == "https://old.example"
    assert overlay["model_slot"]["params"]["nested"] == {"right": 2}


def test_build_effective_config_forces_writable_data_dir(tmp_path: Path) -> None:
    """Apply a generic overlay and force the launch-owned persistence path."""

    startup = _startup(
        tmp_path,
        config={
            "llm_agent": {
                "type": "DefaultAgent",
                "params": {"model": {"model": "base"}},
            },
            "service_config": {
                "data_dir": "/read-only/default",
                "enable_persistence": True,
            },
        },
        overlay={
            "llm_agent": {
                "params": {
                    "model": {"base_url": "https://example.test/v1"},
                }
            },
            "service_config": {"data_dir": "/overlay/must-not-win"},
        },
    )

    effective = build_effective_config(startup)

    assert effective["llm_agent"]["params"]["model"] == {
        "model": "base",
        "base_url": "https://example.test/v1",
    }
    assert effective["service_config"]["data_dir"] == str(startup.data_dir)
    assert effective["service_config"]["enable_persistence"] is True
    assert startup.data_dir.is_dir()


def test_build_effective_config_fills_only_missing_top_level_keys(
    tmp_path: Path,
) -> None:
    """Fill absent slots without recursively mixing into explicit base values."""

    fallbacks = {
        "vad": {
            "type": "FallbackVAD",
            "params": {
                "model_path": "/fallback/vad.onnx",
                "threshold": 0.5,
            },
        },
        "speech_enhancer": {
            "type": "FallbackEnhancer",
            "params": {"model_path": "/fallback/enhancer.onnx"},
        },
    }
    startup = _startup(
        tmp_path,
        config={
            "vad": {
                "type": "ConfiguredVAD",
                "params": {"base_url": "https://vad.example.test"},
            }
        },
        fallbacks=fallbacks,
    )

    effective = build_effective_config(startup)

    assert effective["vad"] == {
        "type": "ConfiguredVAD",
        "params": {"base_url": "https://vad.example.test"},
    }
    assert effective["speech_enhancer"] == fallbacks["speech_enhancer"]

    effective["speech_enhancer"]["params"]["model_path"] = "/changed"
    assert fallbacks["speech_enhancer"]["params"]["model_path"] == (
        "/fallback/enhancer.onnx"
    )
    assert startup.config_fallbacks["speech_enhancer"]["params"]["model_path"] == (
        "/fallback/enhancer.onnx"
    )


def test_build_effective_config_applies_overlay_after_fallbacks(
    tmp_path: Path,
) -> None:
    """Keep the generic overlay at the highest configuration precedence."""

    startup = _startup(
        tmp_path,
        config={"asr": {"type": "ConfiguredASR"}},
        fallbacks={
            "vad": {
                "type": "FallbackVAD",
                "params": {
                    "model_path": "/fallback/vad.onnx",
                    "threshold": 0.5,
                },
            }
        },
        overlay={
            "vad": {
                "type": "OverlayVAD",
                "params": {"threshold": 0.8},
            }
        },
    )

    effective = build_effective_config(startup)

    assert effective["vad"] == {
        "type": "OverlayVAD",
        "params": {
            "model_path": "/fallback/vad.onnx",
            "threshold": 0.8,
        },
    }


def test_build_effective_config_loads_repository_sample(tmp_path: Path) -> None:
    """Use the repository's requested model configuration as the base fixture."""

    repository_root = Path(__file__).resolve().parents[3]
    sample_path = repository_root / "server_configs" / "sample.json"
    startup = StartupConfig.from_mapping(
        {
            "protocol_version": 1,
            "token": TOKEN,
            "config_path": str(sample_path),
            "data_dir": str(tmp_path / "data"),
            "origins": [],
            "config_overlay": {"service_config": {"enable_persistence": False}},
        }
    )

    effective = build_effective_config(startup)

    assert effective["asr"] == json.loads(
        sample_path.read_text(encoding="utf-8")
    )["asr"]
    assert effective["llm_agent"]["type"] == "DefaultAgent"
    assert effective["service_config"] == {
        "enable_persistence": False,
        "data_dir": str(startup.data_dir),
    }


def test_build_effective_config_rejects_non_object_service_config(
    tmp_path: Path,
) -> None:
    """Surface an invalid service configuration rather than silently replacing it."""

    startup = _startup(
        tmp_path,
        config={"service_config": ["invalid"]},
    )

    with pytest.raises(ValueError, match="service_config"):
        build_effective_config(startup)
