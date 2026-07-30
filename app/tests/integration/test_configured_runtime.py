"""Integration checks driven by the repository sample model configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.config import StartupConfig, build_effective_config
from backend.xtalk_adapter import build_xtalk_runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_CONFIG = REPOSITORY_ROOT / "server_configs" / "sample.json"
VAD_MODEL = (
    REPOSITORY_ROOT
    / "app"
    / "resources"
    / "models"
    / "audio"
    / "silero_vad.onnx"
)


def build_launch_config(tmp_path: Path) -> StartupConfig:
    """Create a launch configuration that reads the required sample config.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary application data directory.

    Returns
    -------
    backend.config.StartupConfig
        Valid launch configuration.
    """

    overlay_text = os.environ.get("XTALK_TEST_CONFIG_OVERLAY", "{}")
    overlay = json.loads(overlay_text)
    if not isinstance(overlay, dict):
        raise ValueError("XTALK_TEST_CONFIG_OVERLAY must contain a JSON object")
    return StartupConfig.from_mapping(
        {
            "protocol_version": 1,
            "token": "integration-token-with-at-least-32-bytes",
            "config_path": str(SAMPLE_CONFIG),
            "data_dir": str(tmp_path),
            "origins": ["tauri://localhost"],
            "config_fallbacks": {
                "vad": {
                    "type": "SileroVAD",
                    "params": {"model_path": str(VAD_MODEL)},
                },
            },
            "config_overlay": overlay,
        }
    )


def test_effective_config_uses_repository_sample(tmp_path: Path) -> None:
    """Load all base model slots from ``server_configs/sample.json``."""

    effective = build_effective_config(build_launch_config(tmp_path))
    sample = json.loads(SAMPLE_CONFIG.read_text(encoding="utf-8"))
    for slot in ("asr", "llm_agent", "tts"):
        assert effective[slot] == sample[slot]
    assert effective["vad"] == {
        "type": "SileroVAD",
        "params": {"model_path": str(VAD_MODEL.resolve())},
    }
    assert effective["service_config"]["data_dir"] == str(tmp_path.resolve())


def test_provider_free_runtime_builds_without_an_agent() -> None:
    """Build the setup-state runtime before any providers are configured."""

    runtime = build_xtalk_runtime(
        {"service_config": {"enable_persistence": False}}
    )

    assert runtime is not None


@pytest.mark.model
def test_configured_runtime_builds_from_sample(tmp_path: Path) -> None:
    """Build the real public XTalk runtime when model tests are enabled."""

    if os.environ.get("XTALK_RUN_MODEL_TESTS") != "1":
        pytest.skip("set XTALK_RUN_MODEL_TESTS=1 to instantiate configured models")
    effective = build_effective_config(build_launch_config(tmp_path))
    runtime = build_xtalk_runtime(effective)
    assert runtime is not None
