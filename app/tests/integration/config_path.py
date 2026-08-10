"""Shared resolution of user-provided model configuration paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


CONFIG_PATH_ENVIRONMENT_VARIABLE = "XTALK_TEST_CONFIG_PATH"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "server_configs" / "sample.json"
)


def resolve_test_config_path() -> Path:
    """Return the configured model configuration path.

    The repository sample configuration is not committed to git, so tests
    that need a real model configuration read ``XTALK_TEST_CONFIG_PATH`` when
    set and otherwise fall back to the legacy repository path.

    Returns
    -------
    pathlib.Path
        Resolved model configuration path.
    """

    configured = os.environ.get(CONFIG_PATH_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def require_test_config_path() -> Path:
    """Resolve a model configuration path, skipping the caller when absent.

    Returns
    -------
    pathlib.Path
        Existing model configuration path.

    Raises
    ------
    pytest.skip.Exception
        Raised when no configuration file exists at the resolved path.
    """

    config_path = resolve_test_config_path()
    if not config_path.is_file():
        pytest.skip(
            f"set {CONFIG_PATH_ENVIRONMENT_VARIABLE} to an existing model "
            "configuration JSON to run this test"
        )
    return config_path
