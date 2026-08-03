"""Startup protocol and configuration composition for the desktop sidecar."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TextIO
from urllib.parse import urlsplit


PROTOCOL_VERSION = 1
MAX_STARTUP_LINE_BYTES = 64 * 1024
MIN_STARTUP_TOKEN_BYTES = 32


def normalize_origin(origin: str) -> str:
    """Normalize one explicitly allowed WebView origin.

    Parameters
    ----------
    origin : str
        Origin in ``scheme://authority`` form. The special opaque origin
        ``"null"`` is accepted only when it is explicitly configured.

    Returns
    -------
    str
        Normalized origin without a trailing slash.

    Raises
    ------
    ValueError
        Raised when the value is not an origin.
    """

    value = origin.strip()
    if value == "null":
        return value

    parsed = urlsplit(value)
    if (
        not parsed.scheme
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("origins must contain only scheme://authority values")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


@dataclass(frozen=True)
class StartupConfig:
    """Validated launch configuration received from the Tauri parent process.

    Parameters
    ----------
    protocol_version : int
        Sidecar protocol version. Version ``1`` is currently supported.
    token : str
        Per-launch secret used by the app security middleware.
    config_path : pathlib.Path
        Path to the base XTalk JSON configuration.
    data_dir : pathlib.Path
        Writable per-user directory forced into ``service_config.data_dir``.
    origins : tuple[str, ...]
        Exact WebView origins permitted to call the loopback service.
    web_search_enabled : bool
        Whether the trusted desktop parent enabled asynchronous web search.
    config_overlay : dict[str, Any]
        Generic configuration overlay recursively merged over the base JSON.
    config_fallbacks : dict[str, Any], optional
        Top-level fallback values used only when a key is absent from the base
        configuration.
    """

    protocol_version: int
    token: str = field(repr=False)
    config_path: Path
    data_dir: Path
    origins: tuple[str, ...]
    web_search_enabled: bool
    config_overlay: dict[str, Any]
    config_fallbacks: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StartupConfig":
        """Validate and construct a startup configuration.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Decoded single-line startup JSON object.

        Returns
        -------
        StartupConfig
            Normalized immutable startup configuration.

        Raises
        ------
        ValueError
            Raised when a required field is absent or invalid.
        """

        version = payload.get("protocol_version", PROTOCOL_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("protocol_version must be an integer")
        if version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version: {version}")

        token = payload.get("token")
        if (
            not isinstance(token, str)
            or len(token.encode("utf-8")) < MIN_STARTUP_TOKEN_BYTES
        ):
            raise ValueError(
                f"token must contain at least {MIN_STARTUP_TOKEN_BYTES} bytes"
            )

        config_path_value = payload.get("config_path")
        if not isinstance(config_path_value, str) or not config_path_value.strip():
            raise ValueError("config_path must be a non-empty string")

        data_dir_value = payload.get("data_dir")
        if not isinstance(data_dir_value, str) or not data_dir_value.strip():
            raise ValueError("data_dir must be a non-empty string")

        origins_value = payload.get("origins", [])
        if isinstance(origins_value, (str, bytes)) or not isinstance(
            origins_value, list
        ):
            raise ValueError("origins must be a JSON array of strings")
        normalized_origins: list[str] = []
        for origin in origins_value:
            if not isinstance(origin, str):
                raise ValueError("origins must be a JSON array of strings")
            normalized = normalize_origin(origin)
            if normalized not in normalized_origins:
                normalized_origins.append(normalized)

        web_search_enabled = payload.get("web_search_enabled", False)
        if not isinstance(web_search_enabled, bool):
            raise ValueError("web_search_enabled must be a boolean")

        overlay_value = payload.get("config_overlay", {})
        if not isinstance(overlay_value, dict):
            raise ValueError("config_overlay must be a JSON object")

        fallbacks_value = payload.get("config_fallbacks", {})
        if not isinstance(fallbacks_value, dict):
            raise ValueError("config_fallbacks must be a JSON object")

        return cls(
            protocol_version=version,
            token=token,
            config_path=Path(config_path_value).expanduser().resolve(),
            data_dir=Path(data_dir_value).expanduser().resolve(),
            origins=tuple(normalized_origins),
            web_search_enabled=web_search_enabled,
            config_overlay=copy.deepcopy(overlay_value),
            config_fallbacks=copy.deepcopy(fallbacks_value),
        )


def read_startup_config(stream: TextIO) -> StartupConfig:
    """Read exactly one newline-delimited startup JSON message.

    Parameters
    ----------
    stream : TextIO
        Parent-process stdin stream.

    Returns
    -------
    StartupConfig
        Validated startup configuration.

    Raises
    ------
    ValueError
        Raised when stdin is empty, JSON is malformed, or the payload is not an
        object with valid startup fields.
    """

    line = stream.readline(MAX_STARTUP_LINE_BYTES + 1)
    if not line:
        raise ValueError("startup configuration was not provided on stdin")
    if (
        len(line.encode("utf-8")) > MAX_STARTUP_LINE_BYTES
        or (
            len(line) == MAX_STARTUP_LINE_BYTES + 1
            and not line.endswith("\n")
        )
    ):
        raise ValueError(
            f"startup configuration exceeds {MAX_STARTUP_LINE_BYTES} bytes"
        )
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError("startup configuration is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("startup configuration must be a JSON object")
    return StartupConfig.from_mapping(payload)


def deep_merge_config(
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge a generic configuration overlay over a base mapping.

    Dictionary values are merged recursively. Every other JSON value, including
    arrays and ``null``, replaces the base value unchanged. Neither input is
    mutated and no model type or parameter is interpreted.

    Parameters
    ----------
    base : Mapping[str, Any]
        Base configuration mapping.
    overlay : Mapping[str, Any]
        Higher-priority generic overlay.

    Returns
    -------
    dict[str, Any]
        Independent deeply copied merged configuration.
    """

    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, Mapping):
            merged[key] = deep_merge_config(base_value, overlay_value)
        else:
            merged[key] = copy.deepcopy(overlay_value)
    return merged


def load_config_object(path: Path) -> dict[str, Any]:
    """Load one XTalk JSON configuration object.

    Parameters
    ----------
    path : pathlib.Path
        Configuration file path.

    Returns
    -------
    dict[str, Any]
        Decoded configuration object.

    Raises
    ------
    ValueError
        Raised when the JSON root is not an object.
    OSError
        Raised when the file cannot be read.
    json.JSONDecodeError
        Raised when the file does not contain valid JSON.
    """

    with path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("XTalk configuration must be a JSON object")
    return payload


def build_effective_config(startup: StartupConfig) -> dict[str, Any]:
    """Build the generic XTalk configuration used for one sidecar launch.

    Missing top-level base keys are first copied from ``config_fallbacks``.
    Existing base values are preserved as complete values rather than being
    recursively merged with fallbacks. The generic overlay is then recursively
    merged over that result.
    ``service_config.data_dir`` is then forcibly set to the writable launch
    directory so packaged defaults cannot write into read-only resources.

    Parameters
    ----------
    startup : StartupConfig
        Validated launch configuration.

    Returns
    -------
    dict[str, Any]
        Effective configuration suitable for ``Xtalk.configure``.

    Raises
    ------
    ValueError
        Raised when ``service_config`` is present but is not an object.
    """

    base_config = load_config_object(startup.config_path)
    config_with_fallbacks = copy.deepcopy(base_config)
    for key, fallback_value in startup.config_fallbacks.items():
        if key not in config_with_fallbacks:
            config_with_fallbacks[key] = copy.deepcopy(fallback_value)
    effective_config = deep_merge_config(
        config_with_fallbacks,
        startup.config_overlay,
    )

    service_config_value = effective_config.get("service_config", {})
    if not isinstance(service_config_value, dict):
        raise ValueError("service_config must be a JSON object")
    service_config = copy.deepcopy(service_config_value)

    startup.data_dir.mkdir(parents=True, exist_ok=True)
    service_config["data_dir"] = str(startup.data_dir)
    effective_config["service_config"] = service_config
    return effective_config
