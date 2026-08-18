"""Tests for desktop XTalk runtime composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend.desktop_service import DesktopXtalk


def test_desktop_runtime_builds_standard_default_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Reuse the generic XTalk service pipeline instead of desktop overrides."""

    recorded: dict[str, Any] = {}

    class FakeDefaultService:
        """Record construction arguments like the standard service."""

        def __init__(
            self,
            *,
            models: Any,
            service_config: dict[str, Any],
        ) -> None:
            """Store the service construction inputs."""

            recorded["models"] = models
            recorded["service_config"] = service_config
            self.models = models
            self.service_config = service_config

    monkeypatch.setattr(
        DesktopXtalk,
        "create_models_from_config",
        classmethod(lambda cls, **kwargs: object()),
    )
    monkeypatch.setattr(
        "backend.desktop_service.DesktopService",
        FakeDefaultService,
    )

    config = {
        "service_config": {
            "data_dir": str(tmp_path),
            "enable_persistence": False,
        }
    }
    runtime = DesktopXtalk._build_from_config_dict(config)

    assert isinstance(runtime, DesktopXtalk)
    assert recorded["service_config"] == config["service_config"]


def test_desktop_runtime_honors_max_connections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Forward the optional session limit into the built runtime."""

    class FakeDefaultService:
        """Record construction arguments like the standard service."""

        def __init__(
            self,
            *,
            models: Any,
            service_config: dict[str, Any],
        ) -> None:
            """Store the service construction inputs."""

            self.models = models
            self.service_config = service_config

    monkeypatch.setattr(
        DesktopXtalk,
        "create_models_from_config",
        classmethod(lambda cls, **kwargs: object()),
    )
    monkeypatch.setattr(
        "backend.desktop_service.DesktopService",
        FakeDefaultService,
    )

    runtime = DesktopXtalk._build_from_config_dict(
        {
            "max_connections": 3,
            "service_config": {
                "data_dir": str(tmp_path),
                "enable_persistence": False,
            },
        }
    )

    assert runtime._session_limiter is not None


def test_desktop_login_uses_stable_anonymous_identity(tmp_path) -> None:
    """Restore sessions under one stable desktop identity across restarts."""

    service_prototype = SimpleNamespace(
        service_config={
            "data_dir": str(tmp_path),
            "enable_persistence": True,
        },
        models=SimpleNamespace(),
    )

    first = DesktopXtalk(service_prototype=service_prototype)
    first._anonymous_user_id = "xtalk-desktop-user"
    assert first._login()["user"] == {"id": "xtalk-desktop-user"}

    created = first._persistence.create_session("xtalk-desktop-user")
    first._persistence.append_message(
        user_id="xtalk-desktop-user",
        session_id=created["session_id"],
        role="user",
        content="persist me",
    )

    restarted = DesktopXtalk(service_prototype=service_prototype)
    restarted._anonymous_user_id = "xtalk-desktop-user"
    assert restarted._login()["user"] == {"id": "xtalk-desktop-user"}
    assert restarted._list_sessions("xtalk-desktop-user") == [
        {"session_id": created["session_id"], "title": "persist me"}
    ]


def test_desktop_login_without_identity_uses_fresh_user(tmp_path) -> None:
    """Fall back to a per-login identity when none is bound."""

    service_prototype = SimpleNamespace(
        service_config={
            "data_dir": str(tmp_path),
            "enable_persistence": True,
        },
        models=SimpleNamespace(),
    )
    runtime = DesktopXtalk(service_prototype=service_prototype)

    first_user = runtime._login()["user"]["id"]
    second_user = runtime._login()["user"]["id"]

    assert first_user
    assert first_user != second_user
