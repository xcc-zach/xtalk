"""Tests for optional stable identities on the built-in anonymous login route."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from examples.sample_app.host.application import ApplicationHost
from xtalk.serving import Service


def _service_prototype(service_config: dict[str, Any]) -> Service:
    return cast(
        Service,
        SimpleNamespace(
            service_config=service_config,
            models=SimpleNamespace(),
        ),
    )


def test_stable_anonymous_identity_restores_sessions_after_restart(tmp_path) -> None:
    service_config = {
        "anonymous_user_id": " desktop-user ",
        "data_dir": str(tmp_path),
        "enable_persistence": True,
    }
    first_runtime = ApplicationHost(None, str(tmp_path), user_id=" desktop-user ")
    first_login = first_runtime._login()
    assert first_login["user"] == {"id": "desktop-user"}
    assert first_runtime._persistence is not None
    created = first_runtime._persistence.create_session("desktop-user")
    first_runtime._persistence.append_message(
        user_id="desktop-user",
        session_id=created["session_id"],
        role="user",
        content="persist me",
    )

    restarted_runtime = ApplicationHost(None, str(tmp_path), user_id=" desktop-user ")
    restarted_login = restarted_runtime._login()

    assert restarted_login["user"] == {"id": "desktop-user"}
    assert restarted_runtime._list_sessions("desktop-user") == [
        {
            "session_id": created["session_id"],
            "title": "persist me",
        }
    ]


@pytest.mark.parametrize("value", ["", "   ", 42, False])
def test_rejects_invalid_stable_anonymous_identity(tmp_path, value: object) -> None:
    with pytest.raises(
        ValueError,
        match="user_id must be a non-empty string",
    ):
        ApplicationHost(None, str(tmp_path), user_id=value)
