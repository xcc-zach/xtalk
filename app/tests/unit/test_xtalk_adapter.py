"""Contract tests for the public XTalk adapter."""

from __future__ import annotations

import copy
from typing import Any

from backend import xtalk_adapter


class _FakeRuntime:
    """Record builder output and route mounting for adapter tests."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        """Initialize one fake built runtime."""

        self.config = config
        self.tools = tools or []
        self.mounted_app: Any = None
        self._anonymous_user_id: str | None = None

    def mount_routes(self, app: Any) -> None:
        """Record the application passed to the public mount method."""

        self.mounted_app = app


class _FakeBuilder:
    """Record public builder operations used by the desktop adapter."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize a builder for one source configuration."""

        self.config = config
        self.tools: list[Any] = []
        self.built = False

    def add_agent_tools(self, tools: list[Any]) -> "_FakeBuilder":
        """Record standard tools added to the configured Agent."""

        self.tools.extend(tools)
        return self

    def build(self) -> _FakeRuntime:
        """Return a fake runtime after recording the build call."""

        self.built = True
        return _FakeRuntime(config=self.config, tools=list(self.tools))


class _FakeXtalk:
    """Stand in for the public ``xtalk.Xtalk`` entrypoint."""

    latest_builder: _FakeBuilder | None = None

    @classmethod
    def configure(cls, config: dict[str, Any]) -> _FakeBuilder:
        """Return a recording builder for the supplied configuration."""

        cls.latest_builder = _FakeBuilder(config)
        return cls.latest_builder


def test_adapter_builds_runtime_and_tools_through_public_apis(
    monkeypatch,
) -> None:
    """Build the desktop runtime through the public builder API."""

    config = {
        "llm_agent": {
            "type": "DefaultAgent",
            "params": {"tools": ["configured-tool"]},
        },
        "service_config": {"nested": {"preserve": True}},
    }
    original_config = copy.deepcopy(config)
    _FakeXtalk.latest_builder = None
    monkeypatch.setattr(xtalk_adapter, "Xtalk", _FakeXtalk)

    runtime = xtalk_adapter.build_xtalk_runtime(config)

    assert config == original_config
    assert _FakeXtalk.latest_builder is not None
    assert _FakeXtalk.latest_builder.config is config
    assert _FakeXtalk.latest_builder.tools == []
    assert _FakeXtalk.latest_builder.built
    assert runtime.config is config
    assert runtime.tools == []


def test_adapter_keeps_provider_free_config_usable(monkeypatch) -> None:
    """Build a provider-free shell without registering an Agent tool."""

    config = {"service_config": {"enable_persistence": True}}
    _FakeXtalk.latest_builder = None
    monkeypatch.setattr(xtalk_adapter, "Xtalk", _FakeXtalk)

    runtime = xtalk_adapter.build_xtalk_runtime(config)

    assert _FakeXtalk.latest_builder is not None
    assert _FakeXtalk.latest_builder.tools == []
    assert _FakeXtalk.latest_builder.built
    assert runtime.config is config
    assert runtime.tools == []


def test_adapter_binds_desktop_identity_outside_service_config(
    monkeypatch,
) -> None:
    """Bind the private desktop identity without modifying service config."""

    config = {"service_config": {"enable_persistence": True}}
    original_config = copy.deepcopy(config)
    _FakeXtalk.latest_builder = None
    monkeypatch.setattr(xtalk_adapter, "Xtalk", _FakeXtalk)

    runtime = xtalk_adapter.build_xtalk_runtime(
        config,
        anonymous_user_id="desktop-user",
    )

    assert config == original_config
    assert "anonymous_user_id" not in config["service_config"]
    assert runtime._anonymous_user_id == "desktop-user"


def test_adapter_loads_unified_user_and_builtin_tool_roots(
    monkeypatch,
    tmp_path,
) -> None:
    """Pass both App-managed roots through the unified tool loader."""

    class DeveloperTimer:
        """Stand in for a developer-provided timer tool."""

        name = "timer"

    config = {"llm_agent": {"type": "DefaultAgent", "params": {}}}
    _FakeXtalk.latest_builder = None
    monkeypatch.setattr(xtalk_adapter, "Xtalk", _FakeXtalk)
    observed: dict[str, Any] = {}

    def _load_tools(
        tools_root,
        *,
        builtin_tools_root,
        tool_ui_broker,
    ):
        observed["tools_root"] = tools_root
        observed["builtin_tools_root"] = builtin_tools_root
        observed["tool_ui_broker"] = tool_ui_broker
        return [DeveloperTimer]

    monkeypatch.setattr(
        xtalk_adapter,
        "load_enabled_tools",
        _load_tools,
    )

    tools_root = tmp_path / "data" / "tools"
    builtin_tools_root = tmp_path / "resources" / "tools"
    runtime = xtalk_adapter.build_xtalk_runtime(
        config,
        tools_root=tools_root,
        builtin_tools_root=builtin_tools_root,
    )

    assert _FakeXtalk.latest_builder is not None
    assert _FakeXtalk.latest_builder.tools == [DeveloperTimer]
    assert runtime.tools == [DeveloperTimer]
    assert observed == {
        "tools_root": tools_root,
        "builtin_tools_root": builtin_tools_root,
        "tool_ui_broker": None,
    }


def test_adapter_mounts_routes_through_the_runtime_public_method() -> None:
    """Delegate route registration without inspecting runtime internals."""

    runtime = _FakeRuntime()
    app = object()

    xtalk_adapter.mount_xtalk_routes(runtime, app)

    assert runtime.mounted_app is app
