"""Contract tests for the public XTalk adapter."""

from __future__ import annotations

import copy
from typing import Any

from backend import xtalk_adapter
from backend.timer_tool import TimerTool


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


class _FakeTimeTool:
    """Stand in for the core time tool."""

    name = "get_time"


class _FakeWebSearchTool:
    """Stand in for the core asynchronous web-search tool."""

    name = "web_search"


def test_adapter_builds_runtime_and_tools_through_public_apis(
    monkeypatch,
) -> None:
    """Build the desktop runtime through public builder and tool APIs."""

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
    monkeypatch.setattr(
        xtalk_adapter,
        "build_time_tool",
        lambda: _FakeTimeTool,
    )

    runtime = xtalk_adapter.build_xtalk_runtime(config)

    assert config == original_config
    assert _FakeXtalk.latest_builder is not None
    assert _FakeXtalk.latest_builder.config is config
    assert _FakeXtalk.latest_builder.tools == [TimerTool, _FakeTimeTool]
    assert _FakeXtalk.latest_builder.built
    assert runtime.config is config
    assert runtime.tools == [TimerTool, _FakeTimeTool]


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


def test_adapter_registers_web_search_when_enabled(monkeypatch) -> None:
    """Register asynchronous web search only when the desktop enables it."""

    config = {"llm_agent": {"type": "DefaultAgent", "params": {}}}
    _FakeXtalk.latest_builder = None
    monkeypatch.setattr(xtalk_adapter, "Xtalk", _FakeXtalk)
    monkeypatch.setattr(
        xtalk_adapter,
        "build_time_tool",
        lambda: _FakeTimeTool,
    )
    monkeypatch.setattr(
        xtalk_adapter,
        "build_async_web_search_tool",
        lambda: _FakeWebSearchTool,
    )

    runtime = xtalk_adapter.build_xtalk_runtime(
        config,
        web_search_enabled=True,
    )

    assert runtime.tools == [TimerTool, _FakeTimeTool, _FakeWebSearchTool]


def test_adapter_allows_installed_tools_to_replace_bundled_tools(
    monkeypatch,
    tmp_path,
) -> None:
    """Prefer enabled developer tools over bundled tools with the same name."""

    class DeveloperTimer:
        """Stand in for a developer-provided timer tool."""

        name = "timer"

    class DeveloperTime:
        """Stand in for a developer-provided time tool."""

        name = "get_time"

    config = {"llm_agent": {"type": "DefaultAgent", "params": {}}}
    _FakeXtalk.latest_builder = None
    monkeypatch.setattr(xtalk_adapter, "Xtalk", _FakeXtalk)
    monkeypatch.setattr(
        xtalk_adapter,
        "load_enabled_tools",
        lambda tools_root: [DeveloperTimer, DeveloperTime],
    )

    runtime = xtalk_adapter.build_xtalk_runtime(
        config,
        tools_root=tmp_path / "tools",
    )

    assert _FakeXtalk.latest_builder is not None
    assert _FakeXtalk.latest_builder.tools == [DeveloperTimer, DeveloperTime]
    assert runtime.tools == [DeveloperTimer, DeveloperTime]


def test_adapter_mounts_routes_through_the_runtime_public_method() -> None:
    """Delegate route registration without inspecting runtime internals."""

    runtime = _FakeRuntime()
    app = object()

    xtalk_adapter.mount_xtalk_routes(runtime, app)

    assert runtime.mounted_app is app
