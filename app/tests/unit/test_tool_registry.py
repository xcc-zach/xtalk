"""Unit tests for loading developer tool directories from application data."""

from __future__ import annotations

import json
from pathlib import Path

from backend.tool_registry import load_enabled_tools


def _write_tool(
    tools_root: Path,
    *,
    identifier: str = "tool-1",
    enabled: bool = True,
    manifest: dict[str, object] | None = None,
) -> None:
    """Write one representative installed tool and registry."""

    tool_directory = tools_root / identifier
    tool_directory.mkdir(parents=True)
    (tool_directory / "xtalk_tool.json").write_text(
        json.dumps(
            manifest
            or {
                "display_name": "Test timer",
                "entrypoint": "timer_tool:create_tools",
            }
        ),
        encoding="utf-8",
    )
    (tool_directory / "timer_tool.py").write_text(
        "\n".join(
            [
                "def developer_timer():",
                "    return 'done'",
                "",
                "def create_tools():",
                "    return [developer_timer]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tools_root / "registry.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "id": identifier,
                        "enabled": enabled,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_builtin_tool(
    builtin_tools_root: Path,
    *,
    enabled_by_default: bool = True,
    can_disable: bool = True,
) -> None:
    """Write one representative built-in tool and catalog."""

    tool_directory = builtin_tools_root / "timer"
    tool_directory.mkdir(parents=True)
    (tool_directory / "xtalk_tool.json").write_text(
        json.dumps(
            {
                "display_name": {"zh": "计时器", "en": "Timer"},
                "entrypoint": "timer_tool:create_tools",
            }
        ),
        encoding="utf-8",
    )
    (tool_directory / "timer_tool.py").write_text(
        "\n".join(
            [
                "class BuiltinTimer:",
                "    name = 'timer'",
                "",
                "def create_tools():",
                "    return [BuiltinTimer]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (builtin_tools_root / "builtin_tools.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tools": [
                    {
                        "id": "timer",
                        "path": "timer",
                        "enabled_by_default": enabled_by_default,
                        "can_disable": can_disable,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_enabled_tools_calls_manifest_entrypoint(tmp_path: Path) -> None:
    """Load the list returned by an enabled directory factory."""

    tools_root = tmp_path / "tools"
    _write_tool(tools_root)

    tools = load_enabled_tools(tools_root)

    assert len(tools) == 1
    assert tools[0].__name__ == "developer_timer"


def test_load_enabled_tools_skips_disabled_directory(tmp_path: Path) -> None:
    """Do not import a directory whose registry entry is disabled."""

    tools_root = tmp_path / "tools"
    _write_tool(tools_root, enabled=False)

    assert load_enabled_tools(tools_root) == []


def test_load_enabled_tools_accepts_missing_registry(tmp_path: Path) -> None:
    """Treat an application with no installed tools as an empty registry."""

    assert load_enabled_tools(tmp_path / "tools") == []


def test_load_enabled_tools_loads_builtin_catalog(tmp_path: Path) -> None:
    """Load enabled built-ins without copying them into AppData."""

    tools_root = tmp_path / "data" / "tools"
    builtin_tools_root = tmp_path / "resources" / "tools"
    _write_builtin_tool(builtin_tools_root)

    tools = load_enabled_tools(
        tools_root,
        builtin_tools_root=builtin_tools_root,
    )

    assert len(tools) == 1
    assert tools[0].name == "timer"


def test_repository_builtin_catalog_loads_without_backend_imports(
    tmp_path: Path,
) -> None:
    """Load the packaged timer directly from its self-contained resource."""

    app_root = Path(__file__).resolve().parents[2]

    tools = load_enabled_tools(
        tmp_path / "data" / "tools",
        builtin_tools_root=app_root / "resources" / "tools",
    )

    assert {tool.name for tool in tools} == {
        "get_time",
        "timer",
        "fetch_text",
        "add_text",
        "delete_text",
        "update_text",
    }
    timer = next(tool for tool in tools if tool.name == "timer")
    assert timer.__module__.startswith("_xtalk_desktop_tool_builtin_timer")


def test_load_enabled_tools_applies_builtin_disable_preference(
    tmp_path: Path,
) -> None:
    """Exclude a built-in after the native shell persists an override."""

    tools_root = tmp_path / "data" / "tools"
    builtin_tools_root = tmp_path / "resources" / "tools"
    _write_builtin_tool(builtin_tools_root)
    tools_root.parent.mkdir(parents=True)
    (tools_root.parent / "tool_preferences.json").write_text(
        json.dumps(
            {
                "version": 1,
                "builtin": {"timer": {"enabled": False}},
            }
        ),
        encoding="utf-8",
    )

    assert (
        load_enabled_tools(
            tools_root,
            builtin_tools_root=builtin_tools_root,
        )
        == []
    )


def test_load_enabled_tools_ignores_disable_for_required_builtin(
    tmp_path: Path,
) -> None:
    """Keep a required built-in enabled despite a stale preference file."""

    tools_root = tmp_path / "data" / "tools"
    builtin_tools_root = tmp_path / "resources" / "tools"
    _write_builtin_tool(builtin_tools_root, can_disable=False)
    tools_root.parent.mkdir(parents=True)
    (tools_root.parent / "tool_preferences.json").write_text(
        json.dumps(
            {
                "version": 1,
                "builtin": {"timer": {"enabled": False}},
            }
        ),
        encoding="utf-8",
    )

    tools = load_enabled_tools(
        tools_root,
        builtin_tools_root=builtin_tools_root,
    )

    assert len(tools) == 1
    assert tools[0].name == "timer"


def test_user_tool_name_overrides_builtin_export(tmp_path: Path) -> None:
    """Prefer a user implementation when it exports a built-in tool name."""

    tools_root = tmp_path / "data" / "tools"
    builtin_tools_root = tmp_path / "resources" / "tools"
    _write_tool(tools_root)
    (tools_root / "tool-1" / "timer_tool.py").write_text(
        "\n".join(
            [
                "class UserTimer:",
                "    name = 'timer'",
                "",
                "def create_tools():",
                "    return [UserTimer]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_builtin_tool(builtin_tools_root)

    tools = load_enabled_tools(
        tools_root,
        builtin_tools_root=builtin_tools_root,
    )

    assert len(tools) == 1
    assert tools[0].__name__ == "UserTimer"


def test_load_enabled_tools_supports_package_relative_imports(
    tmp_path: Path,
) -> None:
    """Keep helper modules isolated inside the installed tool namespace."""

    tools_root = tmp_path / "tools"
    _write_tool(tools_root)
    tool_directory = tools_root / "tool-1"
    (tool_directory / "helper.py").write_text(
        "\n".join(
            [
                "def developer_timer():",
                "    return 'done'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tool_directory / "timer_tool.py").write_text(
        "\n".join(
            [
                "from .helper import developer_timer",
                "",
                "def create_tools():",
                "    return [developer_timer]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    tools = load_enabled_tools(tools_root)

    assert len(tools) == 1
    assert tools[0].__name__ == "developer_timer"


def test_load_enabled_tools_omits_broken_factory(
    tmp_path: Path,
    capsys,
) -> None:
    """Keep sidecar startup usable when one developer module fails."""

    tools_root = tmp_path / "tools"
    _write_tool(tools_root)
    (tools_root / "tool-1" / "timer_tool.py").write_text(
        "raise RuntimeError('broken tool')\n",
        encoding="utf-8",
    )

    assert load_enabled_tools(tools_root) == []
    assert "Test timer" in capsys.readouterr().err


def test_load_enabled_tools_accepts_localized_ui_manifest(
    tmp_path: Path,
) -> None:
    """Accept localized names and optional UI metadata without changing tools."""

    tools_root = tmp_path / "tools"
    _write_tool(
        tools_root,
        manifest={
            "display_name": {"zh": "计时器", "en": "Timer"},
            "entrypoint": "timer_tool:create_tools",
            "ui": {
                "entrypoint": "ui/index.html",
                "update_every_s": -1,
            },
        },
    )
    ui_directory = tools_root / "tool-1" / "ui"
    ui_directory.mkdir()
    (ui_directory / "index.html").write_text(
        "<!doctype html><title>Timer</title>",
        encoding="utf-8",
    )

    tools = load_enabled_tools(tools_root)

    assert len(tools) == 1
    assert tools[0].__name__ == "developer_timer"


def test_load_enabled_tools_omits_invalid_ui_interval(
    tmp_path: Path,
    capsys,
) -> None:
    """Reject an interval outside the App's bounded polling contract."""

    tools_root = tmp_path / "tools"
    _write_tool(
        tools_root,
        manifest={
            "display_name": {"en": "Timer"},
            "entrypoint": "timer_tool:create_tools",
            "ui": {
                "entrypoint": "ui/index.html",
                "update_every_s": 0,
            },
        },
    )
    ui_directory = tools_root / "tool-1" / "ui"
    ui_directory.mkdir()
    (ui_directory / "index.html").write_text(
        "<!doctype html><title>Timer</title>",
        encoding="utf-8",
    )

    assert load_enabled_tools(tools_root) == []
    assert "tool-1" in capsys.readouterr().err
