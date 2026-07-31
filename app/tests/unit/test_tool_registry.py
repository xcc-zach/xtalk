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
