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
) -> None:
    """Write one representative installed tool and registry."""

    tool_directory = tools_root / identifier
    tool_directory.mkdir(parents=True)
    (tool_directory / "xtalk_tool.json").write_text(
        json.dumps(
            {
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
