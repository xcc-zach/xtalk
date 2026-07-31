"""Load developer-provided Python tools installed in application data."""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.tools import BaseTool
from xtalk.models.agents.tools import AsyncTool, SyncTool

from .tool_ui import ToolUIBinding, ToolUIBroker, wrap_tools_with_ui


TOOL_MANIFEST_FILE = "xtalk_tool.json"
TOOL_REGISTRY_FILE = "registry.json"


@dataclass(frozen=True)
class InstalledTool:
    """One enabled tool directory recorded by the desktop shell.

    Parameters
    ----------
    identifier : str
        App-generated directory identifier.
    display_name : str | dict[str, str]
        Human-readable localized name from ``xtalk_tool.json``.
    entrypoint : str
        Python ``module:factory`` entrypoint.
    directory : pathlib.Path
        Copied tool directory under application data.
    ui_entrypoint : str | None
        Optional self-contained HTML entrypoint relative to ``directory``.
    ui_update_every_s : float
        Live status polling interval or ``-1`` when polling is disabled.
    """

    identifier: str
    display_name: str | dict[str, str]
    entrypoint: str
    directory: Path
    ui_entrypoint: str | None
    ui_update_every_s: float


def load_enabled_tools(
    tools_root: Path,
    *,
    tool_ui_broker: ToolUIBroker | None = None,
) -> list[Any]:
    """Instantiate all enabled tools copied into application data.

    A broken developer tool is omitted so that it cannot prevent the local
    XTalk service from starting. The loader reports only the display name and
    exception type to stderr.

    Parameters
    ----------
    tools_root : pathlib.Path
        Application-owned directory containing ``registry.json`` and copied
        tool directories.
    tool_ui_broker : ToolUIBroker | None, optional
        App-only observer used to wrap native asynchronous tools that declare
        a UI entrypoint.

    Returns
    -------
    list[Any]
        Tool classes, instances, or factories accepted by
        ``XtalkBuilder.add_agent_tools``.
    """

    installed_tools = _read_enabled_tool_definitions(tools_root)
    loaded_tools: list[Any] = []
    for installed_tool in installed_tools:
        try:
            loaded_tools.extend(
                _load_tool_factory(
                    installed_tool,
                    tool_ui_broker=tool_ui_broker,
                )
            )
        except Exception as exc:
            print(
                "developer tool "
                f"{_display_name_for_log(installed_tool.display_name)!r} "
                "failed to load "
                f"({type(exc).__name__})",
                file=sys.stderr,
            )
    return loaded_tools


def _read_enabled_tool_definitions(tools_root: Path) -> list[InstalledTool]:
    registry_path = tools_root / TOOL_REGISTRY_FILE
    if not registry_path.is_file():
        return []

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or set(registry) != {"tools"}:
        raise ValueError("tool registry root must contain only `tools`")
    entries = registry["tools"]
    if not isinstance(entries, list):
        raise ValueError("tool registry `tools` must be a list")

    definitions: list[InstalledTool] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "enabled"}:
            raise ValueError("tool registry entries require `id` and `enabled`")
        identifier = entry["id"]
        enabled = entry["enabled"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("tool registry `id` must be a non-empty string")
        if not isinstance(enabled, bool):
            raise ValueError("tool registry `enabled` must be a boolean")
        if not enabled:
            continue

        directory = tools_root / identifier
        try:
            manifest = _read_manifest(directory / TOOL_MANIFEST_FILE)
        except Exception as exc:
            print(
                f"developer tool {identifier!r} failed to validate "
                f"({type(exc).__name__})",
                file=sys.stderr,
            )
            continue
        definitions.append(
            InstalledTool(
                identifier=identifier,
                display_name=manifest["display_name"],
                entrypoint=manifest["entrypoint"],
                directory=directory,
                ui_entrypoint=(
                    manifest["ui"]["entrypoint"]
                    if manifest["ui"] is not None
                    else None
                ),
                ui_update_every_s=(
                    manifest["ui"]["update_every_s"]
                    if manifest["ui"] is not None
                    else 1.0
                ),
            )
        )
    return definitions


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) not in (
        {"display_name", "entrypoint"},
        {"display_name", "entrypoint", "ui"},
    ):
        raise ValueError(
            "tool manifest requires `display_name`, `entrypoint`, and optional `ui`"
        )
    display_name = _normalize_display_name(manifest["display_name"])
    entrypoint = manifest["entrypoint"]
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("tool entrypoint must be a non-empty string")
    ui = _normalize_ui_config(path.parent, manifest.get("ui"))
    return {
        "display_name": display_name,
        "entrypoint": entrypoint.strip(),
        "ui": ui,
    }


def _load_tool_factory(
    installed_tool: InstalledTool,
    *,
    tool_ui_broker: ToolUIBroker | None,
) -> list[Any]:
    module_name, separator, factory_name = installed_tool.entrypoint.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("tool entrypoint must use `module:factory`")

    package_name = (
        "_xtalk_desktop_tool_" + installed_tool.identifier.replace("-", "_")
    )
    _remove_tool_modules(package_name)
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(installed_tool.directory)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        package_name,
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[package_name] = package
    try:
        with _temporary_import_path(installed_tool.directory):
            module = importlib.import_module(f"{package_name}.{module_name}")
    except Exception:
        _remove_tool_modules(package_name)
        raise

    try:
        factory = _resolve_factory(module, factory_name)
        tools = factory()
        if not isinstance(tools, list):
            raise TypeError("tool entrypoint factory must return a list")
        if not all(_is_supported_tool(tool) for tool in tools):
            raise TypeError("tool entrypoint returned an unsupported tool value")
        if installed_tool.ui_entrypoint is not None and tool_ui_broker is not None:
            tools = wrap_tools_with_ui(
                tools,
                binding=ToolUIBinding(
                    tool_id=installed_tool.identifier,
                    update_every_s=installed_tool.ui_update_every_s,
                ),
                broker=tool_ui_broker,
            )
        return tools
    except Exception:
        _remove_tool_modules(package_name)
        raise


def _resolve_factory(module: ModuleType, factory_name: str) -> Any:
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"tool factory {factory_name!r} is not callable")
    return factory


def _is_supported_tool(value: Any) -> bool:
    if isinstance(value, BaseTool):
        return True
    if isinstance(value, type) and issubclass(value, (SyncTool, AsyncTool)):
        return True
    return callable(value)


def _normalize_display_name(value: Any) -> str | dict[str, str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "tool display_name must be a string or language dictionary"
        )
    normalized: dict[str, str] = {}
    for language, display_name in value.items():
        if (
            not isinstance(language, str)
            or not language.strip()
            or not isinstance(display_name, str)
            or not display_name.strip()
        ):
            raise ValueError(
                "tool display_name language dictionary contains an invalid entry"
            )
        language = language.strip().lower()
        if language in normalized:
            raise ValueError("tool display_name contains a duplicate language")
        normalized[language] = display_name.strip()
    return normalized


def _normalize_ui_config(
    tool_directory: Path,
    value: Any,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {
        "entrypoint",
        "update_every_s",
    }:
        raise ValueError(
            "tool ui must contain `entrypoint` and optional `update_every_s`"
        )
    entrypoint = value.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("tool ui.entrypoint must be a non-empty string")
    entrypoint = entrypoint.strip()
    relative = Path(entrypoint)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".html"
        or not (tool_directory / relative).is_file()
    ):
        raise ValueError("tool ui.entrypoint must name a safe HTML file")
    update_every_s = value.get("update_every_s", 1.0)
    if isinstance(update_every_s, bool) or not isinstance(
        update_every_s,
        (int, float),
    ):
        raise ValueError("tool ui.update_every_s must be numeric")
    update_every_s = float(update_every_s)
    if update_every_s != -1.0 and not 0.1 <= update_every_s <= 3600.0:
        raise ValueError(
            "tool ui.update_every_s must be -1 or between 0.1 and 3600"
        )
    return {
        "entrypoint": entrypoint,
        "update_every_s": update_every_s,
    }


def _display_name_for_log(value: str | dict[str, str]) -> str:
    if isinstance(value, str):
        return value
    return (
        value.get("en")
        or value.get("zh")
        or next(iter(sorted(value.values())), "Developer tool")
    )


def _remove_tool_modules(package_name: str) -> None:
    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


@contextmanager
def _temporary_import_path(directory: Path):
    path = str(directory)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass
