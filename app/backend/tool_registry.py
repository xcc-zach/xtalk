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


TOOL_MANIFEST_FILE = "xtalk_tool.json"
TOOL_REGISTRY_FILE = "registry.json"


@dataclass(frozen=True)
class InstalledTool:
    """One enabled tool directory recorded by the desktop shell.

    Parameters
    ----------
    identifier : str
        App-generated directory identifier.
    display_name : str
        Human-readable name from ``xtalk_tool.json``.
    entrypoint : str
        Python ``module:factory`` entrypoint.
    directory : pathlib.Path
        Copied tool directory under application data.
    """

    identifier: str
    display_name: str
    entrypoint: str
    directory: Path


def load_enabled_tools(tools_root: Path) -> list[Any]:
    """Instantiate all enabled tools copied into application data.

    A broken developer tool is omitted so that it cannot prevent the local
    XTalk service from starting. The loader reports only the display name and
    exception type to stderr.

    Parameters
    ----------
    tools_root : pathlib.Path
        Application-owned directory containing ``registry.json`` and copied
        tool directories.

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
            loaded_tools.extend(_load_tool_factory(installed_tool))
        except Exception as exc:
            print(
                "developer tool "
                f"{installed_tool.display_name!r} failed to load "
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
        manifest = _read_manifest(directory / TOOL_MANIFEST_FILE)
        definitions.append(
            InstalledTool(
                identifier=identifier,
                display_name=manifest["display_name"],
                entrypoint=manifest["entrypoint"],
                directory=directory,
            )
        )
    return definitions


def _read_manifest(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "display_name",
        "entrypoint",
    }:
        raise ValueError(
            "tool manifest requires only `display_name` and `entrypoint`"
        )
    display_name = manifest["display_name"]
    entrypoint = manifest["entrypoint"]
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError("tool display_name must be a non-empty string")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("tool entrypoint must be a non-empty string")
    return {
        "display_name": display_name.strip(),
        "entrypoint": entrypoint.strip(),
    }


def _load_tool_factory(installed_tool: InstalledTool) -> list[Any]:
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
