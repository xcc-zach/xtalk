"""Load built-in and user-provided Python tools through one manifest protocol."""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from langchain_core.tools import BaseTool
from xtalk.models.agents.tools import AsyncTool, SyncTool

from .tool_ui import ToolUIBinding, ToolUIBroker, wrap_tools_with_ui


TOOL_MANIFEST_FILE = "xtalk_tool.json"
TOOL_REGISTRY_FILE = "registry.json"
BUILTIN_TOOL_CATALOG_FILE = "builtin_tools.json"
TOOL_PREFERENCES_FILE = "tool_preferences.json"
BUILTIN_ID_PREFIX = "builtin:"


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
    origin : str
        App-owned source, either ``"builtin"`` or ``"user"``.
    data_directory : pathlib.Path
        Writable AppData directory reserved for this tool bundle.
    """

    identifier: str
    display_name: str | dict[str, str]
    entrypoint: str
    directory: Path
    ui_entrypoint: str | None
    ui_update_every_s: float
    origin: Literal["builtin", "user"]
    data_directory: Path


def load_enabled_tools(
    tools_root: Path,
    *,
    builtin_tools_root: Path | None = None,
    tool_ui_broker: ToolUIBroker | None = None,
) -> list[Any]:
    """Instantiate all enabled built-in and user-installed tools.

    A broken tool is omitted so that it cannot prevent the local
    XTalk service from starting. The loader reports only the display name and
    exception type to stderr. User-installed exported names take precedence
    over built-in tools with the same name.

    Parameters
    ----------
    tools_root : pathlib.Path
        Application-owned directory containing ``registry.json`` and copied
        tool directories.
    builtin_tools_root : pathlib.Path | None, optional
        Read-only App resource containing ``builtin_tools.json`` and bundled
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

    user_tools = _read_enabled_tool_definitions(tools_root)
    builtin_tools = (
        _read_enabled_builtin_tool_definitions(
            builtin_tools_root,
            tools_root.parent / TOOL_PREFERENCES_FILE,
        )
        if builtin_tools_root is not None
        else []
    )
    loaded_tools: list[Any] = []
    user_exported_names: set[str] = set()
    for installed_tool in [*user_tools, *builtin_tools]:
        try:
            exports = _load_tool_factory(
                installed_tool,
                tool_ui_broker=tool_ui_broker,
            )
            if installed_tool.origin == "builtin":
                exports = [
                    tool
                    for tool in exports
                    if (
                        _tool_exported_name(tool) is None
                        or _tool_exported_name(tool) not in user_exported_names
                    )
                ]
            else:
                user_exported_names.update(
                    name
                    for tool in exports
                    if (name := _tool_exported_name(tool)) is not None
                )
            loaded_tools.extend(exports)
        except Exception as exc:
            print(
                f"{installed_tool.origin} tool "
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
    identifiers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "enabled"}:
            raise ValueError("tool registry entries require `id` and `enabled`")
        identifier = entry["id"]
        enabled = entry["enabled"]
        if not _is_safe_name(identifier) or identifier in identifiers:
            raise ValueError(
                "tool registry `id` must be a safe unique identifier"
            )
        identifiers.add(identifier)
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
                origin="user",
                data_directory=tools_root.parent / "tool-data" / identifier,
            )
        )
    return definitions


def _read_enabled_builtin_tool_definitions(
    builtin_tools_root: Path,
    preferences_path: Path,
) -> list[InstalledTool]:
    """Read enabled bundled tools using the same manifest as user tools.

    Parameters
    ----------
    builtin_tools_root : pathlib.Path
        Read-only directory containing the built-in catalog and tool folders.
    preferences_path : pathlib.Path
        Writable AppData preference file containing enabled-state overrides.

    Returns
    -------
    list[InstalledTool]
        Validated enabled built-in definitions.
    """

    catalog = json.loads(
        (builtin_tools_root / BUILTIN_TOOL_CATALOG_FILE).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(catalog, dict)
        or set(catalog) != {"version", "tools"}
        or catalog["version"] != 1
        or not isinstance(catalog["tools"], list)
    ):
        raise ValueError("built-in tool catalog is invalid")
    preferences = _read_builtin_preferences(preferences_path)
    definitions: list[InstalledTool] = []
    identifiers: set[str] = set()
    paths: set[str] = set()
    for entry in catalog["tools"]:
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "path",
            "enabled_by_default",
        }:
            raise ValueError("built-in tool catalog entry is invalid")
        identifier = entry["id"]
        relative_path = entry["path"]
        enabled_by_default = entry["enabled_by_default"]
        if (
            not _is_safe_name(identifier)
            or not _is_safe_name(relative_path)
            or identifier in identifiers
            or relative_path in paths
            or not isinstance(enabled_by_default, bool)
        ):
            raise ValueError("built-in tool catalog entry is invalid")
        identifiers.add(identifier)
        paths.add(relative_path)
        preference = preferences.get(identifier)
        enabled = (
            preference["enabled"]
            if preference is not None
            else enabled_by_default
        )
        if not enabled:
            continue
        directory = builtin_tools_root / relative_path
        try:
            manifest = _read_manifest(directory / TOOL_MANIFEST_FILE)
        except Exception as exc:
            print(
                f"builtin tool {identifier!r} failed to validate "
                f"({type(exc).__name__})",
                file=sys.stderr,
            )
            continue
        definitions.append(
            InstalledTool(
                identifier=f"{BUILTIN_ID_PREFIX}{identifier}",
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
                origin="builtin",
                data_directory=(
                    preferences_path.parent / "tool-data" / identifier
                ),
            )
        )
    return definitions


def _read_builtin_preferences(
    preferences_path: Path,
) -> dict[str, dict[str, bool]]:
    """Read built-in enabled-state overrides from AppData.

    Parameters
    ----------
    preferences_path : pathlib.Path
        Preference file written by the native shell.

    Returns
    -------
    dict[str, dict[str, bool]]
        Validated preferences keyed by built-in catalog identifier.
    """

    if not preferences_path.is_file():
        return {}
    preferences = json.loads(preferences_path.read_text(encoding="utf-8"))
    if (
        not isinstance(preferences, dict)
        or set(preferences) != {"version", "builtin"}
        or preferences["version"] != 1
        or not isinstance(preferences["builtin"], dict)
    ):
        raise ValueError("tool preferences are invalid")
    for identifier, preference in preferences["builtin"].items():
        if (
            not _is_safe_name(identifier)
            or not isinstance(preference, dict)
            or set(preference) != {"enabled"}
            or not isinstance(preference["enabled"], bool)
        ):
            raise ValueError("built-in tool preference is invalid")
    return preferences["builtin"]


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

    package_name = "_xtalk_desktop_tool_" + installed_tool.identifier.replace(
        "-",
        "_",
    ).replace(":", "_")
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
    installed_tool.data_directory.mkdir(parents=True, exist_ok=True)
    try:
        with (
            _temporary_import_path(installed_tool.directory),
            _temporary_tool_data_directory(installed_tool.data_directory),
        ):
            module = importlib.import_module(
                f"{package_name}.{module_name}"
            )
    except Exception:
        _remove_tool_modules(package_name)
        raise

    try:
        factory = _resolve_factory(module, factory_name)
        with _temporary_tool_data_directory(installed_tool.data_directory):
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


@contextmanager
def _temporary_tool_data_directory(directory: Path):
    """Expose one tool's writable AppData directory during factory creation.

    Parameters
    ----------
    directory : pathlib.Path
        App-owned directory that the factory may capture for persistent data.
    """

    variable = "XTALK_TOOL_DATA_DIR"
    previous = os.environ.get(variable)
    os.environ[variable] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def _is_supported_tool(value: Any) -> bool:
    if isinstance(value, BaseTool):
        return True
    if isinstance(value, type) and issubclass(value, (SyncTool, AsyncTool)):
        return True
    return callable(value)


def _tool_exported_name(value: Any) -> str | None:
    """Return one stable exported tool name when the implementation has one.

    Parameters
    ----------
    value : Any
        Tool class, instance, or callable returned by a manifest factory.

    Returns
    -------
    str | None
        Non-empty XTalk tool name, or ``None`` for anonymous callables.
    """

    name = value.name if isinstance(value, BaseTool) else getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    if callable(value):
        callable_name = getattr(value, "__name__", None)
        if isinstance(callable_name, str) and callable_name:
            return callable_name
    return None


def _is_safe_name(value: Any) -> bool:
    """Return whether a catalog value is one non-traversing path component.

    Parameters
    ----------
    value : Any
        Candidate identifier or relative directory name.

    Returns
    -------
    bool
        ``True`` when the value cannot escape its catalog root.
    """

    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and ":" not in value
    )


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
