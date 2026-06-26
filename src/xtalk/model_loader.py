"""Shared model loading helpers for configuration-driven model construction."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any


# ImportSpec can be a module path string or "module:attr_chain".
ImportSpec = str

# Slot -> ordered import specs. Built-in slots are populated from
# ``@model_type`` decorators during model-type discovery.
_MODEL_REGISTRY: dict[str, list[ImportSpec]] = {}


def ensure_model_types_registered(
    registry: dict[str, list[ImportSpec]] | None = None,
) -> None:
    """Synchronize all discoverable built-in model types into a search registry.

    Parameters
    ----------
    registry : dict[str, list[ImportSpec]] | None, optional
        Registry to mutate. When omitted, the shared internal registry is used.
    """
    _scan_model_type_interfaces()

    from .models.registry import iter_model_search_specs

    target_registry = registry if registry is not None else _MODEL_REGISTRY
    for slot, spec in iter_model_search_specs():
        paths = target_registry.get(slot)
        if paths is None:
            target_registry[slot] = [spec]
        elif spec not in paths:
            paths.append(spec)


def _registered_model_slots() -> tuple[str, ...]:
    """Return model slots known to the shared internal registry."""
    ensure_model_types_registered()
    return tuple(_MODEL_REGISTRY)


def ensure_model_type_registered(
    slot: str,
    registry: dict[str, list[ImportSpec]] | None = None,
) -> None:
    """Discover and synchronize one model type by folder name or alias.

    Parameters
    ----------
    slot : str
        Model type folder name such as ``"tts"`` or retained alias such as
        ``"llm_agent"``.
    registry : dict[str, list[ImportSpec]] | None, optional
        Registry to mutate. When omitted, the shared internal registry is used.
    """
    from .models.registry import get_model_type_info

    info = get_model_type_info(slot)
    if info is None:
        _import_model_type_interfaces_by_folder(slot)
        info = get_model_type_info(slot)

    if info is None:
        _scan_model_type_interfaces(target_slot=slot)
        info = get_model_type_info(slot)

    if info is None:
        return

    target_registry = registry if registry is not None else _MODEL_REGISTRY
    for config_key in info.config_keys:
        paths = target_registry.get(config_key)
        if paths is None:
            target_registry[config_key] = [info.search_spec]
        elif info.search_spec not in paths:
            paths.append(info.search_spec)


def _import_model_type_interfaces_by_folder(folder_name: str) -> None:
    """Import the interfaces module for a model type folder if it exists."""
    if not folder_name.isidentifier():
        return
    try:
        importlib.import_module(f"xtalk.models.{folder_name}.interfaces")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            f"xtalk.models.{folder_name}",
            f"xtalk.models.{folder_name}.interfaces",
        }:
            raise
    except ImportError:
        raise


def _scan_model_type_interfaces(target_slot: str | None = None) -> None:
    """Import model type interface modules, optionally stopping at a slot match."""
    from .models.registry import get_model_type_info

    package = importlib.import_module("xtalk.models")
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    prefix = f"{package.__name__}."
    for module_info in pkgutil.iter_modules(package_paths, prefix):
        if not module_info.ispkg:
            continue
        try:
            importlib.import_module(f"{module_info.name}.interfaces")
        except ModuleNotFoundError as exc:
            if exc.name != f"{module_info.name}.interfaces":
                raise
            continue
        if target_slot is not None and get_model_type_info(target_slot) is not None:
            return


def normalize_import_spec(spec: ImportSpec) -> str:
    """Normalize an import spec into a string representation.

    Parameters
    ----------
    spec : ImportSpec
        Import spec represented as a module path string.

    Returns
    -------
    str
        Normalized module path.
    """

    if isinstance(spec, str):
        return spec
    raise TypeError(f"Invalid ImportSpec type: {type(spec)}")


def resolve_attr_chain(obj: Any, chain: str) -> Any:
    """Resolve an attribute chain such as ``"a.b.c"`` on an object.

    Parameters
    ----------
    obj : Any
        Root module or object.
    chain : str
        Dot-delimited attribute chain.

    Returns
    -------
    Any
        Resolved attribute value.
    """

    current = obj
    for part in chain.split("."):
        current = getattr(current, part)
    return current


def import_candidate(spec: ImportSpec) -> Any:
    """Import a candidate module or object from an import spec.

    Parameters
    ----------
    spec : ImportSpec
        Candidate module path or ``module:attr`` reference.

    Returns
    -------
    Any
        Imported module or resolved object.
    """

    spec_str = normalize_import_spec(spec)

    if ":" in spec_str:
        base, attr_chain = spec_str.split(":", 1)
        base_obj = import_candidate(base)
        return resolve_attr_chain(base_obj, attr_chain)

    return importlib.import_module(spec_str)


def init_model(model_config: dict | str, import_specs: list[ImportSpec]) -> Any:
    """Instantiate a model from configuration and import specs.

    Parameters
    ----------
    model_config : dict | str
        Model config containing ``type`` and optional ``params``, or a shorthand
        type string.
    import_specs : list[ImportSpec]
        Ordered import locations that may export the requested model class.

    Returns
    -------
    Any
        Instantiated model object, or ``None`` when ``model_config`` is empty.
    """

    if not model_config:
        return None

    if isinstance(model_config, dict) and "type" not in model_config:
        raise ValueError("Model config must contain 'type' field.")

    model_type = model_config["type"] if isinstance(model_config, dict) else model_config
    model_params = model_config.get("params", {}) if isinstance(model_config, dict) else {}

    errors: list[str] = []
    for spec in import_specs:
        try:
            container = import_candidate(spec)
        except Exception as exc:
            errors.append(f"{spec!r} import failed: {exc!r}")
            continue

        model_class = getattr(container, model_type, None)
        if model_class is None:
            continue
        if not isinstance(model_class, type):
            errors.append(
                f"{spec!r} has attribute {model_type!r} but it is not a class"
            )
            continue

        return model_class(**model_params)

    detail = "\n  - " + "\n  - ".join(errors) if errors else ""
    raise ValueError(
        f"Model class {model_type!r} not found. Tried specs: {import_specs}.{detail}"
    )


def _model_type_and_params(model_config: dict | str) -> tuple[str, dict[str, Any]]:
    """Return the configured model type and constructor parameters."""
    if isinstance(model_config, dict) and "type" not in model_config:
        raise ValueError("Model config must contain 'type' field.")
    model_type = model_config["type"] if isinstance(model_config, dict) else model_config
    model_params = model_config.get("params", {}) if isinstance(model_config, dict) else {}
    return str(model_type), model_params


def _resolve_registry_slot(
    slot: str,
    registry: dict[str, list[ImportSpec]],
) -> str | None:
    """Resolve a canonical slot or alias to a key present in a registry."""
    if slot in registry:
        return slot

    from .models.registry import get_model_type_info

    info = get_model_type_info(slot)
    if info is None:
        return None
    for slot_name in info.config_keys:
        if slot_name in registry:
            return slot_name
    return None


def _is_module_suffix(value: str) -> bool:
    """Return whether a config type can be used as a module suffix."""
    return all(part.isidentifier() for part in value.split("."))


def _should_skip_model_scan_module(module_name: str) -> bool:
    """Return whether a module should be skipped during fallback scanning."""
    parts = module_name.split(".")
    leaf = parts[-1]
    if leaf in {"interfaces", "__init__", "grpc_audio_service"}:
        return True
    if leaf.endswith("_pb2") or leaf.endswith("_pb2_grpc"):
        return True
    return "grpc_pb" in parts or "tools" in parts


def _import_module_for_discovery(
    module_name: str,
    errors: list[str],
) -> None:
    """Import a module while collecting discovery errors."""
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        errors.append(f"{module_name!r} import failed: {exc!r}")


def _scan_model_package(
    *,
    package_name: str,
    slot: str,
    model_type: str,
    errors: list[str],
) -> type[Any] | None:
    """Scan one model package until the requested registered model is found."""
    from .models.registry import get_model_class

    try:
        package = importlib.import_module(package_name)
    except Exception as exc:
        errors.append(f"{package_name!r} import failed: {exc!r}")
        return None

    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return None

    prefix = f"{package.__name__}."
    for module_info in pkgutil.walk_packages(package_paths, prefix):
        module_name = module_info.name
        if _should_skip_model_scan_module(module_name):
            continue
        _import_module_for_discovery(module_name, errors)
        model_class = get_model_class(slot, model_type)
        if model_class is not None:
            return model_class
    return None


def _discover_registered_model_class(
    *,
    slot: str,
    model_type: str,
    import_specs: list[ImportSpec],
) -> tuple[type[Any] | None, list[str]]:
    """Find a registered model class by importing candidate model modules."""
    from .models.registry import get_model_class

    errors: list[str] = []
    model_class = get_model_class(slot, model_type)
    if model_class is not None:
        return model_class, errors

    for spec in import_specs:
        spec_str = normalize_import_spec(spec)
        if ":" in spec_str:
            continue

        if _is_module_suffix(model_type):
            _import_module_for_discovery(f"{spec_str}.{model_type}", errors)
            model_class = get_model_class(slot, model_type)
            if model_class is not None:
                return model_class, errors

        model_class = _scan_model_package(
            package_name=spec_str,
            slot=slot,
            model_type=model_type,
            errors=errors,
        )
        if model_class is not None:
            return model_class, errors

    return None, errors


def init_registered_model(
    *,
    slot: str,
    model_config: dict | str,
    registry: dict[str, list[ImportSpec]] | None = None,
) -> Any:
    """Instantiate a model by looking up its import specs from a registry slot.

    Parameters
    ----------
    slot : str
        Registry slot such as ``"asr"`` or ``"tts"``.
    model_config : dict | str
        Model config containing ``type`` and optional ``params``, or a shorthand
        type string.
    registry : dict[str, list[ImportSpec]] | None, optional
        Registry to query. When omitted, the shared internal registry is used.

    Returns
    -------
    Any
        Instantiated model object, or ``None`` when ``model_config`` is empty.
    """

    ensure_model_type_registered(slot, registry)

    target_registry = registry if registry is not None else _MODEL_REGISTRY
    registry_slot = _resolve_registry_slot(slot, target_registry)
    if registry_slot is None:
        raise KeyError(f"Unknown model registry slot: {slot}")

    import_specs = target_registry[registry_slot]
    if not model_config:
        return None

    model_type, model_params = _model_type_and_params(model_config)

    model_class, discovery_errors = _discover_registered_model_class(
        slot=registry_slot,
        model_type=model_type,
        import_specs=import_specs,
    )
    if model_class is not None:
        return model_class(**model_params)

    try:
        return init_model(model_config=model_config, import_specs=import_specs)
    except ValueError as exc:
        detail = "\n  - " + "\n  - ".join(discovery_errors) if discovery_errors else ""
        raise ValueError(f"{exc}{detail}") from exc


def init_configured_model(
    *,
    slot: str,
    config: dict[str, Any],
    registry: dict[str, list[ImportSpec]] | None = None,
) -> Any:
    """Instantiate a model for a pipeline slot from a config dictionary.

    Parameters
    ----------
    slot : str
        Pipeline init key or model registry slot.
    config : dict[str, Any]
        Full service configuration dictionary.
    registry : dict[str, list[ImportSpec]] | None, optional
        Registry to query. When omitted, the shared internal registry is used.
    """
    ensure_model_type_registered(slot, registry)

    from .models.registry import resolve_config_slot

    config_slot = resolve_config_slot(slot, config)
    return init_registered_model(
        slot=config_slot,
        model_config=config.get(config_slot, {}),
        registry=registry,
    )


def is_registered_model_slot(
    slot: str,
    registry: dict[str, list[ImportSpec]] | None = None,
) -> bool:
    """Return whether a slot or alias is registered as a model type."""
    ensure_model_type_registered(slot, registry)

    target_registry = registry if registry is not None else _MODEL_REGISTRY
    if _resolve_registry_slot(slot, target_registry) is not None:
        return True

    from .models.registry import is_model_slot

    return is_model_slot(slot)


__all__ = [
    "ensure_model_type_registered",
    "ensure_model_types_registered",
    "init_configured_model",
    "init_registered_model",
    "is_registered_model_slot",
]
