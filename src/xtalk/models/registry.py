from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

ImportSpec: TypeAlias = str


@dataclass(frozen=True)
class ModelTypeInfo:
    """Registered model interface metadata."""

    interface: type[Any]
    aliases: tuple[str, ...]
    search_spec: ImportSpec

    @property
    def config_key(self) -> str:
        """Return the folder-derived config key for this model type."""
        return _infer_config_key(self.search_spec)

    @property
    def config_keys(self) -> tuple[str, ...]:
        """Return the folder-derived config key followed by aliases."""
        return (self.config_key, *self.aliases)


@dataclass(frozen=True)
class ModelImplInfo:
    """Registered model implementation metadata."""

    interface: type[Any]
    model_class: type[Any]
    name: str
    aliases: tuple[str, ...]

    @property
    def all_names(self) -> tuple[str, ...]:
        """Return the canonical name followed by aliases."""
        return (self.name, *self.aliases)


_MODEL_TYPES_BY_SLOT: dict[str, ModelTypeInfo] = {}
_MODEL_TYPE_ORDER: list[ModelTypeInfo] = []
_MODEL_IMPLS_BY_SLOT: dict[str, dict[str, ModelImplInfo]] = {}


def _infer_config_key(search_spec: ImportSpec) -> str:
    """Infer the model type config key from an implementation package."""
    return search_spec.rsplit(".", 1)[-1]


def _infer_search_spec(interface: type[Any]) -> ImportSpec:
    """Infer the package containing implementations from an interface class."""
    module_name = interface.__module__
    if module_name.endswith(".interfaces"):
        return module_name.rsplit(".", 1)[0]
    return module_name


def _register_slot_alias(slot: str, info: ModelTypeInfo, *, replace: bool) -> None:
    """Register one slot or alias for a model type."""
    existing = _MODEL_TYPES_BY_SLOT.get(slot)
    if existing is not None and existing.interface is not info.interface and not replace:
        raise ValueError(f"model slot {slot!r} is already registered")
    _MODEL_TYPES_BY_SLOT[slot] = info
    _MODEL_IMPLS_BY_SLOT.setdefault(slot, {})


def _infer_model_interface(impl_cls: type[Any]) -> type[Any]:
    """Infer the registered model interface from an implementation class."""
    matches = [
        base
        for base in impl_cls.__mro__[1:]
        if "__model_type_key__" in base.__dict__
    ]
    if not matches:
        raise TypeError(
            f"{impl_cls.__name__} must inherit from a class registered with @model_type"
        )
    if len(matches) > 1:
        names = ", ".join(base.__name__ for base in matches)
        raise TypeError(
            f"{impl_cls.__name__} inherits multiple model interfaces: {names}"
        )
    return matches[0]


def _default_model_aliases(impl_cls: type[Any], name: str) -> tuple[str, ...]:
    """Infer convenient config aliases for a model implementation class."""
    aliases: list[str] = []
    module_name = impl_cls.__module__.rsplit(".", 1)[-1]

    for alias in (module_name,):
        if alias and alias != name and alias not in aliases:
            aliases.append(alias)
    return tuple(aliases)


def model_type(
    cls: type[Any] | None = None,
    *,
    aliases: list[str] | tuple[str, ...] | None = None,
    replace: bool = False,
) -> Callable[[type[Any]], type[Any]] | type[Any]:
    """Register a model interface as a config-loadable model type.

    Parameters
    ----------
    cls : type[Any] | None, optional
        Interface class when the decorator is used as ``@model_type``.
    aliases : list[str] | tuple[str, ...] | None, optional
        Additional config keys accepted for backwards compatibility.
    replace : bool, optional
        Whether an existing slot registration may be replaced.
    """

    def decorator(interface: type[Any]) -> type[Any]:
        resolved_aliases = tuple(aliases or ())
        if any(not alias for alias in resolved_aliases):
            raise ValueError("model type aliases must be non-empty")

        search_spec = _infer_search_spec(interface)
        config_key = _infer_config_key(search_spec)
        if not config_key:
            raise ValueError("model type config key must be non-empty")

        info = ModelTypeInfo(
            interface=interface,
            aliases=resolved_aliases,
            search_spec=search_spec,
        )

        interface.__model_type_key__ = config_key
        interface.__model_aliases__ = resolved_aliases
        interface.__model_search_spec__ = search_spec

        _MODEL_TYPE_ORDER[:] = [
            existing_info
            for existing_info in _MODEL_TYPE_ORDER
            if existing_info.interface is not interface
        ]
        _MODEL_TYPE_ORDER.append(info)
        for slot_name in info.config_keys:
            _register_slot_alias(slot_name, info, replace=replace)

        return interface

    if cls is not None:
        return decorator(cls)
    return decorator


def model(
    cls: type[Any] | None = None,
    *,
    name: str | None = None,
    aliases: list[str] | tuple[str, ...] | None = None,
    replace: bool = False,
) -> Callable[[type[Any]], type[Any]] | type[Any]:
    """Register a model implementation class for configuration loading.

    Parameters
    ----------
    cls : type[Any] | None, optional
        Implementation class when the decorator is used as ``@model``.
    name : str | None, optional
        Canonical config name. Defaults to the class name.
    aliases : list[str] | tuple[str, ...] | None, optional
        Additional accepted config names.
    replace : bool, optional
        Whether an existing model registration may be replaced.
    """

    explicit_aliases = tuple(aliases or ())

    def decorator(impl_cls: type[Any]) -> type[Any]:
        interface = _infer_model_interface(impl_cls)
        type_info = get_model_type_info(interface.__model_type_key__)
        if type_info is None:
            raise TypeError(
                f"{interface.__name__} is not registered with @model_type"
            )

        impl_name = name or impl_cls.__name__
        if not impl_name:
            raise ValueError("model implementation name must be non-empty")
        if any(not alias for alias in explicit_aliases):
            raise ValueError("model implementation aliases must be non-empty")

        generated_aliases = _default_model_aliases(impl_cls, impl_name)
        resolved_aliases = tuple(dict.fromkeys((*explicit_aliases, *generated_aliases)))
        impl_info = ModelImplInfo(
            interface=interface,
            model_class=impl_cls,
            name=impl_name,
            aliases=resolved_aliases,
        )

        for slot_name in type_info.config_keys:
            bucket = _MODEL_IMPLS_BY_SLOT.setdefault(slot_name, {})
            for impl_key in impl_info.all_names:
                existing = bucket.get(impl_key)
                if existing is None or existing.model_class is impl_cls or replace:
                    bucket[impl_key] = impl_info
                    continue
                if impl_key == impl_name or impl_key in explicit_aliases:
                    raise ValueError(
                        f"model {impl_key!r} is already registered for {slot_name!r}"
                    )
        return impl_cls

    if cls is not None:
        return decorator(cls)
    return decorator


def get_model_type_info(slot: str) -> ModelTypeInfo | None:
    """Return model type metadata for a canonical slot or alias."""
    return _MODEL_TYPES_BY_SLOT.get(slot)


def iter_model_search_specs() -> tuple[tuple[str, ImportSpec], ...]:
    """Return all registered slots and aliases with their search specs."""
    items: list[tuple[str, ImportSpec]] = []
    seen: set[str] = set()
    for info in _MODEL_TYPE_ORDER:
        for slot_name in info.config_keys:
            if slot_name in seen:
                continue
            items.append((slot_name, info.search_spec))
            seen.add(slot_name)
    return tuple(items)


def iter_model_type_infos() -> tuple[ModelTypeInfo, ...]:
    """Return registered model type metadata in discovery order."""
    return tuple(_MODEL_TYPE_ORDER)


def resolve_config_slot(slot: str, config: dict[str, Any]) -> str:
    """Resolve which config key should be used for a requested model slot."""
    if slot in config:
        return slot

    info = get_model_type_info(slot)
    if info is None:
        return slot

    for config_key in info.config_keys:
        if config_key in config:
            return config_key

    return slot


def is_model_slot(slot: str) -> bool:
    """Return whether a slot or alias is registered as a model type."""
    return slot in _MODEL_TYPES_BY_SLOT


def get_model_class(slot: str, name: str) -> type[Any] | None:
    """Return a registered model implementation class."""
    impl_info = _MODEL_IMPLS_BY_SLOT.get(slot, {}).get(name)
    if impl_info is None:
        return None
    return impl_info.model_class


__all__ = [
    "model",
    "model_type",
]
