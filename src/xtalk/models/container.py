from __future__ import annotations

from typing import Any, TypeVar


T = TypeVar("T")


def _maybe_clone(obj: Any) -> Any:
    """Clone an object when it exposes ``clone()``, otherwise share it."""
    if obj is None:
        return None
    clone_fn = getattr(obj, "clone", None)
    if callable(clone_fn):
        return clone_fn()
    return obj


class Models:
    """Store model instances keyed by their model interface type."""

    def __init__(self, entries: dict[type[Any], Any] | None = None) -> None:
        """Initialize a model container.

        Parameters
        ----------
        entries : dict[type[Any], Any] | None, optional
            Initial interface-to-instance mapping.
        """

        self._entries: dict[type[Any], Any] = dict(entries or {})

    def get(self, interface: type[T]) -> T | None:
        """Return the model registered for an interface, if present.

        Parameters
        ----------
        interface : type[T]
            Model interface type used as the lookup key.

        Returns
        -------
        T | None
            Registered model instance, or ``None`` when absent.
        """

        value = self._entries.get(interface)
        return value

    def require(self, interface: type[T]) -> T:
        """Return a required model or raise a clear configuration error.

        Parameters
        ----------
        interface : type[T]
            Model interface type used as the lookup key.

        Returns
        -------
        T
            Registered model instance.

        Raises
        ------
        RuntimeError
            Raised when no model is configured for ``interface``.
        """

        value = self.get(interface)
        if value is None:
            raise RuntimeError(
                f"Required model is not configured: {interface.__name__}"
            )
        return value

    def set(self, interface: type[T], model: T | None) -> None:
        """Set or remove a model for an interface.

        Parameters
        ----------
        interface : type[T]
            Model interface type used as the lookup key.
        model : T | None
            Model instance to store. Passing ``None`` removes the mapping.
        """

        if model is None:
            self._entries.pop(interface, None)
            return
        self._entries[interface] = model

    def clone(self) -> "Models":
        """Clone every cloneable model and share the rest."""

        return Models(
            {
                interface: _maybe_clone(model)
                for interface, model in self._entries.items()
            }
        )


__all__ = ["Models"]
