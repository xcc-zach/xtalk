"""Validate the public Realtime endpoint configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RealtimeConfig:
    """Hold the model alias and optional API keys.

    Parameters
    ----------
    model : str
        Public model alias.
    api_keys : tuple[str, ...]
        Allowed keys; an empty tuple disables API key verification.
    """

    model: str = "xtalk"
    api_keys: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict) -> "RealtimeConfig":
        """Validate configuration without exposing credentials.

        Parameters
        ----------
        value : dict
            Realtime service configuration.

        Returns
        -------
        RealtimeConfig
            Validated immutable configuration.
        """
        if not isinstance(value, dict) or set(value) - {"model", "api_keys"}:
            raise ValueError("realtime_api supports only model and api_keys")
        model = value.get("model", "xtalk")
        keys = value.get("api_keys", [])
        if not isinstance(model, str) or not model.strip():
            raise ValueError("realtime_api.model must be a nonempty string")
        if not isinstance(keys, list) or any(
            not isinstance(k, str) or not k for k in keys
        ):
            raise ValueError("realtime_api.api_keys must be a list of nonempty strings")
        return cls(model=model, api_keys=tuple(keys))
