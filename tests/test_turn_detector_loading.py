"""Tests for configuration-driven turn-detector loading."""

from xtalk.model_loader import init_registered_model


def test_xturnix_loads_through_model_discovery() -> None:
    """Load XTurnix by discovering its registered implementation module."""
    detector = init_registered_model(
        slot="turn_detector",
        model_config={
            "type": "XTurnix",
            "params": {
                "base_url": "http://127.0.0.1:8000",
                "timeout": 2.0,
                "max_model_len": 2048,
            },
        },
    )

    from xtalk.models.turn_detector.xturnix import XTurnix

    assert isinstance(detector, XTurnix)
