from __future__ import annotations

from typing import Any

__all__ = [
    "Agent",
    "AgentContext",
    "AgentOutput",
    "ASR",
    "Captioner",
    "Embeddings",
    "ForcedAligner",
    "ForcedAlignmentUnit",
    "Models",
    "PuntRestorer",
    "Rewriter",
    "SpeakerEncoder",
    "SpeechEnhancer",
    "SpeechSpeedController",
    "StreamingTextTTS",
    "TTS",
    "TurnDetector",
    "VAD",
]

_INTERFACE_IMPORTS = {
    "Agent": ("xtalk.models.agents.interfaces", "Agent"),
    "AgentContext": ("xtalk.models.agents.interfaces", "AgentContext"),
    "AgentOutput": ("xtalk.models.agents.interfaces", "AgentOutput"),
    "ASR": ("xtalk.models.asr.interfaces", "ASR"),
    "Captioner": ("xtalk.models.captioner.interfaces", "Captioner"),
    "Embeddings": ("xtalk.models.embeddings.interfaces", "Embeddings"),
    "ForcedAligner": (
        "xtalk.models.forced_aligner.interfaces",
        "ForcedAligner",
    ),
    "ForcedAlignmentUnit": (
        "xtalk.models.forced_aligner.interfaces",
        "ForcedAlignmentUnit",
    ),
    "Models": ("xtalk.models.container", "Models"),
    "PuntRestorer": ("xtalk.models.punt_restorer.interfaces", "PuntRestorer"),
    "Rewriter": ("xtalk.models.rewriters.interfaces", "Rewriter"),
    "SpeakerEncoder": ("xtalk.models.speaker_encoder.interfaces", "SpeakerEncoder"),
    "SpeechEnhancer": ("xtalk.models.speech_enhancer.interfaces", "SpeechEnhancer"),
    "SpeechSpeedController": (
        "xtalk.models.speech_speed_controller.interfaces",
        "SpeechSpeedController",
    ),
    "StreamingTextTTS": ("xtalk.models.tts.interfaces", "StreamingTextTTS"),
    "TTS": ("xtalk.models.tts.interfaces", "TTS"),
    "TurnDetector": ("xtalk.models.turn_detector.interfaces", "TurnDetector"),
    "VAD": ("xtalk.models.vad.interfaces", "VAD"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose model interfaces from the top-level models package."""
    import importlib

    try:
        module_name, attr_name = _INTERFACE_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
