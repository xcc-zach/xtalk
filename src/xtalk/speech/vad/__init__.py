__all__ = []

# Silero VAD
try:
    from .silero_vad import SileroVAD as SileroVAD

    __all__.append("SileroVAD")
except:
    pass
