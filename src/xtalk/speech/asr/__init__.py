__all__ = []

# Dummy ASR for testing
try:
    from .dummy import DummyASR as DummyASR

    __all__.append("DummyASR")
except:
    pass

# Paraformer Local (funasr)
try:
    from .paraformer_local import ParaformerLocal as ParaformerLocal

    __all__.append("ParaformerLocal")
except:
    pass

# Zipformer Local (sherpa-onnx)
try:
    from .zipformer_local import ZipformerLocal as ZipformerLocal

    __all__.append("ZipformerLocal")
except:
    pass

# SherpaOnnx ASR (remote websocket)
try:
    from .sherpa_onnx_asr import SherpaOnnxASR as SherpaOnnxASR

    __all__.append("SherpaOnnxASR")
except:
    pass

# ElevenLabs ASR
try:
    from .elevenlabs import ElevenLabsASR as ElevenLabsASR

    __all__.append("ElevenLabsASR")
except:
    pass

# Qwen3 Flash ASR
try:
    from .qwen3_asr_flash_realtime import Qwen3ASRFlashRealtime as Qwen3ASRFlashRealtime

    __all__.append("Qwen3ASRFlashRealtime")
except Exception as e:
    print(f"Failed to import Qwen3ASRFlashRealtime: {e}")

# SenseVoiceSmall Local (funasr, non-streaming)
try:
    from .sensevoice_small_local import SenseVoiceSmallLocal as SenseVoiceSmallLocal

    __all__.append("SenseVoiceSmallLocal")
except:
    pass

# EasyTurn (non-streaming)
try:
    from .easy_turn.easy_turn import EasyTurnASR as EasyTurnASR

    __all__.append("EasyTurnASR")
except:
    pass
