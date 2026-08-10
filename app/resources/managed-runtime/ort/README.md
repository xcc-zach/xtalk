# Shared managed ONNX Runtime

Release preparation places one platform ONNX Runtime 1.27 dynamic library in
this directory together with the shared sherpa-onnx libraries. SenseVoice,
Matcha TTS, and the Rust MOSS TTS sidecar all resolve this same library at
runtime. Generated native libraries are intentionally not committed.
