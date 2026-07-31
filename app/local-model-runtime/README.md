# XTalk Local Model Runtime

This crate is the native ONNX sidecar for optional desktop models. The first
implemented engine is MOSS-TTS-Nano. It uses ONNX Runtime and SentencePiece
directly and does not depend on Python, PyTorch, Transformers, FastAPI, or
Uvicorn.

## Run the MOSS HTTP service

`--model-root` may point either to the directory containing both model snapshot
directories or directly to `MOSS-TTS-Nano-100M-ONNX`.

```bash
cargo run --release -- \
  --ort-dylib /path/to/libonnxruntime.dylib \
  --model-root /path/to/MOSS-TTS-Nano/models \
  --port 18083 \
  --cpu-threads 2
```

The Rust binary loads ONNX Runtime from the explicit `--ort-dylib` path. The
desktop package should ship the matching platform library beside the sidecar
and pass its resolved resource path at launch; no system-wide ONNX Runtime
installation is used.

The process writes one readiness JSON object to stdout after all ONNX sessions
have loaded.

```bash
curl http://127.0.0.1:18083/health

curl -X POST \
  -F 'text=你好，这是 Rust ONNX Runtime 测试。' \
  -F 'prompt_audio=@/path/to/reference.wav' \
  http://127.0.0.1:18083/api/generate \
  --output moss-response.json
```

`POST /api/generate` matches the official Python/FastAPI service: it accepts
multipart `text` and `prompt_audio` fields and returns base64-encoded WAV in
the `audio_base64` JSON field. Uploaded reference audio is decoded and
converted to 48 kHz before ONNX codec encoding. Generated audio is always
48 kHz mono PCM16 WAV. The older built-in-voice `/v1/audio/speech` endpoint
remains available for compatibility.
