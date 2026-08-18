# XTalk MLX model runtime

This Apple Silicon macOS sidecar loads pinned local SenseVoice Small,
MOSS-TTS-Nano, and AgenticASR Refiner safetensor snapshots through
`mlx-audio-swift` and `mlx-swift-lm`.

It intentionally implements the protocols already consumed by XTalk:

- SenseVoice: sherpa-onnx's offline WebSocket packet
  (`sample_rate`, float-byte length, float32 PCM), followed by a text frame.
- MOSS-TTS-Nano: multipart `POST /api/generate` with `text` and
  `prompt_audio`, returning JSON with a base64 48 kHz mono PCM16 WAV.
- AgenticASR Refiner: `GET /v1/models` and OpenAI-compatible
  `POST /v1/chat/completions`, using greedy no-thinking generation.
- Process startup: one JSON line containing `status=ready`,
  `protocol_version=1`, and the loopback port.

Use Xcode for package builds because MLX requires the generated
`mlx-swift_Cmlx.bundle/default.metallib`. The desktop preparation script builds
the executable and stages that bundle into the application resources.

```sh
xcodebuild -downloadComponent MetalToolchain
swift test
xcodebuild build \
  -scheme XTalkMLXRuntime \
  -destination 'platform=macOS,arch=arm64' \
  CODE_SIGNING_ALLOWED=NO
```
