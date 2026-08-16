The model *quick-start* repositories are convenience adapters under development. If you encounter any issue, open an issue directly in the X-Talk repository and use the *original repository* to start the service.

`[Managed]` in a title indicates that the X-Talk desktop app can automatically download and enable the model. Each entry also lists its available inference backends and the automatic default selection rule.

### Speech Recognition

**Name in config**: `asr`

<details markdown="1">
<summary>SherpaOnnx [Recommended] [Managed]</summary>

**Dependency:** `pip install "xtalk[agentic-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Implementation path:** [`src/xtalk/models/asr/sherpa_onnx_asr.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/sherpa_onnx_asr.py)

A high-performance speech recognition framework, and more. It can run many speech recognition models.

**[Managed] Models and inference backends:**

- `sensevoice-small`: supports `cpu`, `cuda`, and `mlx`. By default, it uses CUDA when available, otherwise MLX on supported Apple Silicon devices, and CPU in all other cases.
- `qwen3-asr-0.6b-int8`: supports `cpu`, `cuda`, and `coreml`. By default, it uses CUDA when available, otherwise Core ML when available, and CPU in all other cases.

[Quick Start](https://github.com/xcc-zach/xtalk-sherpa-onnx-asr)

[Original Repository](https://github.com/k2-fsa/sherpa-onnx)

</details>

<details markdown="1">
<summary>Qwen3ASRFlashRealtime</summary>

**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/asr/qwen3_asr_flash_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/qwen3_asr_flash_realtime.py)

[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)

</details>

<details markdown="1">
<summary>AgenticASR [Managed]</summary>

**Dependency:** `pip install "xtalk[agentic-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/asr/agentic_asr.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/agentic_asr.py)

A speech recognition wrapper that combines the sherpa-onnx WebSocket ASR with
AgenticASR's K=3 sliding-window refinement. In `streaming` mode it uses the
online sherpa-onnx WebSocket server; in `offline` mode it simulates streaming
through `MockStreamRecognizer` over the offline WebSocket server. The Refiner
is an OpenAI-compatible chat-completions service. X-Talk Desktop can manage both
services locally with `managed://sensevoice-small` and
`managed://agentic-asr-refiner`.

**[Managed] Inference backends:** supports `cpu`, `cuda`, and `mlx`. By default, it uses CUDA when available, otherwise MLX on supported Apple Silicon devices, and CPU in all other cases.

**Config params:** `asr_base_url` (sherpa-onnx WebSocket server),
`refiner_base_url` (OpenAI-compatible Refiner service), `asr_mode`
(`"streaming"` or `"offline"`, default `"offline"`).

```json
{
  "type": "AgenticASR",
  "params": {
    "asr_base_url": "ws://127.0.0.1:6006",
    "refiner_base_url": "http://127.0.0.1:8000/v1",
    "asr_mode": "offline"
  }
}
```

Managed desktop configuration:

```json
{
  "type": "AgenticASR",
  "params": {
    "asr_base_url": "managed://sensevoice-small",
    "refiner_base_url": "managed://agentic-asr-refiner",
    "asr_mode": "offline"
  }
}
```

[Original Repository](https://github.com/AnXMuy/AgenticASR)

</details>

### Text to Speech

**Name in config**: `tts`

<details markdown="1">
<summary>IndexTTS 1.5&2 [Recommended]</summary>

**Dependency:** `pip install "xtalk[index-tts] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:**
- [`src/xtalk/models/tts/index_tts.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/index_tts.py)

[Quick Start](https://github.com/xcc-zach/xtalk-index-tts)

[Model Original Repository](https://github.com/index-tts/index-tts)

[Inference Acceleration Original Repository](https://github.com/Ksuriuri/index-tts-vllm)

</details>

<details markdown="1">
<summary>MossTTSNano [Managed]</summary>

**Dependency:** None

**Path:** [`src/xtalk/models/tts/moss_tts_nano.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/moss_tts_nano.py)

An HTTP client for the official Python/FastAPI service and XTalk's native Rust
ONNX and Swift MLX services. All use multipart `POST /api/generate` with `text`
and `prompt_audio`, return base64 PCM16 WAV, and keep client output fixed at
48 kHz mono PCM16. `voices` uses the same `{name, path}` entries as IndexTTS.

**[Managed] Inference backends:** supports `cpu`, `cuda`, and `mlx`. By default, it uses CUDA when available, otherwise MLX on supported Apple Silicon devices, and CPU in all other cases.

```json
{
  "type": "MossTTSNano",
  "params": {
    "base_url": "http://127.0.0.1:18083",
    "voices": [{"name": "zh", "path": "/path/to/reference.wav"}]
  }
}
```

[Original Repository](https://github.com/OpenMOSS/MOSS-TTS-Nano)

</details>

<details markdown="1">
<summary>MossTTSRealtime</summary>

**Dependency:** None

**Path:** [`src/xtalk/models/tts/moss_tts_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/moss_tts_realtime.py)

A full-duplex streaming TTS client that sends incremental text and receives
PCM audio from a MOSS-TTS-Realtime service.

[Quick Start](https://github.com/xcc-zach/xtalk-moss-tts-realtime)

[Original Repository](https://github.com/OpenMOSS/MOSS-TTS)

</details>

<details markdown="1">
<summary>CosyVoice</summary>

**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/tts/cosyvoice.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/cosyvoice.py)

[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)

</details>

<details markdown="1">
<summary>SherpaOnnxTTS [Managed]</summary>

**Dependency:** None

**Path:** [`src/xtalk/models/tts/sherpa_onnx_tts.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/sherpa_onnx_tts.py)

An HTTP client for the local sherpa-onnx Matcha Chinese-English TTS service. It posts text to `/v1/audio/speech` and returns 48 kHz mono PCM16 audio.

**[Managed] Inference backends:** supports `cpu` and `cuda`. By default, it uses CUDA when available and CPU otherwise.

**Config params:** only `base_url`.

[Original Repository](https://github.com/k2-fsa/sherpa-onnx)

</details>

### Forced Alignment

The forced aligner maps confirmed TTS playback time back to character or word
positions in the assistant response, so the frontend can track which text has
actually been spoken. Its audio input is fixed to 48 kHz, mono, signed 16-bit
PCM; callers do not pass a sample rate.

**Name in config**: `forced_aligner`

<details markdown="1">
<summary>Qwen3ForcedAligner</summary>

**Dependency:** None

**Path:** [`src/xtalk/models/forced_aligner/qwen3_forced_aligner.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/forced_aligner/qwen3_forced_aligner.py)

Quick Start:
```bash
python -m pip install -U "vllm[audio]"
vllm serve \
    Qwen/Qwen3-ForcedAligner-0.6B \
    --runner pooling \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --trust-request-chat-template \
    --hf-overrides \
    '{"architectures":["Qwen3ASRForcedAlignerForTokenClassification"]}' \
    --host 0.0.0.0 \
    --port 8001
```

[Original Model Repository](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

</details>

### Voice Activity Detection

**Name in config**: `vad`

X-Talk already provides VAD on the client side, so you may not need to deploy an additional one.

<details markdown="1">
<summary>Silero VAD</summary>

**Dependency:** `pip install "xtalk[silero-vad] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/vad/silero_vad.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/silero_vad.py)

[Quick Start](https://github.com/xcc-zach/xtalk-silero-vad)

[Original Repository](https://github.com/snakers4/silero-vad)

[VAD-Web](https://github.com/ricky0123/vad)

</details>

### Turn Detection

**Name in config**: `turn_detector`

Turn detector is used to determine whether the user has finished speaking and decide when the system should start generating a response.

<details markdown="1">
<summary>TurnSense</summary>

**Dependency:** `pip install "xtalk[turn-sense] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/turn_detector/turn_sense.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/turn_sense.py)

[Quick Start](https://github.com/xcc-zach/xtalk-TurnSense)

[Original Repository](https://github.com/Bairong-Xdynamics/TurnSense)

</details>

### Speaker Diarization

**Name in config**: `speaker_diarization`

<details markdown="1">
<summary>CampPlusDiarization [Managed]</summary>

**Dependency:** None

**Path:** [`src/xtalk/models/speaker_diarization/campplus.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_diarization/campplus.py)

Extracts speaker embeddings through a CAM++ service, then clusters them and assigns speaker IDs within the current session.

**[Managed] Inference backends:** supports `cpu`, `cuda`, and `coreml`. By default, it uses CUDA when available, otherwise Core ML when available, and CPU in all other cases.

```json
{
  "type": "CampPlusDiarization",
  "params": {
    "base_url": "managed://campplus"
  }
}
```

</details>

<details markdown="1">
<summary>MossTranscribeDiarize [Managed]</summary>

**Dependency:** None

**Path:** [`src/xtalk/models/speaker_diarization/moss_transcribe_diarize.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_diarization/moss_transcribe_diarize.py)

Calls a MOSS Transcribe Diarize service and returns timestamped segments containing speaker IDs and transcribed text.

**[Managed] Inference backends:** supports `cpu` and `metal`. By default, it uses Metal when available and CPU otherwise.

```json
{
  "type": "MossTranscribeDiarize",
  "params": {
    "base_url": "managed://moss-transcribe-diarize",
    "request_timeout_s": 30.0,
    "temperature": 0.0,
    "max_tokens": 2048
  }
}
```

</details>

### Speech Enhancement

**Name in config**: `speech_enhancer`

<details markdown="1">
<summary>FastEnhancer</summary>

**Dependency:** `pip install "xtalk[fast-enhancer] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/speech_enhancer/fast_enhancer.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/fast_enhancer.py)

[Quick Start](https://github.com/xcc-zach/xtalk-fastenhancer)

[Original Repository](https://github.com/aask1357/fastenhancer)

</details>

<details markdown="1">
<summary>PyWebRTCAudio</summary>

**Dependency:** `pip install "xtalk[pywebrtc-audio] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/speech_enhancer/pywebrtc_audio.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/pywebrtc_audio.py)

**Config params:** only `base_url`.

[Quick Start](https://github.com/xcc-zach/xtalk-pywebrtc-audio)

[Original Repository](https://github.com/strands-labs/pywebrtc-audio)

</details>
