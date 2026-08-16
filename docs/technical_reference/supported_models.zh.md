模型的*快速开始*仓库为开发中的便捷适配；如遇任何问题可直接在X-Talk仓库提Issue，并采用*原始仓库*启动服务。

标题中的 `[受管]` 表示 X-Talk 桌面应用可以自动下载并启用该模型。条目中同时列出了可用的推理后端和自动选择后端时的默认规则。

### 语音识别

**配置中的名称**：`asr`

<details markdown="1">
<summary>SherpaOnnx [推荐] [受管]</summary>

**依赖：** `pip install "xtalk[agentic-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**实现路径：** [`src/xtalk/models/asr/sherpa_onnx_asr.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/sherpa_onnx_asr.py)

一个高性能的语音识别框架，并且不仅限于此。可运行多种语音识别模型。

**[受管] 模型与推理后端：**

- `sensevoice-small`：支持 `cpu`、`cuda`、`mlx`。默认优先使用可用的 CUDA；否则在受支持的 Apple Silicon 设备上使用 MLX；其他情况使用 CPU。
- `qwen3-asr-0.6b-int8`：支持 `cpu`、`cuda`、`coreml`。默认优先使用可用的 CUDA；否则使用可用的 Core ML；其他情况使用 CPU。

[快速开始](https://github.com/xcc-zach/xtalk-sherpa-onnx-asr)

[原始仓库](https://github.com/k2-fsa/sherpa-onnx)

</details>

<details markdown="1">
<summary>Qwen3ASRFlashRealtime</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/asr/qwen3_asr_flash_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/qwen3_asr_flash_realtime.py)

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)

</details>

<details markdown="1">
<summary>AgenticASR [受管]</summary>

**依赖：** `pip install "xtalk[agentic-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/asr/agentic_asr.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/agentic_asr.py)

结合 sherpa-onnx WebSocket ASR 与 AgenticASR K=3 滑窗精修的语音识别封装。
`streaming` 模式使用 sherpa-onnx 在线 WebSocket 服务；`offline` 模式通过
`MockStreamRecognizer` 包装离线 WebSocket 服务来模拟流式。Refiner 为
OpenAI 兼容的 chat-completions 服务。X-Talk Desktop 可通过
`managed://sensevoice-small` 和 `managed://agentic-asr-refiner` 自动下载并
托管两个本地服务。

**[受管] 推理后端：** 支持 `cpu`、`cuda`、`mlx`。默认优先使用可用的 CUDA；否则在受支持的 Apple Silicon 设备上使用 MLX；其他情况使用 CPU。

**配置参数：** `asr_base_url`（sherpa-onnx WebSocket 服务地址）、
`refiner_base_url`（OpenAI 兼容的 Refiner 服务地址）、`asr_mode`
（`"streaming"` 或 `"offline"`，默认 `"offline"`）。

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

Desktop 托管配置：

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

[原始仓库](https://github.com/AnXMuy/AgenticASR)

</details>

### 文本转语音

**配置中的名称**：`tts`

<details markdown="1">
<summary>IndexTTS 1.5&2[推荐]</summary>

**依赖：** `pip install "xtalk[index-tts] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：**
- [`src/xtalk/models/tts/index_tts.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/index_tts.py)

[快速开始](https://github.com/xcc-zach/xtalk-index-tts)

[模型原始仓库](https://github.com/index-tts/index-tts)

[推理加速原始仓库](https://github.com/Ksuriuri/index-tts-vllm)

</details>

<details markdown="1">
<summary>MossTTSNano [受管]</summary>

**依赖：** 无

**路径：** [`src/xtalk/models/tts/moss_tts_nano.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/moss_tts_nano.py)

同时兼容官方 Python/FastAPI 服务及 XTalk 本地 Rust ONNX、Swift MLX 服务的
HTTP 客户端。各端统一使用 multipart `POST /api/generate`，字段为 `text` 和 `prompt_audio`，
返回 base64 PCM16 WAV；客户端输出固定为 48 kHz 单声道 PCM16。`voices` 与
IndexTTS 一样使用 `{name, path}` 配置。

**[受管] 推理后端：** 支持 `cpu`、`cuda`、`mlx`。默认优先使用可用的 CUDA；否则在受支持的 Apple Silicon 设备上使用 MLX；其他情况使用 CPU。

```json
{
  "type": "MossTTSNano",
  "params": {
    "base_url": "http://127.0.0.1:18083",
    "voices": [{"name": "zh", "path": "/path/to/reference.wav"}]
  }
}
```

[原始仓库](https://github.com/OpenMOSS/MOSS-TTS-Nano)

</details>

<details markdown="1">
<summary>MossTTSRealtime</summary>

**依赖：** 无

**路径：** [`src/xtalk/models/tts/moss_tts_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/moss_tts_realtime.py)

一个全双工流式 TTS 客户端，可向 MOSS-TTS-Realtime 服务增量发送文本并接收
PCM 音频。

[快速开始](https://github.com/xcc-zach/xtalk-moss-tts-realtime)

[原始仓库](https://github.com/OpenMOSS/MOSS-TTS)

</details>

<details markdown="1">
<summary>CosyVoice</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/tts/cosyvoice.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/cosyvoice.py)

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)

</details>

<details markdown="1">
<summary>SherpaOnnxTTS [受管]</summary>

**依赖：** 无

**路径：** [`src/xtalk/models/tts/sherpa_onnx_tts.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/sherpa_onnx_tts.py)

本地 sherpa-onnx Matcha 中英 TTS 服务的 HTTP 客户端。向 `/v1/audio/speech` 提交文本，输出固定为 48 kHz 单声道 PCM16。

**[受管] 推理后端：** 支持 `cpu`、`cuda`。默认优先使用可用的 CUDA，否则使用 CPU。

**配置参数：** 仅 `base_url`。

[原始仓库](https://github.com/k2-fsa/sherpa-onnx)

</details>

### 强制对齐

用于把 TTS 已播放的音频时间映射回回复文本中的字/词位置，从而让前端更准确地跟踪“当前说到哪个字”。其音频输入固定为 48 kHz、单声道、有符号 16 位 PCM，调用方不再传入采样率。

**配置中的名称**：`forced_aligner`

<details markdown="1">
<summary>Qwen3ForcedAligner</summary>

**依赖：** 无

**路径：** [`src/xtalk/models/forced_aligner/qwen3_forced_aligner.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/forced_aligner/qwen3_forced_aligner.py)

快速开始：
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

[模型原始仓库](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

</details>

### 语音活动检测

**配置中的名称**：`vad`

X-Talk 已经在客户端侧提供了 VAD，因此您可能不一定需要额外部署一个。

<details markdown="1">
<summary>Silero VAD</summary>

**依赖：** `pip install "xtalk[silero-vad] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/vad/silero_vad.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/silero_vad.py)

[快速开始](https://github.com/xcc-zach/xtalk-silero-vad)

[原始仓库](https://github.com/snakers4/silero-vad)

[VAD-Web](https://github.com/ricky0123/vad)

</details>

### 轮次检测

**配置中的名称**：`turn_detector`

Turn detector 用于判断用户是否已经说完，并决定系统何时开始生成回复。

<details markdown="1">
<summary>TurnSense</summary>

**依赖：** `pip install "xtalk[turn-sense] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/turn_detector/turn_sense.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/turn_sense.py)

[快速开始](https://github.com/xcc-zach/xtalk-TurnSense)

[原始仓库](https://github.com/Bairong-Xdynamics/TurnSense)

</details>

### 说话人分离

**配置中的名称**：`speaker_diarization`

<details markdown="1">
<summary>CampPlusDiarization [受管]</summary>

**依赖：** 无

**路径：** [`src/xtalk/models/speaker_diarization/campplus.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_diarization/campplus.py)

通过 CAM++ 服务提取说话人嵌入，并在当前会话中聚类和分配说话人编号。

**[受管] 推理后端：** 支持 `cpu`、`cuda`、`coreml`。默认优先使用可用的 CUDA；否则使用可用的 Core ML；其他情况使用 CPU。

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
<summary>MossTranscribeDiarize [受管]</summary>

**依赖：** 无

**路径：** [`src/xtalk/models/speaker_diarization/moss_transcribe_diarize.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_diarization/moss_transcribe_diarize.py)

调用 MOSS Transcribe Diarize 服务，返回带时间戳、说话人编号和文本的分段结果。

**[受管] 推理后端：** 支持 `cpu`、`metal`。默认优先使用可用的 Metal，否则使用 CPU。

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

### 语音增强

**配置中的名称**：`speech_enhancer`

<details markdown="1">
<summary>FastEnhancer</summary>

**依赖：** `pip install "xtalk[fast-enhancer] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/speech_enhancer/fast_enhancer.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/fast_enhancer.py)

[快速开始](https://github.com/xcc-zach/xtalk-fastenhancer)

[原始仓库](https://github.com/aask1357/fastenhancer)

</details>

<details markdown="1">
<summary>PyWebRTCAudio</summary>

**依赖：** `pip install "xtalk[pywebrtc-audio] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/speech_enhancer/pywebrtc_audio.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/pywebrtc_audio.py)

**配置参数：** 仅 `base_url`。

[快速开始](https://github.com/xcc-zach/xtalk-pywebrtc-audio)

[原始仓库](https://github.com/strands-labs/pywebrtc-audio)

</details>
