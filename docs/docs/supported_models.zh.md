模型的*快速开始*仓库为开发中的便捷适配；如遇任何问题可直接在X-Talk仓库提Issue，并采用*原始仓库*启动服务。

### 语音识别

**配置中的名称**：`asr`

<details markdown="1">
<summary>SherpaOnnx [推荐]</summary>

**依赖：** `pip install "xtalk[sherpa-onnx-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**实现路径：** [`src/xtalk/models/asr/sherpa_onnx_asr.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/sherpa_onnx_asr.py)

一个高性能的语音识别框架，并且不仅限于此。可运行多种语音识别模型。

[快速开始](https://github.com/xcc-zach/xtalk-sherpa-onnx-asr)

[原始仓库](https://github.com/k2-fsa/sherpa-onnx)

</details>

<details markdown="1">
<summary>Qwen3ASRFlashRealtime</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/asr/qwen3_asr_flash_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/qwen3_asr_flash_realtime.py)

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)

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
<summary>CosyVoice</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/tts/cosyvoice.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/cosyvoice.py)

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)

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
<summary>XTurnix</summary>

**依赖：** XTurnix 使用 X-Talk 已包含的 `aiohttp` 依赖。

**路径：** [`src/xtalk/models/turn_detector/xturnix.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/xturnix.py)

XTurnix 消费累计 ASR 文本，以及已确认播放给用户的累计 AI 回复文本。它连接
无鉴权的本地 vLLM 服务，并始终请求服务模型名 `xturnix`。

使用 2048 token 上下文启动模型服务：

```bash
vllm serve /path/to/xturnix-Qwen3-0.6B \
  --served-model-name xturnix \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 2048 \
  --generation-config vllm
```

配置 X-Talk：

```json
{
  "turn_detector": {
    "type": "XTurnix",
    "params": {
      "base_url": "http://127.0.0.1:8000",
      "timeout": 2.0,
      "max_model_len": 2048
    }
  }
}
```

vLLM 服务必须以 `xturnix` 名称暴露模型。适配器没有 `model` 或 `api_key`
参数，也不会发送 `Authorization` header。

</details>

<details markdown="1">
<summary>TurnSense</summary>

**依赖：** `pip install "xtalk[turn-sense] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/turn_detector/turn_sense.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/turn_sense.py)

[快速开始](https://github.com/xcc-zach/xtalk-TurnSense)

[原始仓库](https://github.com/Bairong-Xdynamics/TurnSense)

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
