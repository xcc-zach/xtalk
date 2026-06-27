### 语音识别

**配置中的名称**：`asr`

<details markdown="1">
<summary>SherpaOnnx [推荐]</summary>

**依赖：** `pip install "xtalk[sherpa-onnx-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/asr/sherpa_onnx_asr.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/sherpa_onnx_asr.py)

一个高性能的语音识别框架，并且不仅限于此。

[仓库](https://github.com/k2-fsa/sherpa-onnx)

[模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)

[启动语音识别服务的教程](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)

</details>

<details markdown="1">
<summary>Qwen3ASRFlashRealtime</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/asr/qwen3_asr_flash_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/qwen3_asr_flash_realtime.py)

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)

</details>

<details markdown="1">
<summary>Zipformer</summary>

**依赖：** `pip install "xtalk[zipformer-local] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/asr/zipformer_local.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/zipformer_local.py)

[详情](https://www.modelscope.cn/models/yhdai666/xtalk_zipformer_onnx/summary)

</details>

<details markdown="1">
<summary>ElevenLabs</summary>

**依赖：** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/asr/elevenlabs.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/elevenlabs.py)

[API 参考](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)

</details>

### 文本转语音

**配置中的名称**：`tts`

<details markdown="1">
<summary>IndexTTS [推荐]</summary>

**依赖：** `pip install "xtalk[index-tts] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：**
- [`src/xtalk/models/tts/index_tts.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/index_tts.py)
- [`src/xtalk/models/tts/index_tts2.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/index_tts2.py)

[仓库](https://github.com/index-tts/index-tts)

[安装方式（vllm boost）](https://github.com/Ksuriuri/index-tts-vllm)


</details>

<details markdown="1">
<summary>GPT-SoVITS</summary>

> 实验性支持。如遇问题，欢迎提交 issue。

**依赖：** `pip install "xtalk[gpt-sovits] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/tts/gpt_sovits.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/gpt_sovits.py)


[仓库](https://github.com/RVC-Boss/GPT-SoVITS)



</details>

<details markdown="1">
<summary>CosyVoice</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/tts/cosyvoice.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/cosyvoice.py)

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)

</details>

<details markdown="1">
<summary>ElevenLabs</summary>

**依赖：** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/tts/elevenlabs.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/elevenlabs.py)

[API 参考](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)

</details>

### 语音活动检测

**配置中的名称**：`vad`

X-Talk 已经在客户端侧提供了 VAD，因此您可能不一定需要额外部署一个。

<details markdown="1">
<summary>Silero VAD</summary>

**依赖：** `pip install "xtalk[silero-vad] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/vad/silero_vad.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/silero_vad.py)

[模型详情](https://github.com/snakers4/silero-vad)
[VAD-Web](https://github.com/ricky0123/vad)

</details>

### 轮次检测

**配置中的名称**：`turn_detector`

Turn detector 用于判断用户是否已经说完，并决定系统何时开始生成回复。

<details markdown="1">
<summary>SoulxDuplug</summary>

**依赖：** `pip install "xtalk[soulx-duplug] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/turn_detector/soulx_duplug.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/soulx_duplug.py)

[仓库](https://github.com/Soul-AILab/SoulX-Duplug)

</details>

<details markdown="1">
<summary>TurnSense</summary>

**依赖：** `pip install "xtalk[turn-sense] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/turn_detector/turn_sense.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/turn_sense.py)

[原仓库](https://github.com/Bairong-Xdynamics/TurnSense)

[适配 X-Talk 的服务部署参考](https://github.com/xcc-zach/TurnSense-server)

</details>

### 语音增强

**配置中的名称**：`speech_enhancer`

<details markdown="1">
<summary>FastEnhancer</summary>

**依赖：** `pip install onnxruntime`

**路径：** [`src/xtalk/models/speech_enhancer/speech_enhancer.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/speech_enhancer.py)

[模型详情](https://github.com/aask1357/fastenhancer)

</details>

### 说话人识别

**配置中的名称**：`speaker_encoder`

<details markdown="1">
<summary>Wespeaker-Voxceleb-Resnet34-LM</summary>

**依赖：** `pip install "xtalk[pyannote] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**路径：** [`src/xtalk/models/speaker_encoder/pyannote_embedding.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_encoder/pyannote_embedding.py)

[Wespeaker](https://github.com/wenet-e2e/wespeaker)
[模型详情](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

</details>

### 字幕生成器

**配置中的名称**：`captioner`

Captioner 用于生成音频片段的文字描述。

<details markdown="1">
<summary>Qwen3-Omni-30B-A3B-Captioner</summary>

**依赖：** 无

**路径：** [`src/xtalk/models/captioner/qwen3_omni_captioner.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/captioner/qwen3_omni_captioner.py)

[HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)
[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Captioner)

</details>
