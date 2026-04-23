### 语音识别

**Slot**：`asr`

推荐使用 [SherpaOnnx](https://github.com/k2-fsa/sherpa-onnx)，因为它支持的模型范围广，推理性能也经过优化。

<details>
<summary>SherpaOnnx</summary>

**依赖：** `pip install "xtalk[sherpa-onnx-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/asr/sherpa_onnx_asr.py`

一个高性能的语音识别框架，并且不仅限于此。

[仓库](https://github.com/k2-fsa/sherpa-onnx)

[模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)

[启动语音识别服务的教程](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)

</details>

<details>
<summary>Qwen3ASRFlashRealtime</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/asr/qwen3_asr_flash_realtime.py`

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)

</details>

<details>
<summary>Zipformer</summary>

**依赖：** `pip install "xtalk[zipformer-local] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/asr/zipformer_local.py`

[详情](https://www.modelscope.cn/models/yhdai666/xtalk_zipformer_onnx/summary)

</details>

<details>
<summary>ElevenLabs</summary>

**依赖：** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/asr/elevenlabs.py`

[API 参考](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)

</details>

### 文本转语音

**Slot**：`tts`

<details>
<summary>IndexTTS</summary>

**依赖：** `pip install "xtalk[index-tts] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：**
- `src/xtalk/speech/tts/index_tts.py`
- `src/xtalk/speech/tts/index_tts2.py`

[仓库](https://github.com/index-tts/index-tts)

[安装方式（vllm boost）](https://github.com/Ksuriuri/index-tts-vllm)


</details>

<details>
<summary>GPT-SoVITS</summary>

> 实验性支持。如遇问题，欢迎提交 issue。

**依赖：** `pip install "xtalk[gpt-sovits] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/tts/gpt_sovits.py`


[仓库](https://github.com/RVC-Boss/GPT-SoVITS)



</details>

<details>
<summary>CosyVoice</summary>

**依赖：** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/tts/cosyvoice.py`

[详情](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)

</details>

<details>
<summary>ElevenLabs</summary>

**依赖：** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/tts/elevenlabs.py`

[API 参考](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)

</details>

### 语音活动检测

**Slot**：`vad`

X-Talk 已经在客户端侧提供了 VAD，因此您可能不一定需要额外部署一个。

<details>
<summary>Silero VAD</summary>

**依赖：** `pip install "xtalk[silero-vad] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/vad/silero_vad.py`

[模型详情](https://github.com/snakers4/silero-vad)
[VAD-Web](https://github.com/ricky0123/vad)

</details>

### 语音增强

**Slot**：`speech_enhancer`

<details>
<summary>FastEnhancer</summary>

**依赖：** `pip install onnxruntime`
**路径：** `src/xtalk/speech/speech_enhancer/speech_enhancer.py`

[模型详情](https://github.com/aask1357/fastenhancer)

</details>

### 说话人识别

**Slot**：`speaker_encoder`

<details>
<summary>Wespeaker-Voxceleb-Resnet34-LM</summary>

**依赖：** `pip install "xtalk[pyannote] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**路径：** `src/xtalk/speech/speaker_encoder/pyannote_embedding.py`

[Wespeaker](https://github.com/wenet-e2e/wespeaker)
[模型详情](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

</details>

### 字幕生成器

**Slot**：`captioner`

Captioner 用于生成音频片段的文字描述。

<details>
<summary>Qwen3-Omni-30B-A3B-Captioner</summary>

**依赖：** 无
**路径：** `src/xtalk/speech/captioner/qwen3_omni_captioner.py`

[HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)
[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Captioner)

</details>
