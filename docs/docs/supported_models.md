The model *quick-start* repositories are convenience adapters under development. If you encounter any issue, open an issue directly in the X-Talk repository and use the *original repository* to start the service.

### Speech Recognition

**Name in config**: `asr`

<details markdown="1">
<summary>SherpaOnnx [Recommended]</summary>

**Dependency:** `pip install "xtalk[sherpa-onnx-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Implementation path:** [`src/xtalk/models/asr/sherpa_onnx_asr.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/sherpa_onnx_asr.py)

A high-performance speech recognition framework, and more. It can run many speech recognition models.

[Quick Start](https://github.com/xcc-zach/xtalk-sherpa-onnx-asr)

[Original Repository](https://github.com/k2-fsa/sherpa-onnx)

</details>

<details markdown="1">
<summary>Qwen3ASRFlashRealtime</summary>

**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/asr/qwen3_asr_flash_realtime.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/qwen3_asr_flash_realtime.py)

[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)

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
<summary>CosyVoice</summary>

**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** [`src/xtalk/models/tts/cosyvoice.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/cosyvoice.py)

[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)

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
