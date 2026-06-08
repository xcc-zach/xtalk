### Speech Recognition
    
**Name in config**: `asr`
    
<details markdown="1">
<summary>SherpaOnnx [Recommended]</summary>
    
**Dependency:** `pip install "xtalk[sherpa-onnx-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/asr/sherpa_onnx_asr.py`
    
A high-performance speech recognition framework and beyond.

[Repo](https://github.com/k2-fsa/sherpa-onnx)
    
[Models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
    
[Tutorial to start speech recognition server](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)
    
</details>

<details markdown="1">
<summary>Qwen3ASRFlashRealtime</summary>
    
**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/asr/qwen3_asr_flash_realtime.py`
    
[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)
    
</details>
    
<details markdown="1">
<summary>Zipformer</summary>
    
**Dependency:** `pip install "xtalk[zipformer-local] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/asr/zipformer_local.py`
    
[Details](https://www.modelscope.cn/models/yhdai666/xtalk_zipformer_onnx/summary)
    
</details>
    
<details markdown="1">
<summary>ElevenLabs</summary>
    
**Dependency:** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/asr/elevenlabs.py`
    
[API Reference](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
    
</details>
    
### Text to Speech
    
**Name in config**: `tts`
    
<details markdown="1">
<summary>IndexTTS [Recommended]</summary>

**Dependency:** `pip install "xtalk[index-tts] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** 
- `src/xtalk/speech/tts/index_tts.py`
- `src/xtalk/speech/tts/index_tts2.py`
    
[Repo](https://github.com/index-tts/index-tts)
    
[Installation (vllm boost)](https://github.com/Ksuriuri/index-tts-vllm)

    
</details>
    
<details markdown="1">
<summary>GPT-SoVITS</summary>

> Experimental. Feel free to open an issue for any problem.

**Dependency:** `pip install "xtalk[gpt-sovits] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/tts/gpt_sovits.py`

    
[Repo](https://github.com/RVC-Boss/GPT-SoVITS)
    

    
</details>
    
<details markdown="1">
<summary>CosyVoice</summary>

**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/tts/cosyvoice.py`

[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)
    
</details>
    
<details markdown="1">
<summary>ElevenLabs</summary>
    
**Dependency:** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/tts/elevenlabs.py`
    
[API Reference](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
    
</details>

### Voice Activity Detection
    
**Name in config**: `vad`
    
X-Talk has VAD on client side, so you may not need one.

<details markdown="1">
<summary>Silero VAD</summary>

**Dependency:** `pip install "xtalk[silero-vad] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/vad/silero_vad.py`
    
[Model Details](https://github.com/snakers4/silero-vad)
[VAD-Web](https://github.com/ricky0123/vad)
    
</details>

### Turn Detection

**Name in config**: `turn_detector`

Turn detectors decide when the user has finished speaking and the system should start generation.

<details markdown="1">
<summary>SoulxDuplug</summary>

**Dependency:** `pip install "xtalk[soulx-duplug] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/turn_detector/soulx_duplug.py`

[Repo](https://github.com/Soul-AILab/SoulX-Duplug)

</details>

<details markdown="1">
<summary>TurnSense</summary>

**Dependency:** `pip install "xtalk[turn-sense] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/turn_detector/turn_sense.py`

[Original Repo](https://github.com/Bairong-Xdynamics/TurnSense)

[X-Talk-adapted service deployment reference](https://github.com/xcc-zach/TurnSense-server)

</details>

### Speech Enhancement
    
**Name in config**: `speech_enhancer`

<details markdown="1">
<summary>FastEnhancer</summary>
    
**Dependency:** `pip install onnxruntime`

**Path:** `src/xtalk/speech/speech_enhancer/speech_enhancer.py`
    
[Model Details](https://github.com/aask1357/fastenhancer)
    
</details>
    
### Speaker Recognition
    
**Name in config**: `speaker_encoder`    
    
<details markdown="1">
<summary>Wespeaker-Voxceleb-Resnet34-LM</summary>
  
**Dependency:** `pip install "xtalk[pyannote] @ git+https://github.com/xcc-zach/xtalk.git@main"`

**Path:** `src/xtalk/speech/speaker_encoder/pyannote_embedding.py`
    
[Wespeaker](https://github.com/wenet-e2e/wespeaker)
[Model Details](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
    
</details>
    
### Captioner
    
**Name in config**: `captioner`
    
Captioners give you description of audio clip.
    
<details markdown="1">
<summary>Qwen3-Omni-30B-A3B-Captioner</summary>
   
**Dependency:** None

**Path:** `src/xtalk/speech/captioner/qwen3_omni_captioner.py`
    
[HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)
[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Captioner)
    
</details>
    
