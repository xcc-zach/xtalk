### Speech Recognition
    
**Slot**: `asr`

[SherpaOnnx](https://github.com/k2-fsa/sherpa-onnx) is recommended for its wide support of models and optimized inference performance.
    
<details>
<summary>SherpaOnnx</summary>
    
**Dependency:** `pip install "xtalk[sherpa-onnx-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/sherpa_onnx_asr.py`
    
A high-performance speech recognition framework and beyond.

[Repo](https://github.com/k2-fsa/sherpa-onnx)
    
[Models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
    
[Tutorial to start speech recognition server](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)
    
</details>

<details>
<summary>Qwen3ASRFlashRealtime</summary>
    
**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/qwen3_asr_flash_realtime.py`
    
[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)
    
</details>
    
<details>
<summary>Zipformer</summary>
    
**Dependency:** `pip install "xtalk[zipformer-local] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/zipformer_local.py`
    
[Details](https://www.modelscope.cn/models/yhdai666/xtalk_zipformer_onnx/summary)
    
</details>
    
<details>
<summary>ElevenLabs</summary>
    
**Dependency:** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/elevenlabs.py`
    
[API Reference](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
    
</details>
    
### Text to Speech
    
**Slot**: `tts`
    
<details>
<summary>IndexTTS</summary>

**Dependency:** `pip install "xtalk[index-tts] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** 
- `src/xtalk/speech/tts/index_tts.py`
- `src/xtalk/speech/tts/index_tts2.py`
    
[Repo](https://github.com/index-tts/index-tts)
    
[Installation (vllm boost)](https://github.com/Ksuriuri/index-tts-vllm)

    
</details>
    
<details>
<summary>GPT-SoVITS</summary>

> Experimental. Feel free to open an issue for any problem.

**Dependency:** `pip install "xtalk[gpt-sovits] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/tts/gpt_sovits.py`

    
[Repo](https://github.com/RVC-Boss/GPT-SoVITS)
    

    
</details>
    
<details>
<summary>CosyVoice</summary>

**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/tts/cosyvoice.py`

[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)
    
</details>
    
<details>
<summary>ElevenLabs</summary>
    
**Dependency:** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/tts/elevenlabs.py`
    
[API Reference](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
    
</details>

### Voice Activity Detection
    
**Slot**: `vad`
    
X-Talk has VAD on client side, so you may not need one.

<details>
<summary>Silero VAD</summary>

**Dependency:** `pip install "xtalk[silero-vad] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/vad/silero_vad.py`
    
[Model Details](https://github.com/snakers4/silero-vad)
[VAD-Web](https://github.com/ricky0123/vad)
    
</details>
    
### Speech Enhancement
    
**Slot**: `speech_enhancer`

<details>
<summary>FastEnhancer</summary>
    
**Dependency:** `pip install onnxruntime`
**Path:** `src/xtalk/speech/speech_enhancer/speech_enhancer.py`
    
[Model Details](https://github.com/aask1357/fastenhancer)
    
</details>
    
### Speaker Recognition
    
**Slot**: `speaker_encoder`    
    
<details>
<summary>Wespeaker-Voxceleb-Resnet34-LM</summary>
  
**Dependency:** `pip install "xtalk[pyannote] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/speaker_encoder/pyannote_embedding.py`
    
[Wespeaker](https://github.com/wenet-e2e/wespeaker)
[Model Details](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
    
</details>
    
### Captioner
    
**Slot**: `captioner`
    
Captioners give you description of audio clip.
    
<details>
<summary>Qwen3-Omni-30B-A3B-Captioner</summary>
   
**Dependency:** None
**Path:** `src/xtalk/speech/captioner/qwen3_omni_captioner.py`
    
[HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)
[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Captioner)
    
</details>
    
