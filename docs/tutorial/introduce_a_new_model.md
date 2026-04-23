> [!NOTE]
> See `examples/sample_app/custom_model.py` and `examples/sample_app/echo_agent.py` for details.
    
> [!NOTE]
> See [Recipe](#recipe) for adding a model of existing types.
    
You may want to introduce a new model of an existing type (e.g. text-to-speech), or add a model of new type (e.g. a model that handles backchannel). This can be achieved by `register_model_search_spec` before a `xtalk_instance` is created from config:
    
```python
from xtalk import Xtalk
Xtalk.register_model_search_spec(
    slot="llm_agent",
    spec=Path(__file__).parent / "echo_agent.py",
)
xtalk_instance = Xtalk.from_config(args.config)
```
    
Here `slot` matches the name of corresponding init arg in `Pipeline`. You can check `Xtalk.MODEL_REGISTRY` for existing slots, or use a new slot to represent a new type of models (see `examples\sample_app\custom_service.py` and there `llm_output_refactor_model` can be the new slot).
    
`spec` is the path to model implementation, an example implementation in `echo_agent.py` looks like this:

```python
from xtalk.model_types import Agent

class EchoAgent(Agent):
    """A simple agent that echoes user input."""

    def generate(self, input) -> str:
        if isinstance(input, dict):
            return input["content"]
        return input

    def clone(self) -> "EchoAgent":
        return EchoAgent()

```
    
Then you can use the custom model in config file:
```python
{
    "asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "<API_KEY>"
        }
    },
    "llm_agent": "EchoAgent",
    "tts": {
        "type": "CosyVoice",
        "params": {
            "api_key": "<API_KEY>"
        }
    }
}
```

#### Recipe

Recipes for major model customization are listed below. You can read source code for interfaces of other model types. We will update these interfaces from time to time.
    
> [!NOTE]
> See `src/xtalk/model_types.py` for all available model types.
    
> [!IMPORTANT]
> X-Talk has asynchronous default implementations for sync versions, which usually with `run_in_executor`, like `async_recognize` for `recognize` w.r.t. ASR. However, in order to achieve best concurrency for production, we recommend to implement these async versions by your self.

##### New ASR (auto-speech-recognition) Model
    
Your ASR class must inherit from `xtalk.speech.interfaces.ASR` and implement the following methods:

* **`recognize(audio: bytes) -> str`**
    * Recognize audio in a single pass. 
* **`reset() -> None`**
    * Reset internal recognition state.
* **`clone() -> ASR`**
    * Return a new instance for use in new or concurrent sessions.
    * Sharing weights/connections (e.g., `_shared_model`) is allowed, but you can't share states. 
    
Methods below are optional:
* **`recognize_stream(audio: bytes, *, is_final: bool = False) -> str`**
    * Interface for streaming incremental recognition.
    * Returns the "current cumulative recognition result up to this point".
* **`async_recognize(audio: bytes)`**
* **`async def async_recognize_stream(
        self, audio: bytes, *, is_final: bool = False
    )`**

    
> [!IMPORTANT]
> Input for `recognize` and `recognize_stream` is PCM 16-bit mono 16 kHz raw bytes. You may need to do conversion by yourself.
    
> [!NOTE]
> X-Talk have default implementation for `recognize_stream` with a `MockStreamRecognizer`. Therefore, no worry for your non-streaming ASR models.

> [!NOTE]
> You can refer to existing implementations (e.g., `src/xtalk/speech/asr/zipformer_local.py`) when building your own ASR class. We recommend deploying ASR as a separate service and invoking it via API calls within the ASR class, referencing the implementation of `src/xtalk/speech/asr/sherpa_onnx_asr.py`.
    
##### New TTS (text-to-speech) Model
    
Your new TTS class must inherit from `xtalk.speech.interfaces.TTS` and implement the following methods:


- **`synthesize(self, text: str) -> bytes`**

  - Input: The text to synthesize.
  - Output: Raw audio bytes in PCM 16-bit, mono, 48000 Hz.

- **`clone(self) -> TTS`**

  - Return a new TTS instance:
    - It should have isolated runtime state to avoid cross-session interference and it may share read-only resources if your backend supports that.
    
**Optional methods**

- **`synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]`**
  - If your backend supports streaming synthesis, you can override this method.
- **`set_voice(self, voice_names: list[str])`**

  - This method works with the `TTSVoiceChange` event in `TTSManager` to switch voices via language model tool calls.
  - Usually there is only one element in `voice_names`, and this is the current behavior for tool call result. However, some TTS models may support mixing multiple voices for reference. Therefore, `voice_names` is list type.

- **`set_emotion(self, emotion: str | list[float])`**

  - This method works with the `TTSEmotionChange` event in `TTSManager` to switch emotions via language model tool calls.
  - Current tool call result only carries `emotion` as `str`. However, you may also want `list[float]` as emotion vector for future use.
    
- **`async def async_synthesize(self, text: str, **kwargs: Any)`**
- **`async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    )`**    
    