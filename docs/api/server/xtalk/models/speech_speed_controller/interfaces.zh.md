<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.speech_speed_controller.interfaces

## SpeechSpeedController

```python
@model_type(aliases=['speech_speed_controller'])
class SpeechSpeedController(ABC)
```

Interface for TTS speed controllers.

### 方法

#### process

```python
def process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes
```

Apply a speed adjustment to synthesized audio.

##### 参数

- `audio_bytes` (`bytes`)
  Synthesized audio bytes.
- `speed` (`float, optional`)
  Speed multiplier.

##### 返回

- `bytes`
  Processed audio bytes.

#### async_process

```python
async def async_process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes
```

Asynchronously apply a speed adjustment to audio.

##### 参数

- `audio_bytes` (`bytes`)
  Synthesized audio bytes.
- `speed` (`float, optional`)
  Speed multiplier.

##### 返回

- `bytes`
  Processed audio bytes.
