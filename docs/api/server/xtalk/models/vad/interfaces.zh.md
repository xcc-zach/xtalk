<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.vad.interfaces

## VAD

```python
@model_type(aliases=['vad'])
class VAD(ABC)
```

Abstract base class for voice activity detection engines.

### 方法

#### is_speech

```python
def is_speech(self, frame: bytes) -> bool
```

Determine whether an audio frame contains speech.

##### 参数

- `frame` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `bool`
  ``True`` if speech is detected, otherwise ``False``.

#### async_is_speech

```python
async def async_is_speech(self, frame: bytes) -> bool
```

Asynchronously determine whether an audio frame contains speech.

##### 参数

- `frame` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `bool`
  ``True`` if speech is detected, otherwise ``False``.

#### clone

```python
def clone(self) -> 'VAD'
```

Clone the VAD instance for a new session.

##### 返回

- `VAD`
  Clone with shared weights and independent runtime state.
