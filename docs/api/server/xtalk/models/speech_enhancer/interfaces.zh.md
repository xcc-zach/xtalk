<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.speech_enhancer.interfaces

## SpeechEnhancer

```python
@model_type(aliases=['speech_enhancer'])
class SpeechEnhancer(ABC)
```

Abstract base class for speech enhancement engines.

### 说明

Inputs and outputs use PCM 16-bit mono audio bytes at 16 kHz.

### 方法

#### enhance

```python
def enhance(self, audio: bytes) -> bytes
```

Enhance an audio frame.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `bytes`
  Enhanced PCM audio bytes.

#### flush

```python
def flush(self) -> bytes
```

Flush any internally buffered audio.

##### 返回

- `bytes`
  Remaining enhanced PCM audio bytes.

#### async_enhance

```python
async def async_enhance(self, audio: bytes) -> bytes
```

Asynchronously enhance audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `bytes`
  Enhanced PCM audio bytes.

#### async_flush

```python
async def async_flush(self) -> bytes
```

Asynchronously flush buffered audio.

##### 返回

- `bytes`
  Remaining enhanced PCM audio bytes.

#### reset

```python
def reset(self) -> None
```

Reset internal buffers and caches.

#### clone

```python
def clone(self) -> 'SpeechEnhancer'
```

Clone the speech enhancer for a new session.

##### 返回

- `SpeechEnhancer`
  Clone with shared weights and isolated runtime state.
