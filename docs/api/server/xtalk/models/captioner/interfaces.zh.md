<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.captioner.interfaces

## Captioner

```python
@model_type(aliases=['captioner'])
class Captioner(ABC)
```

Abstract base class for audio captioning models.

### 方法

#### caption

```python
def caption(self, audio: bytes) -> str
```

Generate a caption for audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `str`
  Generated caption text.

#### caption_stream

```python
def caption_stream(self, audio: bytes) -> Iterable[str]
```

Stream caption text for audio input.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 生成

- `str`
  Streamed caption text.

#### async_caption

```python
async def async_caption(self, audio: bytes) -> str
```

Asynchronously caption audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `str`
  Generated caption text.

#### async_caption_stream

```python
async def async_caption_stream(self, audio: bytes) -> AsyncIterator[str]
```

Asynchronously stream caption text.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 生成

- `str`
  Streamed caption text.
