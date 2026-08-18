<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.speaker_encoder.interfaces

## SpeakerEncoder

```python
@model_type(aliases=['speaker_encoder'])
class SpeakerEncoder(ABC)
```

Abstract base class for speaker embedding models.

### 方法

#### extract

```python
def extract(self, audio: bytes) -> np.ndarray
```

Generate a speaker embedding vector.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes.

##### 返回

- `np.ndarray`
  Speaker embedding vector.

#### async_extract

```python
async def async_extract(self, audio: bytes) -> np.ndarray
```

Asynchronously extract a speaker embedding.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes.

##### 返回

- `np.ndarray`
  Speaker embedding vector.

#### similarity

```python
def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float
```

Compute similarity between two speaker embeddings.

##### 参数

- `embedding1` (`np.ndarray`)
  First speaker embedding.
- `embedding2` (`np.ndarray`)
  Second speaker embedding.

##### 返回

- `float`
  Cosine similarity score.
