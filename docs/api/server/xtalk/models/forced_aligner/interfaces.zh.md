<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.forced_aligner.interfaces

## ForcedAlignmentUnit

```python
@dataclass(frozen=True)
class ForcedAlignmentUnit
```

One text unit aligned onto a synthesized audio timeline.

### 类字段

- `text: str`
- `start_ms: float`
- `end_ms: float`
- `char_start: int` = `-1`
- `char_end: int` = `-1`

## ForcedAligner

```python
@model_type
class ForcedAligner(ABC)
```

Abstract interface for forced alignment models.

### 方法

#### align

```python
def align(self, *, audio: bytes, text: str, language: str | None = None) -> list[ForcedAlignmentUnit]
```

Align text units against 48 kHz PCM audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 48 kHz.
- `text` (`str`)
  Original text that the audio speaks.
- `language` (`str | None, optional`)
  Optional model-specific language hint.

#### async_align

```python
async def async_align(self, *, audio: bytes, text: str, language: str | None = None) -> list[ForcedAlignmentUnit]
```

Asynchronously align text units against 48 kHz PCM audio.

#### clone

```python
def clone(self) -> 'ForcedAligner'
```

Clone the aligner for a new service session.
