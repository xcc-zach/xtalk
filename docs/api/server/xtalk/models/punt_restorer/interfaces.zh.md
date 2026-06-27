<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.punt_restorer.interfaces

## PuntRestorer

```python
@model_type(aliases=['punt_restorer_model'])
class PuntRestorer(ABC)
```

Abstract base class for punctuation restoration models.

### 方法

#### restore

```python
def restore(self, text: str) -> str
```

Restore punctuation in text.

##### 参数

- `text` (`str`)
  Text without reliable punctuation.

##### 返回

- `str`
  Text with restored punctuation.

#### async_restore

```python
async def async_restore(self, text: str) -> str
```

Asynchronously restore punctuation in text.

##### 参数

- `text` (`str`)
  Text without reliable punctuation.

##### 返回

- `str`
  Restored text.
