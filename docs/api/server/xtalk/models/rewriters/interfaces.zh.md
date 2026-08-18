<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.rewriters.interfaces

## Rewriter

```python
@model_type(aliases=['caption_rewriter'])
class Rewriter(ABC)
```

Abstract interface for text rewriting helpers.

### 方法

#### rewrite

```python
def rewrite(self, input: str) -> str
```

Rewrite input text.

##### 参数

- `input` (`str`)
  Source text to rewrite.

##### 返回

- `str`
  Rewritten text.

#### async_rewrite

```python
async def async_rewrite(self, input: str) -> str
```

Asynchronously rewrite input text.

##### 参数

- `input` (`str`)
  Source text to rewrite.

##### 返回

- `str`
  Rewritten text.
