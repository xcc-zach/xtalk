<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.model_loader

Shared model loading helpers for configuration-driven model construction.

## init_configured_model

```python
def init_configured_model(*, slot: str, config: dict[str, Any], registry: dict[str, list[ImportSpec]] | None = None) -> Any
```

Instantiate a model for a model slot from a config dictionary.

### 参数

- `slot` (`str`)
  Model registry slot.
- `config` (`dict[str, Any]`)
  Full service configuration dictionary.
- `registry` (`dict[str, list[ImportSpec]] | None, optional`)
  Registry to query. When omitted, the shared internal registry is used.

## init_registered_model

```python
def init_registered_model(*, slot: str, model_config: dict | str, registry: dict[str, list[ImportSpec]] | None = None) -> Any
```

Instantiate a model by looking up its import specs from a registry slot.

### 参数

- `slot` (`str`)
  Registry slot such as ``"asr"`` or ``"tts"``.
- `model_config` (`dict | str`)
  Model config containing ``type`` and optional ``params``, or a shorthand
  type string.
- `registry` (`dict[str, list[ImportSpec]] | None, optional`)
  Registry to query. When omitted, the shared internal registry is used.

### 返回

- `Any`
  Instantiated model object, or ``None`` when ``model_config`` is empty.
