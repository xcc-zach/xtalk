# Logging

Xtalk uses Python's standard `logging` library. The package sets the logging level for `xtalk.*` loggers, while the host application controls output destinations, formats, and handlers.

## Initialization

`src/xtalk/log_utils.py` provides the private helper `_initialize_package_logging()`. It reads the `XTALK_LOG_LEVEL` environment variable, sets the level of the `xtalk` logger, and adds a `NullHandler`.

The following logging levels are supported:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

## Module Loggers

Each module creates a logger using its fully qualified module name:

```python
import logging

logger = logging.getLogger(__name__)
```

Logger names have forms such as:

```text
xtalk.api
xtalk.serving.event_bus
xtalk.serving.modules.input_gateway
```

These loggers inherit the logging level of the `xtalk` logger.

## Configure the Logging Level

Set the environment variable before importing Xtalk:

```bash
XTALK_LOG_LEVEL=DEBUG python server.py
```

This setting only controls `xtalk.*` logs. It does not change the logging levels of the root logger, third-party libraries, or Uvicorn.

## Configure Log Output

The host application is responsible for configuring handlers and log formats. For example:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from xtalk import Xtalk
```

When no output handler is configured, Xtalk logs are received by the `NullHandler` and are not written to the terminal or files. Uvicorn keeps its own independent logging configuration.
