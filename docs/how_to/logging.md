# Logging

This guide sets the log level and output format of a running X-Talk server; it assumes you start the server from your own Python script as in [Start the Service](../tutorial/start_the_service.md).

## Configure the Log Level

Set the logging level with `XTALK_LOG_LEVEL` when starting the service:

```bash
XTALK_LOG_LEVEL=DEBUG python server.py
```

Supported levels are:

- `DEBUG`: detailed diagnostic information;
- `INFO`: normal runtime information;
- `WARNING`: warnings and errors only;
- `ERROR`: errors only;
- `CRITICAL`: critical errors only.

Use `INFO` for normal operation and temporarily switch to `DEBUG` when troubleshooting.

## Configure the Output Format

When using a custom Python server, configure the log format before importing X-Talk:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from xtalk import Xtalk
```

This displays the timestamp, source, level, and message. To save logs to a file, use the `filename` argument of `logging.basicConfig()`:

```python
logging.basicConfig(
    filename="xtalk.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

Configure Uvicorn access and error logs through Uvicorn's own command-line or logging options.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Service Configuration →](../reference/service_config.md)** · Reference

-   **[Testing →](testing.md)** · How-to

</div>
