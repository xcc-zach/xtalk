<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.log_utils

## mute_other_logging

```python
def mute_other_logging()
```

Reduce noise from third-party loggers used by Xtalk.

### 说明

This helper raises the root logger level to ``WARNING`` and applies the
same threshold to common network and SDK loggers so sample applications
can keep terminal output focused on Xtalk events.

## setup_logging

```python
def setup_logging()
```

Configure the process-wide Xtalk logger.

### 返回

- `logging.Logger`
  The configured ``xtalk`` logger instance.

### 说明

A timestamped log file is created under ``logs/`` for every process start.

## logger

```python
logger
```

**值:** `setup_logging()`
