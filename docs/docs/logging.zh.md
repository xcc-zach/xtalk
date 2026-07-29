# 日志

Xtalk 使用 Python 标准库 `logging`。包负责设置 `xtalk.*` logger 的日志等级，日志的输出位置、格式和 handler 由宿主应用负责。

## 初始化

`src/xtalk/log_utils.py` 提供私有辅助函数 `_initialize_package_logging()`。该函数读取环境变量 `XTALK_LOG_LEVEL`，设置 `xtalk` logger 的等级，并添加 `NullHandler`。

支持的日志等级为：

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

## 模块日志

每个模块使用自身的完整模块名创建 logger：

```python
import logging

logger = logging.getLogger(__name__)
```

模块日志名称形如：

```text
xtalk.api
xtalk.serving.event_bus
xtalk.serving.modules.input_gateway
```

这些 logger 继承 `xtalk` logger 的日志等级。

## 配置日志等级

在导入 Xtalk 前设置环境变量：

```bash
XTALK_LOG_LEVEL=DEBUG python server.py
```

该配置仅控制 `xtalk.*` 日志，不调整 root logger、第三方库或 Uvicorn 的日志等级。

## 配置日志输出

宿主应用负责配置 handler 和日志格式。例如：

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from xtalk import Xtalk
```

未配置输出 handler 时，Xtalk 日志由 `NullHandler` 接收，不主动输出到终端或文件。Uvicorn 的日志配置保持独立。
