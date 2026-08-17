# 日志

X-Talk 日志可以帮助您查看服务运行状态和排查问题。

## 配置日志等级

启动服务时通过 `XTALK_LOG_LEVEL` 设置日志等级：

```bash
XTALK_LOG_LEVEL=DEBUG python server.py
```

支持以下等级：

- `DEBUG`：输出详细调试信息，适合排查问题；
- `INFO`：输出一般运行信息；
- `WARNING`：只输出警告和错误；
- `ERROR`：只输出错误；
- `CRITICAL`：只输出严重错误。

日常运行建议使用 `INFO`，需要排查问题时临时改为 `DEBUG`。

## 配置输出格式

如果使用自定义 Python 服务，可以在导入 X-Talk 前配置日志格式：

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from xtalk import Xtalk
```

这会在日志中显示时间、来源、等级和消息。需要保存到文件时，可以使用 `logging.basicConfig()` 的 `filename` 参数：

```python
logging.basicConfig(
    filename="xtalk.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

Uvicorn 的访问日志和错误日志需要通过 Uvicorn 自己的启动参数或日志配置管理。
