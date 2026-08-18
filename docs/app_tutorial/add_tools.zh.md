# 添加工具

X-Talk 桌面应用支持从本地目录添加用户工具。一个工具目录至少包含工具实现文件和 `xtalk_tool.json` 清单。

## 创建工具目录

以下目录实现了一个简单的掷骰子工具：

```text
dice-tool/
├── dice_tool.py
└── xtalk_tool.json
```

在 `dice_tool.py` 中定义工具，并通过一个无参数工厂返回工具列表：

```python
from langchain_core.tools import tool


@tool
def roll_dice(sides: int = 6) -> str:
    """Roll a die with the requested number of sides."""

    import random

    return str(random.randint(1, sides))


def create_tools() -> list:
    """Return the tools exported by this directory."""

    return [roll_dice]
```

创建 `xtalk_tool.json`，使用 `模块名:工厂名` 指定入口：

```json
{
  "display_name": {
    "zh": "掷骰子",
    "en": "Roll Dice"
  },
  "entrypoint": "dice_tool:create_tools"
}
```

工厂必须返回一个列表。列表可以包含 LangChain 工具、X-Talk `SyncTool` 或 `AsyncTool` 类，以及无参数工具工厂。工具运行在 X-Talk 桌面应用自带的 Python 环境中。如果工具依赖应用未包含的第三方 Python 包，将无法加载。

## 添加到桌面应用

1. 打开左上角的展开栏，选择其中的**工具**栏目；
2. 点击**从目录安装工具**，选择包含 `xtalk_tool.json` 的工具目录；
3. 确认工具已启用；
4. 点击**应用并重启本地服务**。

应用后，新工具会注册到 Agent。之后可以在**工具**中禁用或删除用户工具，并再次应用更改。

## 添加工具界面

工具界面目前适用于 X-Talk 原生 `AsyncTool`，用于显示异步工具的实时状态和历史结果。首先在工具目录中添加自包含 HTML 文件：

```text
timer-tool/
├── timer_tool.py
├── ui/
│   └── index.html
└── xtalk_tool.json
```

然后在清单中添加 `ui` 字段：

```json
{
  "display_name": {
    "zh": "计时器",
    "en": "Timer"
  },
  "entrypoint": "timer_tool:create_tools",
  "ui": {
    "entrypoint": "ui/index.html",
    "update_every_s": 0.5
  }
}
```

- `entrypoint` 是相对于工具目录的 HTML 文件路径；
- `update_every_s` 是运行期间刷新状态的间隔秒数，默认为 `1`。

在 `ui/index.html` 中，通过 `window.xtalkToolUI` 接收工具事件：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { margin: 0; padding: 12px; font-family: sans-serif; }
      #card { padding: 12px; border: 1px solid #8886; border-radius: 12px; }
    </style>
  </head>
  <body>
    <div id="card">等待工具运行</div>
    <script>
      const card = document.querySelector("#card");
      const { language, mode } = window.xtalkToolUI.context;
      document.documentElement.lang = language;
      document.documentElement.dataset.mode = mode;

      window.xtalkToolUI.status((event) => {
        card.textContent = event.status;
      });

      window.xtalkToolUI.emit((event) => {
        card.textContent = event.message || event.status || event.outcome;
      });
    </script>
  </body>
</html>
```

`status()` 注册实时状态界面，显示在对话顶部的运行中工具区域；`emit()` 注册历史结果界面，显示在对应的对话记录中。可以只注册需要的回调。

`window.xtalkToolUI.context` 提供：

- `language`：桌面应用当前使用的语言，例如 `zh-CN` 或 `en`；
- `mode`：当前界面位置。`live` 是工具运行期间显示在对话顶部的实时状态，会随工具状态更新，并在工具结束后移除；`history` 是保存在对应对话记录中的结果快照，创建后不再随实时状态变化。

修改工具界面后，需要重新添加工具并点击**应用并重启本地服务**。
