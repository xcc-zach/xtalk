# 添加工具

本教程在本地目录中编写一个掷骰子工具，把它安装到 X-Talk 桌面应用，最终让 Agent 在对话中调用这个工具；随后再为另一个工具加上专属界面。本页假定应用已按[模型配置](model_configuration.zh.md)加载配置并运行。

应用从一个目录加载用户工具，目录至少包含工具实现文件和 `xtalk_tool.json` 清单。

## 1. 创建工具目录

建立如下结构：

```text
dice-tool/
├── dice_tool.py
└── xtalk_tool.json
```

`dice_tool.py` 定义工具，并通过一个无参数工厂导出。工厂是应用加载目录时调用的入口：

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

`xtalk_tool.json` 给出工具在应用中的名称，并用 `模块名:工厂名` 指向工厂：

```json
{
  "display_name": {
    "zh": "掷骰子",
    "en": "Roll Dice"
  },
  "entrypoint": "dice_tool:create_tools"
}
```

工厂必须返回一个列表。列表可以包含 LangChain 工具、X-Talk `SyncTool` 或 `AsyncTool` 类，以及无参数工具工厂。工具运行在应用自带的 Python 环境中，因此导入了应用未附带的第三方包的工具会加载失败。

## 2. 安装工具

1. 打开左上角的展开栏，选择其中的**工具**栏目；
2. 点击**从目录安装工具**，选择 `dice-tool/`；
3. 确认工具已启用；
4. 点击**应用并重启本地服务**。

重启后，**掷骰子**出现在**工具**列表中并注册到 Agent。开始一次对话并要求掷骰子，回复里会包含工具给出的数字。在**工具**中禁用或删除该工具并再次应用，即可把它从 Agent 中移除。

## 3. 添加工具界面

工具界面用于显示 X-Talk 原生 `AsyncTool` 的实时状态和历史结果。本步为一个计时器工具添加界面，任何 `AsyncTool` 目录的做法相同。先加入一个自包含的 HTML 文件：

```text
timer-tool/
├── timer_tool.py
├── ui/
│   └── index.html
└── xtalk_tool.json
```

在清单中用 `ui` 字段声明它：

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

注册 `status()` 会在对话顶部的运行中工具区域创建实时状态卡片；注册 `emit()` 会在对应的对话记录中创建结果卡片。只注册界面需要的回调即可。

`window.xtalkToolUI.context` 提供：

- `language`：应用当前使用的语言，例如 `zh-CN` 或 `en`；
- `mode`：当前实例所处的位置。`live` 是工具运行期间显示在对话顶部的实时状态，会随工具状态更新，并在工具结束后移除；`history` 是保存在对话记录中的结果快照，创建后不再随实时状态变化。

按第 2 步安装 `timer-tool/`。计时器运行时，对话顶部的卡片显示其状态；结束后，结果卡片保留在对应的对话记录中。修改界面后，重新添加工具并点击**应用并重启本地服务**以加载新文件。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[联网搜索 →](../../how_to/desktop_app/web_search.zh.md)** · 操作指南

-   **[工具 →](../../reference/tools.zh.md)** · 参考

-   **[为服务端 Agent 挂载工具 →](../attach_tools.zh.md)** · 教程

</div>
