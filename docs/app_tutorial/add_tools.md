# Add Tools

The X-Talk desktop app supports user tools installed from a local directory. A tool directory contains at least a tool implementation and an `xtalk_tool.json` manifest.

## Create a tool directory

The following directory implements a simple dice tool:

```text
dice-tool/
├── dice_tool.py
└── xtalk_tool.json
```

Define the tool in `dice_tool.py` and return a list of tools from a zero-argument factory:

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

Create `xtalk_tool.json` and specify the entrypoint using `module:factory` syntax:

```json
{
  "display_name": {
    "zh": "掷骰子",
    "en": "Roll Dice"
  },
  "entrypoint": "dice_tool:create_tools"
}
```

The factory must return a list. The list can contain LangChain tools, X-Talk `SyncTool` or `AsyncTool` classes, and zero-argument tool factories. Tools run in the Python environment bundled with the X-Talk desktop app. A tool cannot be loaded if it depends on a third-party Python package that is not included in the app.

## Add the tool to the desktop app

1. Open the expandable menu in the upper-left corner, then select **Tools**.
2. Select **Install tool from directory**, then choose the directory containing `xtalk_tool.json`.
3. Make sure the tool is enabled.
4. Select **Apply and restart local service**.

After applying the change, the new tool is registered with the Agent. You can later disable or remove a user tool from **Tools**, then apply the change again.

## Add a tool UI

A tool UI currently applies to native X-Talk `AsyncTool` classes and can display live status and historical results from asynchronous tools. First, add a self-contained HTML file to the tool directory:

```text
timer-tool/
├── timer_tool.py
├── ui/
│   └── index.html
└── xtalk_tool.json
```

Then add the `ui` field to the manifest:

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

- `entrypoint` is the path to the HTML file relative to the tool directory.
- `update_every_s` is the status refresh interval in seconds while the tool runs. It defaults to `1`.

Use `window.xtalkToolUI` in `ui/index.html` to receive tool events:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { margin: 0; padding: 12px; font-family: sans-serif; }
      #card { padding: 12px; border: 1px solid #8886; border-radius: 12px; }
    </style>
  </head>
  <body>
    <div id="card">Waiting for the tool</div>
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

Registering `status()` creates a live status UI in the running-tools area above the conversation. Registering `emit()` creates a historical result UI in the corresponding conversation entry. Register only the callbacks the UI needs.

`window.xtalkToolUI.context` provides:

- `language`: the language currently selected in the desktop app, such as `zh-CN` or `en`.
- `mode`: the current placement. `live` is the status shown above the conversation while the tool runs; it updates with the tool and is removed when the tool finishes. `history` is a result snapshot stored in the corresponding conversation entry and does not change with later live-status updates.

After changing a tool UI, add the tool again and select **Apply and restart local service**.
