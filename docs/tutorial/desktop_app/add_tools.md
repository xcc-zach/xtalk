# Add Tools

In this tutorial you build a dice-rolling tool in a local directory, install it into the X-Talk desktop app, and finish with the Agent calling that tool during a conversation. A second tool then gets its own UI panel. This page assumes the app already runs with a configuration from [Model Configuration](model_configuration.md).

The app loads user tools from a directory that contains at least a tool implementation and an `xtalk_tool.json` manifest.

## 1. Create the tool directory

Create the following layout:

```text
dice-tool/
├── dice_tool.py
└── xtalk_tool.json
```

`dice_tool.py` defines the tool and exports it through a zero-argument factory. The factory is the entry point the app calls when loading the directory:

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

`xtalk_tool.json` names the tool in the app and points to the factory using `module:factory` syntax:

```json
{
  "display_name": {
    "zh": "掷骰子",
    "en": "Roll Dice"
  },
  "entrypoint": "dice_tool:create_tools"
}
```

The factory must return a list. The list can contain LangChain tools, X-Talk `SyncTool` or `AsyncTool` classes, and zero-argument tool factories. Tools run in the Python environment bundled with the app, so a tool that imports a third-party package the app does not ship fails to load.

## 2. Install the tool

1. Open the expandable menu in the upper-left corner, then select **Tools**.
2. Select **Install tool from directory**, then choose `dice-tool/`.
3. Make sure the tool is enabled.
4. Select **Apply and restart local service**.

After the restart, **Roll Dice** is listed under **Tools** and registered with the Agent. Start a conversation and ask for a dice roll; the reply contains a number from the tool. Disabling or removing the tool under **Tools** and applying again takes it out of the Agent.

## 3. Add a tool UI

A tool UI shows live status and historical results of a native X-Talk `AsyncTool`. This step adds one to a timer tool; the mechanism is the same for any `AsyncTool` directory. Add a self-contained HTML file:

```text
timer-tool/
├── timer_tool.py
├── ui/
│   └── index.html
└── xtalk_tool.json
```

Declare it in the manifest with a `ui` field:

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

In `ui/index.html`, receive tool events through `window.xtalkToolUI`:

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

Registering `status()` creates a live status card in the running-tools area above the conversation. Registering `emit()` creates a result card in the matching conversation entry. Register only the callbacks the UI needs.

`window.xtalkToolUI.context` provides:

- `language`: the language currently selected in the app, such as `zh-CN` or `en`.
- `mode`: where this instance is placed. `live` is the status above the conversation while the tool runs; it updates with the tool and is removed when the tool finishes. `history` is a result snapshot stored in the conversation entry and does not change with later live-status updates.

Install `timer-tool/` as in step 2. While the timer runs, the card above the conversation shows its status; once it finishes, the result card stays in the conversation entry. After editing the UI, add the tool again and select **Apply and restart local service** to load the new file.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Web Search →](../../how_to/desktop_app/web_search.md)** · How-to

-   **[Tools →](../../reference/tools.md)** · Reference

-   **[Attach Tools to a server Agent →](../attach_tools.md)** · Tutorial

</div>
