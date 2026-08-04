# 计时器工具

在 XTalk 的“设置与诊断”中选择当前目录，然后点击“应用并重启本地服务”。

该目录导出名为 `timer` 的 XTalk `AsyncTool`，支持：

- 指定计时秒数；
- 可选的周期进度提醒；
- 查询当前进度；
- 提前停止计时。

因为工具名称是 `timer`，启用后会替换桌面应用内置的回退计时器。

工具的 live 与 history UI 会读取
`window.xtalkToolUI.context.language`，并跟随桌面应用的中英文设置。
