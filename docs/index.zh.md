# X-Talk

> ⚠️ X-Talk 仍处于积极原型开发阶段。接口和功能可能会发生变化。我们会尽量保持接口稳定。

X-Talk 是一个开源的全双工语音对话框架：由 ASR、LLM Agent、TTS 和轮次检测模型组成级联流水线，用户可以随时打断回复。后端为纯 Python，每种模型类型都可以在一个脚本内替换；Web 客户端 SDK 通过 WebSocket 使用 OpenAI Realtime 风格的协议；桌面应用把同一套服务打包为本地程序。

[演示链接](https://xtalk.sjtuxlance.com/)

[GitHub](https://github.com/xcc-zach/xtalk)

<div class="grid cards next-step-cards" markdown>

-   **[教程 →](tutorial/start_the_service.zh.md)** · 循序渐进地学习 X-Talk，从启动服务到自定义模型和工具

-   **[操作指南 →](how_to/deploy_fully_local.zh.md)** · 完成具体任务，例如本地部署、日志和测试

-   **[参考 →](reference/service_config.zh.md)** · 查阅配置项、协议事件和模型接口

-   **[原理说明 →](explanation/system_design.zh.md)** · 理解整体架构以及流水线为何这样运作

</div>
