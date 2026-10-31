# GPTLive 单次通话示例

本实现适配新版 `dev` 的 `/v1/realtime` 入口。纯 SDK 负责实时连接与媒体，示例应用负责登录；Live 通话跳过历史保存与恢复。

## 启动

使用已有开发环境并构建浏览器库：

```bash
conda activate xtalk
python -m pip install -e ".[example]"
cd frontend
npm install
npm run build
cd ..
```

复制模板，创建本地配置：

```bash
cp examples/sample_app/configs/openai_gpt_live.example.json \
  examples/sample_app/configs/openai_gpt_live.json
```

在本地配置的 `end_to_end.params.api_key` 中填写 API key。GPTLive 仅从该字段读取密钥。本地配置已加入 Git 忽略；`.example.json` 模板使用占位符。密钥保留在服务端。

如需通过普通 GPT 调用测试密钥，先复制占位脚本：

```bash
cp examples/sample_app/test_openai_key.example.py examples/sample_app/test_openai_key.py
```

在复制后的脚本中填写 `api_key`，用 Python 运行。本地 `test_openai_key.py` 已加入 Git 忽略；提交时只包含使用占位符的模板。

```bash
PYTHONPATH=src python examples/sample_app/configurable_server.py \
  --port 7636 \
  --config examples/sample_app/configs/openai_gpt_live.json
```

打开 <http://localhost:7636>，构建后刷新页面，点击开始。等待连接中状态结束再讲话。同一个按钮可以取消启动或停止通话。静音只影响输入，输出继续；静音偏好重开后保留。转写展示在原聊天区域，停止后保留，下次开始或刷新时清空。错误也会在默认语音视图通过 Toast 展示。

示例使用 `gpt-live-1`、`marin`，显式 Responses delegation（`gpt-5.6-luna`、空工具列表、`tool_choice: "none"`）。保留预期的后端配置并确认模型访问权限。每次通话使用空上下文。Live 的历史、自动重连、暂停声音、上传、动态音色、录音和自定义工具留待后续阶段。

## 媒体与架构

模型的 `end_to_end.params.sample_rate` 支持 16000/24000，默认 24000，同时决定上下行格式。浏览器 AEC 保留，额外 VAD 与增强关闭。静音时输入层生成等时长零 PCM。

适配器有一个按音频节奏工作的上游发送者和一个独立接收者。Manager 独立消费模型事件，播放等待仅发生在投递任务中，待播 PCM 有界。播放预算位于 `end_to_end_manager.py` 内部，沿用原版 TTS 的方法：每块 100 ms、在途 300 ms，窗口满后等待。`TTSManager` 保留原有实现。待播音频默认最多 5 秒；有未确认音频且 15 秒无播放进展时显示错误。TTS 与 Live 共用一个浏览器 PCM 排程核心，详见[协议 2 规格](../technical_reference/end_to_end_live_spec.zh.md)。

Live 默认关闭持久化。显式启用延期能力会在启动前报错。原级联配置和已存数据库数据继续保留。

## 本地 mock

分别在两个终端运行：

```bash
python examples/sample_app/mock_live_server.py --port 8765
```

```bash
PYTHONPATH=src python examples/sample_app/configurable_server.py \
  --port 7636 \
  --config examples/sample_app/configs/mock_gpt_live.json
```

测试服务收到首段非零输入后返回转写及连续六块 100 ms 音频。仅监听回环地址的 `/fixture` 提供测试指标及慢启动/错误场景控制；本地使用无需项目密钥。

## 自动化检查

```bash
PYTHONPATH=src python -m pytest tests/test_gpt_live.py tests/test_audio_delivery_failures.py
cd frontend
npx tsc --noEmit
npm test
npm run build
cd ..
```

两个 mock 服务运行后，在安装了 Playwright 与 Chromium 的测试环境运行：

```bash
XTALK_DEMO_URL=http://127.0.0.1:7636 \
XTALK_FIXTURE_URL=http://127.0.0.1:8765 \
node tests/frontend/live_browser.mjs
```

Playwright 位于 Node 常规搜索路径之外时，用 `XTALK_PLAYWRIGHT_MODULE` 指定其 ESM 入口。浏览器测试使用合成采集，覆盖突发输出、AudioContext 暂停恢复、`onended` 缺失、转写、静音、十次新通话、权限拒绝、启动取消、慢启动、命令拒绝、断线、清理及无历史请求。后端测试覆盖队列上限、停滞取消、16/24 kHz、关闭确认、协议检查和旧数据库保留。

## 新版 dev 适配验证（2026-10-31）

- 基于 `dev` 的 `461df8f`，迁移原可运行 Live 版本 `ca6b7c9`。后端保留 `gpt-live-1` 适配器；前后端接入新版 Realtime 会话、顺序传输队列与事件扩展机制。
- 后端 269 项通过，另有 24 个子测试通过；前端 10 项协议/生命周期测试、类型检查、生产构建和 mock 浏览器流程通过。重点覆盖新协议 Base64/二进制、16/24 kHz、无历史写入、突发播放、静音、十次重开和旧回调隔离。
- 完整 Python 测试有两项原有说话人适配器取消测试失败。已用未修改的 `dev` 源码复现：当前 Python 3.10 缺少它们调用的 `Task.cancelling()`。本次保留原状。
- 修正本地凭据后，真实上游验证通过：原配置值比用户成功调用 GPT 的测试密钥多了 19 个字符。适配器收到语音、转写和输出音频，静音/解除静音确认及完整关闭通过。浏览器使用合成语音完成两次独立通话，问答、播放进度、静音上传零 PCM、静音偏好保留、重开和媒体资源清理通过；未出现上游或页面错误，也未请求历史接口。

这些自动化浏览器测试使用合成麦克风输入。真人听感、自然附和、真实环境回声和原生插话质量仍需手动验收。启动后检查普通问答、简短附和、助手说话时插话、静音、停止与重开。打断由 Live 判断，客户端不承诺逐字回滚。
