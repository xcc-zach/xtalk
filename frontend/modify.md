# uniapp 兼容 H5 + 微信小程序改造清单

## 1. 先说结论

你应该**修改 TS 源码并重新编译**，不要长期直接改编译后的 JS。

当前这套 SDK 已经有良好的分层：

- `bases/*` 是抽象接口层
- `platforms/web.ts` 是 Web 实现层
- `audio-session.ts` / `websocket.ts` 是平台工厂层
- `core.ts` 是业务编排层

所以最合理的路线是：

1. 保持 `core.ts` 和 `action-handler` 业务主流程尽量不变。
2. 新增一个 `platforms/mp-weixin.ts` 来承载小程序实现。
3. 在平台识别 + 工厂层把 `Web` 和 `MpWeixin` 两套实现接起来。

---

## 2. 当前不兼容点对应到哪些文件

你提到的 3 类浏览器 API，不兼容点在源码中的位置如下：

| 不兼容 API | 当前来源文件 | 说明 |
| --- | --- | --- |
| `navigator.mediaDevices.getUserMedia` | `src/platforms/web.ts` | `WebInputAudioSession.setupAudioPipeline()` 里采集麦克风 |
| `window.AudioContext` / `AudioWorkletNode` | `src/platforms/web.ts` | 输入处理（worklet）和输出播放都依赖 Web Audio |
| `document.createElement("script")` | `src/platforms/web.ts` | `ensureModelsEnv()` 通过动态 script 注入 ONNX/VAD |
| `window/document` 平台判断 | `src/utils.ts` | `getPlatform()` 只识别 Web |
| 工厂默认只返回 Web 实现 | `src/audio-session.ts`, `src/websocket.ts` | `switch(getPlatform())` 只有 `Platform.Web` |

---

## 3. 必须修改的现有文件（第一优先级）

## 3.1 `src/utils.ts`

目标：把平台识别从“仅 Web”扩展为“Web + 微信小程序”。

建议改造：

1. 扩展 `Platform` 枚举：新增 `MpWeixin`。
2. `getPlatform()` 中增加 uniapp/小程序判定逻辑。
3. 判断顺序建议：先判小程序，再判 Web，最后抛错。

可参考判定思路（示意）：

- `typeof uni !== 'undefined'` 且存在小程序 API（如 `uni.getSystemInfoSync`）。
- 或使用 uni 条件编译常量（如果你的构建链可用）。

## 3.2 `src/audio-session.ts`

目标：工厂层按平台返回不同输入/输出会话实现。

建议改造：

1. 保留现有 `WebInputAudioSession` / `WebOutputAudioSession`。
2. 新增导入 `MpInputAudioSession` / `MpOutputAudioSession`。
3. 在 `switch(getPlatform())` 中新增 `Platform.MpWeixin` 分支。

## 3.3 `src/websocket.ts`

目标：工厂层按平台返回不同 WebSocket 实现。

建议改造：

1. 保留 `WebWebSocket`。
2. 新增导入 `MpWebSocket`。
3. 在 `switch(getPlatform())` 中新增 `Platform.MpWeixin` 分支。

---

## 4. 需要新增的文件（第二优先级）

## 4.1 `src/platforms/mp-weixin.ts`（核心新增）

目标：实现与 `platforms/web.ts` 对齐的“小程序版本”三件套：

1. `MpWebSocket extends BaseWebSocket`
2. `MpInputAudioSession extends BaseInputAudioSession`
3. `MpOutputAudioSession extends BaseOutputAudioSession`

### A. `MpWebSocket` 实现要点

- 使用 `uni.connectSocket` / `SocketTask.send` / `SocketTask.close`。
- 将 `onOpen/onMessage/onClose/onError` 适配到 `addEventListener` 语义。
- 注意 `onMessage` 里二进制类型转换（`ArrayBuffer`）。

### B. `MpInputAudioSession` 实现要点

Web 版是 `getUserMedia + AudioWorklet`；小程序版应改为：

- 使用 `uni.getRecorderManager()` 采集音频。
- 尽量输出后端可接受的 PCM（若只能拿压缩格式，需要解码链路）。
- VAD/Enhancer 建议分阶段：
	- 第一阶段：先关闭端侧 VAD/Enhancer，只做直传；
	- 第二阶段：再评估小程序可行的 VAD/降噪方案。

### C. `MpOutputAudioSession` 实现要点

Web 版是 `AudioContext + BufferSource`；小程序版可改为：

- `uni.createInnerAudioContext()` 播放。
- 如果后端下发的是 PCM chunk，需要先封装 WAV 或转为可播资源。
- 维持 `pause/resume/stop/pushAudioChunk` 同名接口，保证 `core.ts` 无感。

---

## 5. 建议同步调整的文件（第三优先级）

## 5.1 `src/platforms/web.ts`

目标：保留 Web 能力，同时避免和小程序代码耦合。

建议：

1. 继续只保留纯浏览器实现，不混入小程序分支。
2. 可把资源注入、VAD、Enhancer 拆成私有 helper，便于与 mp 实现对齐结构。

## 5.2 `src/action-handler-functions/client-operations.ts`

目标：处理上传文件跨端差异。

当前问题：直接使用 `FormData + fetch`，在小程序端通常不如 `uni.uploadFile` 稳定。

建议：

1. 增加一个平台无关上传适配器（例如 `src/platforms/upload.ts`）。
2. Web 走 `fetch + FormData`，小程序走 `uni.uploadFile`。
3. `client_upload_file` 只调用适配器，不直接关心平台。

## 5.3 `src/core.ts`

目标：尽量不改业务流程，只处理必要的初始化策略。

建议：

1. 保持 `createSession/open/close` 主流程不变。
2. 若小程序端初始化有异步准备（权限、编码器预热），可在 `initialize()` 前后补最小钩子。

---

## 6. 你当前 uniapp 页面侧需要配合的点

你在页面侧（如 `vite-uview-template/src/pages/index.vue`）已经通过 `createSession` 调用 SDK。为保证双端一致，建议：

1. 页面层不直接写 `window/document/AudioContext`。
2. 页面层只调用统一 API：`open/close/muted/onStateChange`。
3. 波形可视化、最近 60 秒音频等 Web 特性建议按 uni 条件编译分开。

---

## 7. 推荐改造顺序（可直接执行）

1. 改 `src/utils.ts`（先识别平台）。
2. 新建 `src/platforms/mp-weixin.ts`（先做最小可跑版：WebSocket + 输入输出基础能力）。
3. 改 `src/audio-session.ts` 和 `src/websocket.ts`（接入 mp 分支）。
4. 验证 `src/core.ts` 无需大改。
5. 改 `src/action-handler-functions/client-operations.ts`（上传跨端适配）。
6. 再补小程序端增强能力（VAD/Enhancer）而不是一开始就硬上。

---

## 8. 哪些文件先不要改

以下文件暂时可以不动或只做极小改动：

- `src/conversation.ts`
- `src/action-handler.ts`
- `src/action-handler-functions/messages.ts`
- `src/action-handler-functions/session.ts`
- `src/action-handler-functions/meta.ts`
- `src/action-handler-functions/latency.ts`
- `src/action-handler-functions/input.ts`

这些基本属于“协议分发和状态管理”，与平台 API 关系弱。

---

## 9. 一句话架构原则

**平台差异收敛在 `platforms/*` + 工厂层，业务编排留在 `core.ts`。**

这样你才能在 uniapp 下同时兼容 H5 与微信小程序，并且后续维护成本最低。

