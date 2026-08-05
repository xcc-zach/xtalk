# 白板工具方案

## 1. 目标与范围

对话过程中，AI 常常需要把不断演进的结构化内容摆在用户面前：计划、头脑风暴
板、清单或会议大纲。现有工具 UI 链路（`timer`、`codex`）渲染的是“状态卡”，
不适合承载 AI 在多个回合里逐步更新的结构化内容。

本方案新增一个内置**白板**工具：对话中 AI 通过工具调用，把便签内容的规范化
快照推送到与会话绑定的白板视图，白板实时重绘。

硬约束：

- 所有改动只发生在 `app/` 下；`frontend/` 与 `src/` 不动。
- 只使用已批准的 `xtalk.*` 公共 API（沿用 `app/scripts/verify_boundaries.py`
  白名单）。
- Phase 1 中白板对用户是只读视图；用户编辑留到后续阶段。

## 2. 关键设计决策

### 2.1 复用 Tool UI 只读观察通道（推荐，方案 A）

现有链路已经完成了最困难的部分：

AI 调用 `AsyncTool` → `wrap_tools_with_ui` 包装 → `ToolUIBroker.publish_emit`
→ 内存历史 / WebSocket → UI 轮询 `/app/api/tool-ui/events` → 沙箱
`ToolUIFrame` 通过 postMessage 收到 `tool_ui.emit`。

白板只需要两个向后兼容的小改动：

1. emit 事件增加可选的结构化 `payload` 字段。
2. 白板工具声明 `structured_payload = True`，包装器把 emit 的 JSON 内容解析为
   `payload`。

优点：改动面最小，安全边界不变（frame 仍只读、CSP 不变、无新端点），会话隔离
与历史回放天然免费获得。

代价：frame 高度上限限制白板尺寸；内存事件在 sidecar 重启后丢失；用户不能编辑。

### 2.2 专用白板服务（方案 B，后续阶段）

新增 `app/backend/whiteboard_store.py`（按会话持久化到 sqlite/JSON）、带认证的
`GET/POST /app/api/whiteboard/{session_id}` 端点（launch token + Origin），并
由 trusted UI 面板直接渲染大画布。可获得跨重启持久化、更大画布与用户编辑，代价
是新增端点和面板。

路线为“先 A 后 B”。

## 3. 数据契约

### 3.1 工具输入（LLM 看到的 schema）

`whiteboard_update` 接受操作列表（v1 子集）：

| op | 字段 | 说明 |
| --- | --- | --- |
| `set_title` | `title` | 设置标题 |
| `add_note` | `note {id?, text, color?}` | 新增便签；`id` 缺省时自动生成 |
| `update_note` | `id`、`text?`/`color?` | 更新既有便签 |
| `remove_note` | `id` | 删除便签 |
| `clear` | - | 清空白板 |

限制：便签 ≤ 200 条，单条文本 ≤ 2000 字符，单次调用 ops ≤ 50，序列化后 payload
≤ 256 KiB。

### 3.2 emit 内容（幂等快照）

每次 emit 的 `message` 都是完整规范化快照，同一对象同时作为 `payload` 附带：

```json
{
  "version": 1,
  "title": "本周计划",
  "revision": 3,
  "notes": [{"id": "n1", "text": "…", "color": "yellow"}],
  "updated_at": "…"
}
```

全量快照语义让渲染端无状态：任意一次 emit、回放或 history frame 都能独立完整
渲染。

## 4. 分文件改动清单

### 4.1 工具本体（新增）

- `app/resources/tools/whiteboard/xtalk_tool.json` — 显示名
  `{zh: "白板", en: "Whiteboard"}`，entrypoint `whiteboard_tool:create_tools`，
  `ui: {entrypoint: "ui/index.html", update_every_s: -1}`（不轮询 status，内容
  通过 emit 到达）。
- `app/resources/tools/whiteboard/whiteboard_tool.py` — pydantic op/快照模型、
  ops 应用、revision 计数、上限校验，以及一个 `AsyncTool`：`name =
  "whiteboard_update"`、`subscribe_by_default = False`、`structured_payload =
  True`。`emit_initial` 应用 ops 后返回 `Running(snapshot_json)`；
  `emit_updates` 立即 yield `Finished(snapshot_json)`。
- `app/resources/tools/whiteboard/ui/index.html` — 自包含沙箱 frame（见 4.3）。
- `app/resources/tools/builtin_tools.json` — 注册 `whiteboard`
  （`enabled_by_default: true`、`can_disable: true`）。

注意：必须用 `AsyncTool` 而不是 `SyncTool`，因为 UI 观察器只包装 `AsyncTool`
子类。

### 4.2 传输协议（小改，向后兼容）

- `app/backend/tool_ui.py`
  - `ToolUIBroker.publish_emit(..., payload=None)` 增加可选 keyword-only
    `dict` 字段。
  - `_wrap_async_tool` 在 `getattr(original, "structured_payload", False)` 且
    content 能解析为 JSON 对象时附带 `payload`；否则降级为无 payload（message
    不变）。
  - 新增 `MAX_TOOL_UI_EMIT_PAYLOAD_BYTES = 256 * 1024`，超限丢弃 payload、保留
    message。
  - 历史保留逻辑本就拷贝整个 dict，payload 自动随行。
- `app/ui/adapters/tool-ui-adapter.ts`
  - `ToolUIEmitEvent` 增加 `payload?: unknown`；`parseToolUIEvent` 接受
    `undefined` 或普通对象，并校验尺寸上限。
- `app/ui/tool-ui-frame.ts`
  - 协议零改动；补充注释说明 frame 通过 `event.payload` 读取结构化内容。
  - 可选 UX 调整：为白板放宽 live frame 高度上限（否则 frame 内部滚动）。

### 4.3 渲染（新增 frame）

`ui/index.html` 完全自包含：

- 从 `window.xtalkToolUI.emit` 事件渲染；优先 `event.payload`，回退解析
  `event.message`。
- v1 布局：标题 + 便签网格（CSS grid 卡片，支持颜色）、revision/数量徽标、
  空状态。
- 安全：所有便签文本一律用 `textContent` 渲染（绝不 `innerHTML`）；CSP 与
  沙箱不变；自动 `reportHeight`。
- 内置 zh/en 文案，跟随 `context.language`。

### 4.4 注册与挂载

无需改 `main.ts`/`index.html`：白板 emit 自动出现在现有 live tool panel 与
时间线工具行中；history 行渲染该次调用的最终快照（emit payload 无状态，天然
支持）。

## 5. 测试计划

- 新增 `app/tests/unit/test_whiteboard_tool.py`：ops 应用矩阵、revision 递增、
  自动 id、上限与非法输入、快照幂等。
- 扩展 `app/tests/unit/test_tool_ui.py`：`structured_payload` 解析、payload 随
  emit/history 保留、非法 JSON 降级、无 payload 的旧事件仍然有效。
- 扩展 `app/tests/unit/test_runtime.py`：`/app/api/tool-ui/events` 快照包含
  payload。
- 门槛：`python scripts/verify_boundaries.py`、`npm run check`、
  `python -m pytest`。
- 手工验收：让 AI 跨回合新增/修改/删除便签，确认 live 面板实时更新、会话之间
  隔离、history 行渲染正确的最终快照。

## 6. 安全与边界

- 一切改动在 `app/` 内；只使用批准的 `xtalk.models.agents.tools` API。
- iframe 保持 `allow-scripts` only、CSP `connect-src 'none'`、点击/提交禁用
  （v1 只读）。
- payload 与便签文本均有上限；文本用 `textContent` 渲染防 XSS。
- 任何新端点（Phase 2）都必须走 launch token + Origin 认证。

## 7. 分阶段路线

- **Phase 1（本方案）：** 工具 + 结构化 payload + frame，端到端可用。
- **Phase 2：** `whiteboard_store.py`、按会话持久化、带认证的
  `GET /app/api/whiteboard/{session_id}`、更大的专属面板，使白板跨重启保留并
  获得更多空间。
- **Phase 3（可选）：** 用户编辑。推荐改为 trusted UI 直接渲染画布（不走
  iframe）；或在仅针对该受信任内置放宽的沙箱上增加受认证写端点，需单独安全
  评审。

## 8. 验收标准

- 代码与测试全部位于 `app/` 下；`verify_boundaries.py` 通过。
- AI 可跨回合增量更新同一白板；不同会话之间互不影响。
- 便签文本永不按 HTML 处理。
- live 与 history 渲染一致。
