# 文本白板方案

## 1. 目标与范围

桌面白板为**每场对话维护独立的 Markdown 文本板**。AI 通过四个文本工具按会话
读写各自的文档；当前对话的白板以 Markdown 形式渲染在**独立白板窗口**中，窗口
保持在 XTalk 主窗口之上，第一次被调用时自动弹出，切换对话时自动跟随显示，同时
开启对话键右侧增加一个白板按钮用于显示/隐藏窗口，且显示/隐藏状态在对话重新
加载后仍然保持。

硬约束：

- 所有改动只发生在 `app/` 下；`frontend/` 与 `src/` 不动。
- 只使用已批准的 `xtalk.*` 公共 API（沿用 `app/scripts/verify_boundaries.py`
  白名单）。
- 窗口对用户只读；编辑仍由模型驱动。

## 2. 关键设计决策

### 2.1 按会话隔离的 store

`app/backend/whiteboard_store.py` 按会话维护独立文档：

- 注册表把会话 id 映射到各自的 `text`、`revision`、`updated_at`，每个会话
  持久化为工具数据目录下的 `whiteboards/<session>.json`，sidecar 重启后内容
  仍在，且不同会话之间互不干扰。
- `add_text` 用单个换行连接文本块；`delete_text` 与 `update_text` 删除或替换
  所有精确匹配，并规整残留的换行。
- 工具 UI 包装器把当前绑定的会话写入工具调用状态，只读 sidecar 端点也读取
  同一个按会话隔离的 store，窗口内容不依赖 emit 通道。

### 2.2 独立窗口 + 停靠按钮

- Tauri `WebviewWindow`（label 为 `whiteboard`）加载 `whiteboard.html`，作为
  主窗口的子窗口创建以保持在其之上，轮询
  `GET /app/api/whiteboard?session_id=...`（launch token + Origin），用小型
  无依赖渲染器渲染返回的 Markdown。窗口从主窗口持久化的活动会话键读取当前
  对话，并跟随会话切换。
- 窗口内不再显示标题与版本徽标；原生窗口标题通过 i18n 使用
  `t("whiteboard.windowTitle")`。
- 主窗口在开启对话键右侧新增白板按钮；点击切换窗口显示/隐藏，可见性写入
  `localStorage`，对话重新加载时恢复。
- 关闭窗口只隐藏（webview 保留），Rust 侧发出 `whiteboard-window-hidden`
  事件，让停靠按钮状态保持同步。
- 切换到其他对话（包括新建对话）时收起白板窗口（并记住该选择），上一场对话的
  白板不会跟着进入新对话；新对话中第一次白板工具 emit 会再次自动弹出。
- 白板工具第一次 emit 时自动弹出窗口；之后的 emit 只在窗口已打开时保持现状。

## 3. 工具契约

四个工具共用一个描述，暂定为：

> 在需要教学、演示的场景考虑调用该工具。
> Consider using this tool when teaching or demonstrating.

### 3.1 工具 schema

| 工具 | 输入 | 行为 |
| --- | --- | --- |
| `fetch_text` | - | 返回白板全文与 revision |
| `add_text` | `text` | 在末尾追加一段 Markdown 文本 |
| `delete_text` | `text` | 删除所有精确匹配；不存在时报错 |
| `update_text` | `from`, `to` | 把所有精确匹配的 `from` 替换为 `to`；不存在时报错 |

限制：单次操作 ≤ 50 000 字符，emit payload ≤ 256 KiB。每次调用都返回完整
规范化快照 `{action, success, text, revision, message}`。

### 3.2 API 快照

`GET /app/api/whiteboard?session_id=...` 返回对应会话与工具所见一致的文档：

```json
{
  "version": 1,
  "text": "# 本周计划\n- 目标",
  "revision": 3,
  "updated_at": "…"
}
```

## 4. 分文件改动清单

- `app/backend/whiteboard_store.py`（新增）—— 全局 store、JSON 持久化、
  单例辅助函数。
- `app/backend/whiteboard_tool.py` —— 供单测导入的镜像副本。
- `app/resources/tools/whiteboard/whiteboard_tool.py` —— 四个共享上述描述的
  `AsyncTool` 类，`structured_payload = True`。
- `app/resources/tools/whiteboard/ui/index.html` —— 极简沙箱兜底页，渲染文本
  快照（窗口才是主界面）。
- `app/backend/runtime.py` —— 新增 `GET /app/api/whiteboard`，与其他
  `/app/api` 路由一样受 launch token 保护；`session_id` 查询参数选择会话。
- `app/src-tauri/src/whiteboard.rs`（新增）+ `lib.rs` —— 显示/隐藏/设置/查询
  命令、窗口创建（挂到主窗口之下）、关闭请求拦截。
- `app/src-tauri/build.rs` + `capabilities/main.json` + 新增
  `capabilities/whiteboard.json` + `tauri.conf.json` —— 四个窗口命令与白板
  窗口轮询所需的 ACL 权限。
- `app/backend/tool_ui.py` —— 把当前绑定的会话写入工具调用状态，让白板工具
  访问正确的会话 store。
- `app/ui/whiteboard.html` + `whiteboard.ts`（新增）—— 窗口页面，轮询端点并
  渲染 Markdown（窗口隐藏时暂停）。
- `app/ui/markdown.ts`（新增）—— 安全、无依赖的 Markdown 渲染器。
- `app/ui/whiteboard-window.ts`（新增）—— Tauri invoke/事件封装与可见性偏好。
- `app/ui/index.html` + `main.ts` + `styles.css` —— 白板停靠按钮、首次 emit
  自动弹出、隐藏事件同步、可见性恢复。
- `app/ui/i18n.ts` —— 白板文案（中英文）。
- `app/vite.config.ts` —— 增加 `whiteboard.html` 多页入口。

## 5. 测试计划

- 重写 `app/tests/unit/test_whiteboard_tool.py`：追加/删除/更新语义、按会话
  隔离、store 重启后的 JSON 持久化、`from` 别名校验、限制、结构化 emit
  payload。
- `app/tests/unit/test_tool_registry.py` 期望四个新工具名。
- `app/tests/unit/test_runtime.py` 覆盖白板端点鉴权（无 launch token 返回
  401）。
- 门禁：`python -m pytest`、`npm run check`、`npm run build`、`cargo check`。
- 手工验收：让 AI 跨轮次增删改文本；第一次调用自动弹出窗口，按钮可切换，
  Markdown 正常渲染，对话重新加载后可见性与内容仍然保持。

## 6. 安全与边界

- 所有改动在 `app/` 内；只使用已批准的 `xtalk.models.agents.tools` API。
- Markdown 渲染器先转义所有原始字符，再仅回填已转义的占位内容；链接只允许
  `http(s)`，白板文本绝不当 HTML 处理。
- 白板端点受 launch token 与 Origin 校验；窗口是受信 App 页面，不是非受信
  工具 frame。
- 单次操作与 emit payload 均有大小上限。

## 7. 路线图

- **本方案已完成：** 全局文本 store、四个工具、API 端点、独立窗口、停靠按钮、
  Markdown 渲染、持久化、切换对话时收起白板。
- **后续：** 用户编辑白板（需要写端点并做安全评审）；若单一全局板不再够用，
  再考虑按会话分板。
