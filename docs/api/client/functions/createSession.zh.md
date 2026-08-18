[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / createSession

# 函数: createSession()

> **createSession**(`websocketURL`, `config?`): [`Session`](../interfaces/Session.zh.md)

定义于: `session/create.ts:53`

创建一个绑定到指定 WebSocket 端点的会话客户端。

返回的会话对象负责协调整个鉴权、运行时音频流、消息状态同步以及持久化会话恢复流程。

## 参数

### websocketURL

`string` \| `URL`

用于建立实时会话的 WebSocket 端点。

### config?

[`SessionConfig`](../interfaces/SessionConfig.zh.md) = `{}`

可选的会话配置覆盖项。

## 返回

[`Session`](../interfaces/Session.zh.md)

用于打开、关闭并与 X-Talk 交互的会话控制器。
