[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / createSession

# 函数： createSession()

> **createSession**(`websocketURL`, `config?`): [`Session`](../interfaces/Session.zh.md)

定义于： session/create.ts:7

创建仅负责 Realtime 对话的客户端，不执行业务 HTTP 请求。

## 参数

### websocketURL

`string` \| `URL`

### config?

[`SessionConfig`](../interfaces/SessionConfig.zh.md) = `{}`

## 返回值

[`Session`](../interfaces/Session.zh.md)
