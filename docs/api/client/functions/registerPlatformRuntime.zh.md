[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / registerPlatformRuntime

# 函数： registerPlatformRuntime()

> **registerPlatformRuntime**(`runtime`): `void`

定义于： platforms/index.ts:64

使用显式指定的平台运行时，替代浏览器环境探测。

请在创建会话之前调用。之后再次调用会替换后续创建会话所用的运行时；已创建的会话继续持有原有资源。

## 参数

### runtime

[`PlatformRuntime`](../type-aliases/PlatformRuntime.zh.md)

提供全部会话资源的平台运行时。

## 返回值

`void`
