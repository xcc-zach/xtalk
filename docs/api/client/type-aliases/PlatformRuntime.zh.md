[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / PlatformRuntime

# 类型别名： PlatformRuntime

> **PlatformRuntime** = `object`

定义于： platforms/index.ts:39

提供会话所需全部平台相关资源的工厂集合。

通过 [registerPlatformRuntime](../functions/registerPlatformRuntime.zh.md) 注册自定义运行时，即可在浏览器之外（例如自动化测试中）运行客户端。

## 方法

### createDeferredTaskScheduler()

> **createDeferredTaskScheduler**(): [`BaseDeferredTaskScheduler`](../classes/BaseDeferredTaskScheduler.zh.md)

定义于： platforms/index.ts:49

创建用于延迟执行观察者回调的调度器。

#### 返回值

[`BaseDeferredTaskScheduler`](../classes/BaseDeferredTaskScheduler.zh.md)

***

### createEncoding()

> **createEncoding**(): [`BaseEncoding`](../classes/BaseEncoding.zh.md)

定义于： platforms/index.ts:51

创建 Base64 与 PCM 编码器。

#### 返回值

[`BaseEncoding`](../classes/BaseEncoding.zh.md)

***

### createInputAudioSession()

> **createInputAudioSession**(`config`): [`BaseInputAudioSession`](../classes/BaseInputAudioSession.zh.md)

定义于： platforms/index.ts:41

创建麦克风或桥接式音频输入源。

#### 参数

##### config

[`InputAudioSessionConfig`](../interfaces/InputAudioSessionConfig.zh.md)

#### 返回值

[`BaseInputAudioSession`](../classes/BaseInputAudioSession.zh.md)

***

### createOutputAudioSession()

> **createOutputAudioSession**(`config`): [`BaseOutputAudioSession`](../classes/BaseOutputAudioSession.zh.md)

定义于： platforms/index.ts:43

创建扬声器播放输出。

#### 参数

##### config

[`OutputAudioSessionConfig`](../interfaces/OutputAudioSessionConfig.zh.md)

#### 返回值

[`BaseOutputAudioSession`](../classes/BaseOutputAudioSession.zh.md)

***

### createWebSocket()

> **createWebSocket**(`url`, `protocols?`): [`BaseWebSocket`](../classes/BaseWebSocket.zh.md)

定义于： platforms/index.ts:45

创建 Realtime 连接所用的传输层。

#### 参数

##### url

`string` \| `URL`

##### protocols?

`string` \| `string`[]

#### 返回值

[`BaseWebSocket`](../classes/BaseWebSocket.zh.md)

***

### delay()

> **delay**(`milliseconds`): `Promise`&lt;`void`&gt;

定义于： platforms/index.ts:47

在给定延迟后完成。

#### 参数

##### milliseconds

`number`

#### 返回值

`Promise`&lt;`void`&gt;
