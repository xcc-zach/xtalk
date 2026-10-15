[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / Session

# 接口： Session

定义于： session/types.ts:40

前端入口导出的公共会话控制器。

## 属性

### muted

> **muted**: `boolean`

定义于： session/types.ts:80

麦克风采集当前是否静音。

***

### state

> `readonly` **state**: `object`

定义于： session/types.ts:58

当前对话状态快照。

#### caption

> **caption**: `string`

#### connectionState

> **connectionState**: `"connected"` \| `"reconnecting"` \| `"disconnected"`

#### generatedResponses

> **generatedResponses**: `Record`&lt;`string`, \{ `final`: `boolean`; `text`: `string`; \}&gt;

#### latency

> **latency**: `object`

##### latency.asr?

> `optional` **asr?**: `number`

##### latency.llmFirstToken?

> `optional` **llmFirstToken?**: `number`

##### latency.llmSentence?

> `optional` **llmSentence?**: `number`

##### latency.network?

> `optional` **network?**: `number`

##### latency.ttsFirstChunk?

> `optional` **ttsFirstChunk?**: `number`

#### messages

> **messages**: `ConversationMessage`[]

#### retrieval

> **retrieval**: `string`

#### sessionId

> **sessionId**: `string` \| `null`

#### streamState

> **streamState**: `"idle"` \| `"listening"` \| `"processing"` \| `"speaking"`

#### thought

> **thought**: `string`

#### tool\_call

> **tool\_call**: `object`

##### tool\_call.args

> **args**: `Record`&lt;`string`, `any`&gt;

##### tool\_call.name

> **name**: `string`

## 方法

### changeVoice()

> **changeVoice**(`voiceName`): `Promise`&lt;`void`&gt;

定义于： session/types.ts:86

请求修改后续助手语音合成使用的音色。

#### 参数

##### voiceName

`string`

目标音色标识。

#### 返回值

`Promise`&lt;`void`&gt;

***

### close()

> **close**(): `Promise`&lt;`void`&gt;

定义于： session/types.ts:48

关闭当前连接并释放音频资源。

#### 返回值

`Promise`&lt;`void`&gt;

***

### onFullAudioChunk()

> **onFullAudioChunk**(`callback`): `void`

定义于： session/types.ts:76

注册合并后的全双工 PCM 音频块回调。

#### 参数

##### callback

`AudioChunkCallback`

完整音频监听器。

#### 返回值

`void`

***

### onInputAudioChunk()

> **onInputAudioChunk**(`callback`): `void`

定义于： session/types.ts:64

注册麦克风输入 PCM 音频块回调。

#### 参数

##### callback

`AudioChunkCallback`

输入音频监听器。

#### 返回值

`void`

***

### onOutputAudioChunk()

> **onOutputAudioChunk**(`callback`): `void`

定义于： session/types.ts:70

注册扬声器输出 PCM 音频块回调。

#### 参数

##### callback

`AudioChunkCallback`

输出音频监听器。

#### 返回值

`void`

***

### onStateChange()

> **onStateChange**(`callback`): `void`

定义于： session/types.ts:54

注册对话状态变更回调。

#### 参数

##### callback

(`state`) => `void`

状态变更监听器。

#### 返回值

`void`

***

### open()

> **open**(): `Promise`&lt;`void`&gt;

定义于： session/types.ts:44

打开 Realtime 连接及音频资源。

#### 返回值

`Promise`&lt;`void`&gt;

***

### restoreHistory()

> **restoreHistory**(`messages`): `Promise`&lt;`void`&gt;

定义于： session/types.ts:97

注入外部文本历史，不触发回复。

#### 参数

##### messages

`object`[]

#### 返回值

`Promise`&lt;`void`&gt;

***

### sendText()

> **sendText**(`text`): `Promise`&lt;`void`&gt;

定义于： session/types.ts:95

通过已连接的 Realtime 会话提交完整文本轮次。

收到匹配的 `conversation.item.done` 确认后，Promise 完成。
校验失败或连接断开时，Promise 拒绝。

#### 参数

##### text

`string`

用户提交的下一轮文本。

#### 返回值

`Promise`&lt;`void`&gt;
