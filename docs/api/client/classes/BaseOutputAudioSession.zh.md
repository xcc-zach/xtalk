[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / BaseOutputAudioSession

# 抽象类： BaseOutputAudioSession

定义于： bases/audio-session.ts:73

## 构造函数

### 构造函数

> **new BaseOutputAudioSession**(): `BaseOutputAudioSession`

#### 返回值

`BaseOutputAudioSession`

## 方法

### allChunksPlayedCallback()

> `protected` **allChunksPlayedCallback**(`_responseId`): `void` \| `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:105

#### 参数

##### \_responseId

`string`

#### 返回值

`void` \| `Promise`&lt;`void`&gt;

***

### chunkPlayedCallback()

> `protected` **chunkPlayedCallback**(`_responseId`, `_pcmChunkInt16`): `void` \| `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:102

#### 参数

##### \_responseId

`string`

##### \_pcmChunkInt16

`ArrayBuffer`

#### 返回值

`void` \| `Promise`&lt;`void`&gt;

***

### chunkStartedCallback()

> `protected` **chunkStartedCallback**(`_responseId`, `_pcmChunkInt16`): `void` \| `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:99

#### 参数

##### \_responseId

`string`

##### \_pcmChunkInt16

`ArrayBuffer`

#### 返回值

`void` \| `Promise`&lt;`void`&gt;

***

### close()

> `abstract` **close**(): `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:75

#### 返回值

`Promise`&lt;`void`&gt;

***

### notifyTTSFinished()

> `abstract` **notifyTTSFinished**(`responseId`): `void` \| `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:88

标记服务端已完成本轮 TTS 音频的生成。实现应结合本地播放状态后再报告播放完全结束。

#### 参数

##### responseId

`string`

#### 返回值

`void` \| `Promise`&lt;`void`&gt;

***

### onAllChunksPlayed()

> **onAllChunksPlayed**(`callback`): `void`

定义于： bases/audio-session.ts:96

#### 参数

##### callback

(`responseId`) => `void` \| `Promise`&lt;`void`&gt;

#### 返回值

`void`

***

### onChunkPlayed()

> **onChunkPlayed**(`callback`): `void`

定义于： bases/audio-session.ts:93

#### 参数

##### callback

(`responseId`, `pcmChunkInt16`) => `void` \| `Promise`&lt;`void`&gt;

#### 返回值

`void`

***

### onChunkStarted()

> **onChunkStarted**(`callback`): `void`

定义于： bases/audio-session.ts:90

#### 参数

##### callback

(`responseId`, `pcmChunkInt16`) => `void` \| `Promise`&lt;`void`&gt;

#### 返回值

`void`

***

### open()

> `abstract` **open**(): `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:74

#### 返回值

`Promise`&lt;`void`&gt;

***

### pause()

> `abstract` **pause**(): `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:76

#### 返回值

`Promise`&lt;`void`&gt;

***

### pushAudioChunk()

> `abstract` **pushAudioChunk**(`pcmChunkInt16`): `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:82

#### 参数

##### pcmChunkInt16

`ArrayBuffer`

#### 返回值

`Promise`&lt;`void`&gt;

***

### resume()

> `abstract` **resume**(): `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:77

#### 返回值

`Promise`&lt;`void`&gt;

***

### startTTS()

> `abstract` **startTTS**(`responseId`): `void`

定义于： bases/audio-session.ts:79

为一个服务端回复打开本地播放状态。

#### 参数

##### responseId

`string`

#### 返回值

`void`

***

### stop()

> `abstract` **stop**(`responseId?`): `Promise`&lt;[`OutputAudioStopResult`](../interfaces/OutputAudioStopResult.zh.md)&gt;

定义于： bases/audio-session.ts:81

只停止指定回复的播放；关闭时停止全部播放。

#### 参数

##### responseId?

`string`

#### 返回值

`Promise`&lt;[`OutputAudioStopResult`](../interfaces/OutputAudioStopResult.zh.md)&gt;
