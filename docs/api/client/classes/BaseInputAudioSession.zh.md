[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / BaseInputAudioSession

# 抽象类： BaseInputAudioSession

定义于： bases/audio-session.ts:33

## 构造函数

### 构造函数

> **new BaseInputAudioSession**(): `BaseInputAudioSession`

#### 返回值

`BaseInputAudioSession`

## 访问器

### muted

#### 读取签名

> **get** `abstract` **muted**(): `boolean`

定义于： bases/audio-session.ts:36

##### 返回值

`boolean`

#### 写入签名

> **set** `abstract` **muted**(`value`): `void`

定义于： bases/audio-session.ts:37

##### 参数

###### value

`boolean`

##### 返回值

`void`

## 方法

### close()

> `abstract` **close**(): `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:35

#### 返回值

`Promise`&lt;`void`&gt;

***

### frameCallback()

> `protected` **frameCallback**(`_pcmChunkInt16`): `void` \| `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:48

#### 参数

##### \_pcmChunkInt16

`ArrayBuffer`

#### 返回值

`void` \| `Promise`&lt;`void`&gt;

***

### onFrame()

> **onFrame**(`callback`): `void`

定义于： bases/audio-session.ts:39

#### 参数

##### callback

(`pcmChunkInt16`) => `void` \| `Promise`&lt;`void`&gt;

#### 返回值

`void`

***

### onSpeechEnd()

> **onSpeechEnd**(`callback`): `void`

定义于： bases/audio-session.ts:45

#### 参数

##### callback

() => `void` \| `Promise`&lt;`void`&gt;

#### 返回值

`void`

***

### onSpeechStart()

> **onSpeechStart**(`callback`): `void`

定义于： bases/audio-session.ts:42

#### 参数

##### callback

() => `void` \| `Promise`&lt;`void`&gt;

#### 返回值

`void`

***

### open()

> `abstract` **open**(): `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:34

#### 返回值

`Promise`&lt;`void`&gt;

***

### speechEndCallback()

> `protected` **speechEndCallback**(): `void` \| `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:60

仅在启用 VAD 时使用

#### 返回值

`void` \| `Promise`&lt;`void`&gt;

***

### speechStartCallback()

> `protected` **speechStartCallback**(): `void` \| `Promise`&lt;`void`&gt;

定义于： bases/audio-session.ts:54

仅在启用 VAD 时使用

#### 返回值

`void` \| `Promise`&lt;`void`&gt;
