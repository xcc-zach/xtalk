[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / BaseWebSocket

# 抽象类： BaseWebSocket

定义于： bases/websocket.ts:15

## 构造函数

### 构造函数

> **new BaseWebSocket**(): `BaseWebSocket`

#### 返回值

`BaseWebSocket`

## 方法

### addEventListener()

#### 调用签名

> `abstract` **addEventListener**(`type`, `listener`): `void`

定义于： bases/websocket.ts:22

##### 参数

###### type

`"open"` \| `"error"`

###### listener

() => `any`

##### 返回值

`void`

#### 调用签名

> `abstract` **addEventListener**(`type`, `listener`): `void`

定义于： bases/websocket.ts:24

##### 参数

###### type

`"message"`

###### listener

(`evt`) => `any`

##### 返回值

`void`

#### 调用签名

> `abstract` **addEventListener**(`type`, `listener`): `void`

定义于： bases/websocket.ts:26

##### 参数

###### type

`"close"`

###### listener

(`evt`) => `any`

##### 返回值

`void`

***

### close()

> `abstract` **close**(): `void`

定义于： bases/websocket.ts:20

#### 返回值

`void`

***

### ready()

> `abstract` **ready**(): `boolean`

定义于： bases/websocket.ts:16

#### 返回值

`boolean`

***

### send()

> `abstract` **send**(`data`): `void`

定义于： bases/websocket.ts:18

#### 参数

##### data

`string` \| `ArrayBuffer`

#### 返回值

`void`

***

### sendAudioChunk()

> **sendAudioChunk**(`pcm_chunk_int16`): `void`

定义于： bases/websocket.ts:32

#### 参数

##### pcm\_chunk\_int16

`ArrayBuffer`

#### 返回值

`void`

***

### sendJson()

> **sendJson**(`data`): `void`

定义于： bases/websocket.ts:28

#### 参数

##### data

`object`

#### 返回值

`void`
