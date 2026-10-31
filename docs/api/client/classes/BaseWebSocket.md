[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / BaseWebSocket

# Abstract Class: BaseWebSocket

Defined in: bases/websocket.ts:15

## Constructors

### Constructor

> **new BaseWebSocket**(): `BaseWebSocket`

#### Returns

`BaseWebSocket`

## Methods

### addEventListener()

#### Call Signature

> `abstract` **addEventListener**(`type`, `listener`): `void`

Defined in: bases/websocket.ts:22

##### Parameters

###### type

`"open"` \| `"error"`

###### listener

() => `any`

##### Returns

`void`

#### Call Signature

> `abstract` **addEventListener**(`type`, `listener`): `void`

Defined in: bases/websocket.ts:24

##### Parameters

###### type

`"message"`

###### listener

(`evt`) => `any`

##### Returns

`void`

#### Call Signature

> `abstract` **addEventListener**(`type`, `listener`): `void`

Defined in: bases/websocket.ts:26

##### Parameters

###### type

`"close"`

###### listener

(`evt`) => `any`

##### Returns

`void`

***

### close()

> `abstract` **close**(): `void`

Defined in: bases/websocket.ts:20

#### Returns

`void`

***

### ready()

> `abstract` **ready**(): `boolean`

Defined in: bases/websocket.ts:16

#### Returns

`boolean`

***

### send()

> `abstract` **send**(`data`): `void`

Defined in: bases/websocket.ts:18

#### Parameters

##### data

`string` \| `ArrayBuffer`

#### Returns

`void`

***

### sendAudioChunk()

> **sendAudioChunk**(`pcm_chunk_int16`): `void`

Defined in: bases/websocket.ts:32

#### Parameters

##### pcm\_chunk\_int16

`ArrayBuffer`

#### Returns

`void`

***

### sendJson()

> **sendJson**(`data`): `void`

Defined in: bases/websocket.ts:28

#### Parameters

##### data

`object`

#### Returns

`void`
