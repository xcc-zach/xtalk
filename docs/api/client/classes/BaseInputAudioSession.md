[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / BaseInputAudioSession

# Abstract Class: BaseInputAudioSession

Defined in: bases/audio-session.ts:33

## Constructors

### Constructor

> **new BaseInputAudioSession**(): `BaseInputAudioSession`

#### Returns

`BaseInputAudioSession`

## Accessors

### muted

#### Get Signature

> **get** `abstract` **muted**(): `boolean`

Defined in: bases/audio-session.ts:36

##### Returns

`boolean`

#### Set Signature

> **set** `abstract` **muted**(`value`): `void`

Defined in: bases/audio-session.ts:37

##### Parameters

###### value

`boolean`

##### Returns

`void`

## Methods

### close()

> `abstract` **close**(): `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:35

#### Returns

`Promise`&lt;`void`&gt;

***

### frameCallback()

> `protected` **frameCallback**(`_pcmChunkInt16`): `void` \| `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:48

#### Parameters

##### \_pcmChunkInt16

`ArrayBuffer`

#### Returns

`void` \| `Promise`&lt;`void`&gt;

***

### onFrame()

> **onFrame**(`callback`): `void`

Defined in: bases/audio-session.ts:39

#### Parameters

##### callback

(`pcmChunkInt16`) => `void` \| `Promise`&lt;`void`&gt;

#### Returns

`void`

***

### onSpeechEnd()

> **onSpeechEnd**(`callback`): `void`

Defined in: bases/audio-session.ts:45

#### Parameters

##### callback

() => `void` \| `Promise`&lt;`void`&gt;

#### Returns

`void`

***

### onSpeechStart()

> **onSpeechStart**(`callback`): `void`

Defined in: bases/audio-session.ts:42

#### Parameters

##### callback

() => `void` \| `Promise`&lt;`void`&gt;

#### Returns

`void`

***

### open()

> `abstract` **open**(): `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:34

#### Returns

`Promise`&lt;`void`&gt;

***

### speechEndCallback()

> `protected` **speechEndCallback**(): `void` \| `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:60

Used only when VAD enabled

#### Returns

`void` \| `Promise`&lt;`void`&gt;

***

### speechStartCallback()

> `protected` **speechStartCallback**(): `void` \| `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:54

Used only when VAD enabled

#### Returns

`void` \| `Promise`&lt;`void`&gt;
