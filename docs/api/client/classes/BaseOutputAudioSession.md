[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / BaseOutputAudioSession

# Abstract Class: BaseOutputAudioSession

Defined in: bases/audio-session.ts:73

## Constructors

### Constructor

> **new BaseOutputAudioSession**(): `BaseOutputAudioSession`

#### Returns

`BaseOutputAudioSession`

## Methods

### allChunksPlayedCallback()

> `protected` **allChunksPlayedCallback**(`_responseId`): `void` \| `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:105

#### Parameters

##### \_responseId

`string`

#### Returns

`void` \| `Promise`&lt;`void`&gt;

***

### chunkPlayedCallback()

> `protected` **chunkPlayedCallback**(`_responseId`, `_pcmChunkInt16`): `void` \| `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:102

#### Parameters

##### \_responseId

`string`

##### \_pcmChunkInt16

`ArrayBuffer`

#### Returns

`void` \| `Promise`&lt;`void`&gt;

***

### chunkStartedCallback()

> `protected` **chunkStartedCallback**(`_responseId`, `_pcmChunkInt16`): `void` \| `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:99

#### Parameters

##### \_responseId

`string`

##### \_pcmChunkInt16

`ArrayBuffer`

#### Returns

`void` \| `Promise`&lt;`void`&gt;

***

### close()

> `abstract` **close**(): `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:75

#### Returns

`Promise`&lt;`void`&gt;

***

### notifyTTSFinished()

> `abstract` **notifyTTSFinished**(`responseId`): `void` \| `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:88

Marks that the server has finished producing TTS audio for the current turn.
Implementations should combine this signal with local playback state before
reporting that playback is fully finished.

#### Parameters

##### responseId

`string`

#### Returns

`void` \| `Promise`&lt;`void`&gt;

***

### onAllChunksPlayed()

> **onAllChunksPlayed**(`callback`): `void`

Defined in: bases/audio-session.ts:96

#### Parameters

##### callback

(`responseId`) => `void` \| `Promise`&lt;`void`&gt;

#### Returns

`void`

***

### onChunkPlayed()

> **onChunkPlayed**(`callback`): `void`

Defined in: bases/audio-session.ts:93

#### Parameters

##### callback

(`responseId`, `pcmChunkInt16`) => `void` \| `Promise`&lt;`void`&gt;

#### Returns

`void`

***

### onChunkStarted()

> **onChunkStarted**(`callback`): `void`

Defined in: bases/audio-session.ts:90

#### Parameters

##### callback

(`responseId`, `pcmChunkInt16`) => `void` \| `Promise`&lt;`void`&gt;

#### Returns

`void`

***

### open()

> `abstract` **open**(): `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:74

#### Returns

`Promise`&lt;`void`&gt;

***

### pause()

> `abstract` **pause**(): `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:76

#### Returns

`Promise`&lt;`void`&gt;

***

### pushAudioChunk()

> `abstract` **pushAudioChunk**(`pcmChunkInt16`): `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:82

#### Parameters

##### pcmChunkInt16

`ArrayBuffer`

#### Returns

`Promise`&lt;`void`&gt;

***

### resume()

> `abstract` **resume**(): `Promise`&lt;`void`&gt;

Defined in: bases/audio-session.ts:77

#### Returns

`Promise`&lt;`void`&gt;

***

### startTTS()

> `abstract` **startTTS**(`responseId`): `void`

Defined in: bases/audio-session.ts:79

Open local playback state for one server response.

#### Parameters

##### responseId

`string`

#### Returns

`void`

***

### stop()

> `abstract` **stop**(`responseId?`): `Promise`&lt;[`OutputAudioStopResult`](../interfaces/OutputAudioStopResult.md)&gt;

Defined in: bases/audio-session.ts:81

Stop only the identified response, or all playback during shutdown.

#### Parameters

##### responseId?

`string`

#### Returns

`Promise`&lt;[`OutputAudioStopResult`](../interfaces/OutputAudioStopResult.md)&gt;
