[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / Session

# Interface: Session

Defined in: session/types.ts:40

Public session controller exposed by the frontend entrypoint.

## Properties

### muted

> **muted**: `boolean`

Defined in: session/types.ts:80

Whether microphone capture is currently muted.

***

### state

> `readonly` **state**: `object`

Defined in: session/types.ts:58

Current conversation state snapshot.

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

## Methods

### changeVoice()

> **changeVoice**(`voiceName`): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:86

Requests a voice change for subsequent assistant synthesis.

#### Parameters

##### voiceName

`string`

Target voice identifier.

#### Returns

`Promise`&lt;`void`&gt;

***

### close()

> **close**(): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:48

Closes the active runtime connection and audio resources.

#### Returns

`Promise`&lt;`void`&gt;

***

### onFullAudioChunk()

> **onFullAudioChunk**(`callback`): `void`

Defined in: session/types.ts:76

Registers a callback for merged full-duplex PCM chunks.

#### Parameters

##### callback

`AudioChunkCallback`

Full audio listener.

#### Returns

`void`

***

### onInputAudioChunk()

> **onInputAudioChunk**(`callback`): `void`

Defined in: session/types.ts:64

Registers a callback for microphone input PCM chunks.

#### Parameters

##### callback

`AudioChunkCallback`

Input audio listener.

#### Returns

`void`

***

### onOutputAudioChunk()

> **onOutputAudioChunk**(`callback`): `void`

Defined in: session/types.ts:70

Registers a callback for speaker output PCM chunks.

#### Parameters

##### callback

`AudioChunkCallback`

Output audio listener.

#### Returns

`void`

***

### onStateChange()

> **onStateChange**(`callback`): `void`

Defined in: session/types.ts:54

Registers a callback that runs whenever the conversation state changes.

#### Parameters

##### callback

(`state`) => `void`

State change listener.

#### Returns

`void`

***

### open()

> **open**(): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:44

Opens the Realtime connection and its audio resources.

#### Returns

`Promise`&lt;`void`&gt;

***

### restoreHistory()

> **restoreHistory**(`messages`): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:97

Injects external text history without generating a reply.

#### Parameters

##### messages

`object`[]

#### Returns

`Promise`&lt;`void`&gt;

***

### sendText()

> **sendText**(`text`): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:95

Submits a finalized text turn through the connected realtime session.

The promise resolves after the matching `conversation.item.done` receipt.
Validation failures and disconnection reject the promise.

#### Parameters

##### text

`string`

User-authored text for the next turn.

#### Returns

`Promise`&lt;`void`&gt;
