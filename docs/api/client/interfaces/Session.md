[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / Session

# Interface: Session

Defined in: session/types.ts:86

Public session controller exposed by the frontend entrypoint.

## Properties

### muted

> **muted**: `boolean`

Defined in: session/types.ts:126

Whether microphone capture is currently muted.

***

### state

> `readonly` **state**: `object`

Defined in: session/types.ts:104

Current conversation state snapshot.

#### caption

> **caption**: `string`

#### connectionState

> **connectionState**: `"connected"` \| `"reconnecting"` \| `"disconnected"`

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

#### user

> **user**: `ConversationUser` \| `null`

## Methods

### changeVoice()

> **changeVoice**(`voiceName`): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:132

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

Defined in: session/types.ts:94

Closes the active runtime connection and audio resources.

#### Returns

`Promise`&lt;`void`&gt;

***

### getSessions()

> **getSessions**(): `Promise`&lt;`SessionSummary`[]&gt;

Defined in: session/types.ts:152

Fetches available persisted sessions for the current user.

#### Returns

`Promise`&lt;`SessionSummary`[]&gt;

***

### onFullAudioChunk()

> **onFullAudioChunk**(`callback`): `void`

Defined in: session/types.ts:122

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

Defined in: session/types.ts:110

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

Defined in: session/types.ts:116

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

Defined in: session/types.ts:100

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

Defined in: session/types.ts:90

Opens the session runtime and performs authentication if needed.

#### Returns

`Promise`&lt;`void`&gt;

***

### sendText()

> **sendText**(`text`): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:141

Submits a finalized text turn through the connected realtime session.

The promise resolves after a `finish_asr` action echoes the normalized
text with `origin` set to `text`.

#### Parameters

##### text

`string`

User-authored text for the next turn.

#### Returns

`Promise`&lt;`void`&gt;

***

### switchSession()

> **switchSession**(`sessionId`): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:158

Switches the active conversation to a persisted session or starts a new one.

#### Parameters

##### sessionId

`string` \| `null`

Target session identifier, or `null` to start a new session.

#### Returns

`Promise`&lt;`void`&gt;

***

### uploadFile()

> **uploadFile**(`file`, `endpoint?`): `Promise`&lt;`void`&gt;

Defined in: session/types.ts:148

Uploads a file into the current session context.

#### Parameters

##### file

`Blob`

File blob to upload.

##### endpoint?

`string` \| `URL`

Optional upload endpoint override.

#### Returns

`Promise`&lt;`void`&gt;
