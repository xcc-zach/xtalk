[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / PlatformRuntime

# Type Alias: PlatformRuntime

> **PlatformRuntime** = `object`

Defined in: platforms/index.ts:39

Factories that supply every platform-specific resource used by a session.

Register a custom runtime with [registerPlatformRuntime](../functions/registerPlatformRuntime.md) to run the
client outside a browser, for example inside automated tests.

## Methods

### createDeferredTaskScheduler()

> **createDeferredTaskScheduler**(): [`BaseDeferredTaskScheduler`](../classes/BaseDeferredTaskScheduler.md)

Defined in: platforms/index.ts:49

Creates the scheduler used to defer observer callbacks.

#### Returns

[`BaseDeferredTaskScheduler`](../classes/BaseDeferredTaskScheduler.md)

***

### createEncoding()

> **createEncoding**(): [`BaseEncoding`](../classes/BaseEncoding.md)

Defined in: platforms/index.ts:51

Creates the Base64 and PCM encoder.

#### Returns

[`BaseEncoding`](../classes/BaseEncoding.md)

***

### createInputAudioSession()

> **createInputAudioSession**(`config`): [`BaseInputAudioSession`](../classes/BaseInputAudioSession.md)

Defined in: platforms/index.ts:41

Creates the microphone or bridge-backed audio source.

#### Parameters

##### config

[`InputAudioSessionConfig`](../interfaces/InputAudioSessionConfig.md)

#### Returns

[`BaseInputAudioSession`](../classes/BaseInputAudioSession.md)

***

### createOutputAudioSession()

> **createOutputAudioSession**(`config`): [`BaseOutputAudioSession`](../classes/BaseOutputAudioSession.md)

Defined in: platforms/index.ts:43

Creates the speaker-backed playback sink.

#### Parameters

##### config

[`OutputAudioSessionConfig`](../interfaces/OutputAudioSessionConfig.md)

#### Returns

[`BaseOutputAudioSession`](../classes/BaseOutputAudioSession.md)

***

### createWebSocket()

> **createWebSocket**(`url`, `protocols?`): [`BaseWebSocket`](../classes/BaseWebSocket.md)

Defined in: platforms/index.ts:45

Creates the transport used for the Realtime connection.

#### Parameters

##### url

`string` \| `URL`

##### protocols?

`string` \| `string`[]

#### Returns

[`BaseWebSocket`](../classes/BaseWebSocket.md)

***

### delay()

> **delay**(`milliseconds`): `Promise`&lt;`void`&gt;

Defined in: platforms/index.ts:47

Resolves after the given delay.

#### Parameters

##### milliseconds

`number`

#### Returns

`Promise`&lt;`void`&gt;
