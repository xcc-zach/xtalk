[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / SessionConfig

# Interface: SessionConfig

Defined in: session/types.ts:24

Configuration overrides used when creating a session.

## Properties

### apiKey?

> `optional` **apiKey?**: `string`

Defined in: session/types.ts:34

Optional Realtime API key; business credentials belong to the URL owner.

***

### inputConfig?

> `optional` **inputConfig?**: `Partial`&lt;`InputAudioSessionConfig`&gt;

Defined in: session/types.ts:28

Input audio session overrides such as capture sample rate.

***

### outputConfig?

> `optional` **outputConfig?**: `Partial`&lt;`OutputAudioSessionConfig`&gt;

Defined in: session/types.ts:32

Output audio session overrides such as playback sample rate.
