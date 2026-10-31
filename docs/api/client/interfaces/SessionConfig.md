[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / SessionConfig

# Interface: SessionConfig

Defined in: session/types.ts:24

Configuration overrides used when creating a session.

## Properties

### apiKey?

> `optional` **apiKey?**: `string`

Defined in: session/types.ts:36

Optional Realtime API key; business credentials belong to the URL owner.

***

### inputConfig?

> `optional` **inputConfig?**: `Partial`&lt;[`InputAudioSessionConfig`](InputAudioSessionConfig.md)&gt;

Defined in: session/types.ts:30

Input audio session overrides such as capture sample rate.

***

### outputAudioEncoding?

> `optional` **outputAudioEncoding?**: `"base64"` \| `"binary_pcm"`

Defined in: session/types.ts:26

Request binary PCM output; defaults to compatible Base64 JSON.

***

### outputConfig?

> `optional` **outputConfig?**: `Partial`&lt;[`OutputAudioSessionConfig`](OutputAudioSessionConfig.md)&gt;

Defined in: session/types.ts:34

Output audio session overrides such as playback sample rate.
