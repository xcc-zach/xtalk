[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / InputAudioSessionConfig

# Interface: InputAudioSessionConfig

Defined in: bases/audio-session.ts:4

## Indexable

> \[`key`: `string`\]: `any`

## Properties

### autoEmitVad?

> `optional` **autoEmitVad?**: `boolean`

Defined in: bases/audio-session.ts:25

Whether the source should auto-broadcast frontend VAD when publishing output
back into a shared bridge stream.

***

### bridge?

> `optional` **bridge?**: `unknown`

Defined in: bases/audio-session.ts:20

Bridge instance consumed by bridge-backed input implementations.

***

### mode?

> `optional` **mode?**: `"microphone"` \| `"web_bridge"`

Defined in: bases/audio-session.ts:12

Selects the input source implementation.

***

### participantId?

> `optional` **participantId?**: `string`

Defined in: bases/audio-session.ts:16

Participant identifier used by bridge-backed input implementations.

***

### sampleRate

> **sampleRate**: `number`

Defined in: bases/audio-session.ts:8

Target input sample rate forwarded to the session runtime.

***

### vadRedemptionMs?

> `optional` **vadRedemptionMs?**: `number`

Defined in: bases/audio-session.ts:30

VAD redemption window used by bridge-backed input sources that emit frontend
VAD automatically.
