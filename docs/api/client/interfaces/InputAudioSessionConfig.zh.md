[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / InputAudioSessionConfig

# 接口： InputAudioSessionConfig

定义于： bases/audio-session.ts:4

## 索引签名

> \[`key`: `string`\]: `any`

## 属性

### autoEmitVad?

> `optional` **autoEmitVad?**: `boolean`

定义于： bases/audio-session.ts:25

将输出发布回共享桥接流时，输入源是否自动广播前端 VAD。

***

### bridge?

> `optional` **bridge?**: `unknown`

定义于： bases/audio-session.ts:20

桥接式输入实现所使用的桥接实例。

***

### mode?

> `optional` **mode?**: `"microphone"` \| `"web_bridge"`

定义于： bases/audio-session.ts:12

选择输入源实现。

***

### participantId?

> `optional` **participantId?**: `string`

定义于： bases/audio-session.ts:16

桥接式输入实现所使用的参与者标识。

***

### sampleRate

> **sampleRate**: `number`

定义于： bases/audio-session.ts:8

转交给会话运行时的目标输入采样率。

***

### vadRedemptionMs?

> `optional` **vadRedemptionMs?**: `number`

定义于： bases/audio-session.ts:30

自动发出前端 VAD 的桥接式输入源所使用的 VAD 回退窗口。
