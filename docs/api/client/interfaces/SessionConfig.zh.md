[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / SessionConfig

# 接口： SessionConfig

定义于： session/types.ts:24

创建会话时使用的配置。

## 属性

### apiKey?

> `optional` **apiKey?**: `string`

定义于： session/types.ts:36

可选 Realtime API Key；业务凭证由 URL 的提供方管理。

***

### inputConfig?

> `optional` **inputConfig?**: `Partial`&lt;[`InputAudioSessionConfig`](InputAudioSessionConfig.zh.md)&gt;

定义于： session/types.ts:30

输入音频配置，例如采集采样率。

***

### outputAudioEncoding?

> `optional` **outputAudioEncoding?**: `"base64"` \| `"binary_pcm"`

定义于： session/types.ts:26

请求二进制 PCM 输出；默认为兼容的 Base64 JSON。

***

### outputConfig?

> `optional` **outputConfig?**: `Partial`&lt;[`OutputAudioSessionConfig`](OutputAudioSessionConfig.zh.md)&gt;

定义于： session/types.ts:34

输出音频配置，例如播放采样率。
