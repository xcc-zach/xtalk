[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / SessionConfig

# 接口： SessionConfig

定义于： session/types.ts:24

创建会话时使用的配置。

## 属性

### apiKey?

> `optional` **apiKey?**: `string`

定义于： session/types.ts:34

可选 Realtime API Key；业务凭证由 URL 的提供方管理。

***

### inputConfig?

> `optional` **inputConfig?**: `Partial`&lt;`InputAudioSessionConfig`&gt;

定义于： session/types.ts:28

输入音频配置，例如采集采样率。

***

### outputConfig?

> `optional` **outputConfig?**: `Partial`&lt;`OutputAudioSessionConfig`&gt;

定义于： session/types.ts:32

输出音频配置，例如播放采样率。
