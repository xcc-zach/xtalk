[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / BaseEncoding

# 抽象类： BaseEncoding

定义于： bases/encoding.ts:3

## 构造函数

### 构造函数

> **new BaseEncoding**(): `BaseEncoding`

#### 返回值

`BaseEncoding`

## 方法

### decodeBase64()

> `abstract` **decodeBase64**(`base64`): `ArrayBuffer`

定义于： bases/encoding.ts:8

#### 参数

##### base64

`string`

#### 返回值

`ArrayBuffer`

***

### encodeBase64()

> `abstract` **encodeBase64**(`data`): `string`

定义于： bases/encoding.ts:7

将 PCM 字节编码以便通过 JSON 传输。

#### 参数

##### data

`ArrayBuffer`

#### 返回值

`string`

***

### encodePCM()

> `abstract` **encodePCM**(`data`, `sampleRate`, `targetRate`): `string`

定义于： bases/encoding.ts:5

将 PCM 音频转换到线路采样率后编码。

#### 参数

##### data

`ArrayBuffer`

##### sampleRate

`number`

##### targetRate

`number`

#### 返回值

`string`
