[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / BaseEncoding

# Abstract Class: BaseEncoding

Defined in: bases/encoding.ts:3

## Constructors

### Constructor

> **new BaseEncoding**(): `BaseEncoding`

#### Returns

`BaseEncoding`

## Methods

### decodeBase64()

> `abstract` **decodeBase64**(`base64`): `ArrayBuffer`

Defined in: bases/encoding.ts:8

#### Parameters

##### base64

`string`

#### Returns

`ArrayBuffer`

***

### encodeBase64()

> `abstract` **encodeBase64**(`data`): `string`

Defined in: bases/encoding.ts:7

Encodes PCM bytes for JSON transport.

#### Parameters

##### data

`ArrayBuffer`

#### Returns

`string`

***

### encodePCM()

> `abstract` **encodePCM**(`data`, `sampleRate`, `targetRate`): `string`

Defined in: bases/encoding.ts:5

Encodes PCM audio after conversion to the wire sample rate.

#### Parameters

##### data

`ArrayBuffer`

##### sampleRate

`number`

##### targetRate

`number`

#### Returns

`string`
