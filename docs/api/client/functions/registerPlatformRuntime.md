[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / registerPlatformRuntime

# Function: registerPlatformRuntime()

> **registerPlatformRuntime**(`runtime`): `void`

Defined in: platforms/index.ts:64

Use an explicit platform runtime instead of browser detection.

Call this before creating a session. A later call replaces the runtime for
sessions created afterwards; sessions already created keep their resources.

## Parameters

### runtime

[`PlatformRuntime`](../type-aliases/PlatformRuntime.md)

Platform runtime supplying every session resource.

## Returns

`void`
