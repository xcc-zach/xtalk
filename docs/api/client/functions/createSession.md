[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / createSession

# Function: createSession()

> **createSession**(`websocketURL`, `config?`): [`Session`](../interfaces/Session.md)

Defined in: session/create.ts:53

Creates a session client bound to the provided websocket endpoint.

The returned session coordinates authentication, runtime audio streaming,
message state synchronization, and persisted conversation restoration.

## Parameters

### websocketURL

`string` \| `URL`

Websocket endpoint used to establish the realtime session.

### config?

[`SessionConfig`](../interfaces/SessionConfig.md) = `{}`

Optional session configuration overrides.

## Returns

[`Session`](../interfaces/Session.md)

A session controller for opening, closing, and interacting with X-Talk.
