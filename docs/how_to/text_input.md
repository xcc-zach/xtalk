# Text Input

This guide sends a typed user turn through an open `Session` from the Web client SDK; it assumes the client from [Start the Service](../tutorial/start_the_service.md) is already connecting.

After connecting a Session, send text with `sendText()`:

```ts
const session = createSession(websocketURL, config);

await session.open();
await session.sendText("Set a timer for two seconds.");
```

The text is submitted as a complete user turn and continues through the existing Agent, tool-calling, and TTS pipeline. It can be sent while the Session is `idle`, `listening`, `processing`, or `speaking`. If recognition, generation, or response playback is in progress, the new text replaces the current turn.

Sending text does not stop or recreate microphone capture.

## Usage Limits

- The Session must be connected.
- Text must not be empty and may contain at most 2,048 JavaScript characters.
- Only one text submission may await confirmation in a Session.
- The confirmation timeout is 10 seconds.
- Disconnecting, closing, or switching the Session cancels a pending submission.
- The client does not queue or retry submissions automatically.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Session API →](../api/client/interfaces/Session.md)** · Reference

-   **[Realtime API Protocol →](../reference/realtime_api.md)** · Reference

-   **[Connect a Client →](../tutorial/connect_a_client.md)** · Tutorial

</div>
