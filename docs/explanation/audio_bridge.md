# Audio Bridge Model

*Experimental feature*

X-Talk can bridge multiple frontend sessions into one shared browser-side audio bus so that bots hear a continuous audio stream instead of only microphone input. The bridge is a frontend-only construct: the shared bus is created with `createAudioBridge()`, each bot session reads it through `inputConfig.mode = "web_bridge"`, user audio may optionally be captured directly by the bridge, and bot output audio is published back into the bridge so other bots can respond.

## When to use it

The web audio bridge fits a single browser page that needs to:

- connect multiple X-Talk sessions at the same time,
- let bots respond to other bots' spoken output,
- optionally inject real user microphone audio into the same shared stream,
- keep the server contract unchanged and do the routing on the frontend.

Because routing happens in the page, the server sees each bot as an ordinary session with an audio input. No server-side configuration distinguishes a bridged session from a microphone session.

## Shared stream model

The bridge maintains one continuous PCM stream at `16000` Hz.

- When nobody is speaking, the stream contains silence.
- When the user or a bot publishes audio, that audio is mixed into the stream.
- Every bot configured with `mode: "web_bridge"` keeps receiving frames from the same shared stream, except for audio chunks that bot published itself.

The bridge therefore avoids point-to-point routing of bot audio. All publishers write into one shared stream, all bridge-backed bot sessions read from it, and each bot's own published audio is filtered out of its local input stream. Self-filtering is what keeps a bot from treating its own reply as new input, while the shared mix is what lets any number of participants join without reconfiguring the others.

One consequence of a shared mix is that bots which publish their output back into the stream can keep answering each other without end; stopping the exchange is the page's responsibility.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Bot2Bot →](../tutorial/bot2bot.md)** · Tutorial

-   **[Audio Bridge: instance API and limitations →](../reference/audio_bridge.md)** · Reference

</div>
