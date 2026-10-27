# Connect a Client

This tutorial ends with a terminal client completing a spoken round trip against X-Talk: you will run the Realtime API as a standalone ASGI application and drive it with `examples/sample_client.py`, which speaks only WebSocket and PCM audio.

Before starting, complete [Configure the Service](config_the_service.md) so that a working `config.json` exists.

X-Talk exposes an OpenAI Realtime style API with a standard subset and X-Talk extensions. The browser client from [Start the Service](start_the_service.md) hides that protocol; this tutorial talks to it directly.

## Run the ASGI application

`create_asgi_app()` returns a plain ASGI application, so any ASGI server can run it without a web framework in between. Install one, for example `pip install uvicorn`, and create `run.py`:

```python
import uvicorn
from xtalk import Xtalk

runtime = Xtalk.from_config("path/to/config.json")
uvicorn.run(runtime.create_asgi_app(), host="127.0.0.1", port=8000)
```

Run `python run.py`. Uvicorn prints `Uvicorn running on http://127.0.0.1:8000` after the models load.

The application serves two endpoints: `GET /v1/models` and `WS /v1/realtime?model=xtalk`. Check the first one:

```bash
curl http://127.0.0.1:8000/v1/models
```

The response lists one model named `xtalk`.

## Set the model alias and API keys

Both values come from the configuration file:

```json
{"service_config": {"realtime_api": {"model": "xtalk", "api_keys": []}}}
```

The defaults are model `xtalk` and API key authentication disabled, so the file from the previous tutorial needs no change. Configure `api_keys` to require `Authorization: Bearer <key>`; browsers may use the `openai-insecure-api-key.<key>` WebSocket subprotocol instead.

## Run the minimal audio client

`examples/sample_client.py` handles WebSocket communication and audio input/output; VAD and enhancement run on the server.

```sh
pip install websockets numpy soxr soundfile sounddevice
python examples/sample_client.py --base-url http://127.0.0.1:8000 --model xtalk
```

Speak into the microphone. The client prints `conversation.item.input_audio_transcription.completed` followed by the recognized text, plays the reply, and prints `response.output_audio.playback.done` with the reply text once playback ends.

On a machine without a sound card the microphone stream fails to open. Feed a WAV file instead and skip playback:

```sh
python examples/sample_client.py --base-url http://127.0.0.1:8000 --model xtalk \
    --input-wav input.wav --no-speaker --rounds 2
```

The client prints `Completed audio round` after each of the two rounds and exits. Add `--api-key <key>` when `api_keys` is configured; without it the server rejects the connection.

The sample keeps one output stream open, fills consecutive protocol blocks continuously, and acknowledges each block after its final sample reaches the output device. Interruptions flush the stopped response and report only unconfirmed played audio.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Introduce a New Model →](introduce_a_new_model.md)** · Tutorial

-   **[Realtime API Protocol: events, fields and audio formats →](../reference/realtime_api.md)** · Reference

-   **[Mount in FastAPI →](../how_to/mount_in_fastapi.md)** · How-to

</div>
