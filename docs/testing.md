You can input audios to test X-Talk with `scripts/offline_client.py`.

First start an X-Talk server, remember the port, like 7634; in the server config file, add the snippet below to enable recording:
```json
"service_config": {
    "recording": true
}
```

Then install dependencies for the offline client and prepare an audio directory with audio files and a `timestamp.txt` file for testing:

```bash
pip install websockets soundfile numpy soxr
```
```plaintext
/path/to/audio_dir/
├── audio1.wav
├── audio2.wav
└── timestamp.txt

timestamp.txt content:
audio1.wav:0
audio2.wav:on_response_finish
```

Then run the offline client with the WebSocket URL of the server and the input audio directory:

```bash
    python scripts/offline_client.py --ws ws://127.0.0.1:8000/ws --with-vad --input /path/to/audio_dir --output /path/to/recording.wav
```

You will see result in `/path/to/recording.wav`. See `scripts/offline_client.py` for more details.
